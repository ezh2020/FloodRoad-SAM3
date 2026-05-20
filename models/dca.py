from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam3_baseline import SAM3Adapter


class DifferentialConceptAnchor(nn.Module):
    """DCA prompt fusion for flooded-road concept scoring.

    The class extracts positive road exemplars from pre-disaster imagery and
    negative water-like exemplars, then computes a differential score map over
    post-disaster features. The implementation works with generic SAM-style
    feature maps; when a backend exposes stronger prompt APIs, this module can be
    upgraded locally without changing train/evaluate scripts.
    """

    def __init__(
        self,
        sam: SAM3Adapter,
        k_pos: int = 8,
        k_neg: int = 4,
        patch_size: int = 64,
        text_pos: str = "road obstructed by water",
        text_neg: str = "road still passable",
    ) -> None:
        super().__init__()
        self.sam = sam
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.patch_size = patch_size
        self.text_pos = text_pos
        self.text_neg = text_neg

    def forward(
        self,
        post_image: torch.Tensor,
        pre_image: torch.Tensor,
        road_mask: torch.Tensor,
        road_buffer: torch.Tensor,
        post_features: torch.Tensor | None = None,
        pre_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, h, w = post_image.shape
        if post_features is None:
            post_features = self.sam.encode_image(post_image)
        if post_features.ndim != 4:
            raise ValueError("DCA expects 4D image feature maps from the SAM3 adapter")

        if pre_features is not None:
            pos_embs, neg_embs = [], []
            for i in range(b):
                neg_mask = estimate_water_like_mask(pre_image[i], road_mask[i, 0])
                pos_embs.append(
                    sample_feature_embeddings(pre_features[i], road_mask[i, 0], self.k_pos, self.patch_size, (h, w))
                )
                neg_embs.append(
                    sample_feature_embeddings(pre_features[i], neg_mask, self.k_neg, self.patch_size, (h, w))
                )
            pos_emb = torch.cat(pos_embs, dim=0)
            neg_emb = torch.cat(neg_embs, dim=0)
        else:
            patches_pos, patches_neg = [], []
            for i in range(b):
                patches_pos.append(sample_patches(pre_image[i], road_mask[i, 0], self.k_pos, self.patch_size))
                neg_mask = estimate_water_like_mask(pre_image[i], road_mask[i, 0])
                patches_neg.append(sample_patches(pre_image[i], neg_mask, self.k_neg, self.patch_size))
            pos = torch.cat(patches_pos, dim=0)
            neg = torch.cat(patches_neg, dim=0)
            pos_emb = self.sam.encode_patch_embedding(pos)
            neg_emb = self.sam.encode_patch_embedding(neg)

        feature_map = F.normalize(post_features, dim=1)
        pos_emb = match_dim(pos_emb, feature_map.shape[1])
        neg_emb = match_dim(neg_emb, feature_map.shape[1])
        pos_emb = pos_emb.view(b, self.k_pos, -1)
        neg_emb = neg_emb.view(b, self.k_neg, -1)

        phi_pos_img = torch.einsum("bchw,bkc->bkhw", feature_map, pos_emb).mean(dim=1, keepdim=True)
        phi_neg_img = torch.einsum("bchw,bkc->bkhw", feature_map, neg_emb).mean(dim=1, keepdim=True)

        text_emb = self.sam.encode_text([self.text_pos, self.text_neg], device=post_image.device)
        text_emb = match_dim(text_emb, feature_map.shape[1])
        phi_pos_text = torch.einsum("bchw,c->bhw", feature_map, text_emb[0]).unsqueeze(1)
        phi_neg_text = torch.einsum("bchw,c->bhw", feature_map, text_emb[1]).unsqueeze(1)

        phi_flood = 0.5 * phi_pos_text + 0.5 * phi_pos_img
        phi_passable = 0.5 * phi_neg_text + 0.5 * phi_neg_img
        logits_low = phi_flood - phi_passable
        logits = F.interpolate(logits_low, size=(h, w), mode="bilinear", align_corners=False)
        logits = logits + (road_buffer <= 0.5).float() * -20.0
        return logits, {"post_features": post_features, "pos_emb": pos_emb, "neg_emb": neg_emb}


def sample_patches(image: torch.Tensor, mask: torch.Tensor, count: int, patch_size: int) -> torch.Tensor:
    _, h, w = image.shape
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing="ij"
            ),
            dim=-1,
        ).view(-1, 2)
    if coords.shape[0] >= count:
        chosen = coords[torch.randperm(coords.shape[0], device=image.device)[:count]]
    else:
        idx = torch.randint(0, coords.shape[0], (count,), device=image.device)
        chosen = coords[idx]
    pad = patch_size // 2
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    patches = []
    for yx in chosen:
        y = int(yx[0].item())
        x = int(yx[1].item())
        patches.append(padded[:, y : y + patch_size, x : x + patch_size])
    return torch.stack(patches, dim=0)


def sample_feature_embeddings(
    feature: torch.Tensor,
    mask: torch.Tensor,
    count: int,
    patch_size: int,
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    c, hf, wf = feature.shape
    image_h, image_w = image_hw
    mask_f = F.interpolate(mask.float().view(1, 1, image_h, image_w), size=(hf, wf), mode="nearest")[0, 0]
    coords = torch.nonzero(mask_f > 0.5, as_tuple=False)
    if coords.numel() == 0:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(hf, device=feature.device), torch.arange(wf, device=feature.device), indexing="ij"
            ),
            dim=-1,
        ).view(-1, 2)
    if coords.shape[0] >= count:
        chosen = coords[torch.randperm(coords.shape[0], device=feature.device)[:count]]
    else:
        idx = torch.randint(0, coords.shape[0], (count,), device=feature.device)
        chosen = coords[idx]
    rad_y = max(1, int(round((patch_size / max(image_h, 1)) * hf / 2)))
    rad_x = max(1, int(round((patch_size / max(image_w, 1)) * wf / 2)))
    pooled = []
    for yx in chosen:
        y = int(yx[0].item())
        x = int(yx[1].item())
        y0, y1 = max(0, y - rad_y), min(hf, y + rad_y + 1)
        x0, x1 = max(0, x - rad_x), min(wf, x + rad_x + 1)
        pooled.append(feature[:, y0:y1, x0:x1].mean(dim=(-2, -1)))
    return F.normalize(torch.stack(pooled, dim=0), dim=-1)


def estimate_water_like_mask(pre_image: torch.Tensor, road_mask: torch.Tensor) -> torch.Tensor:
    # Input is normalized RGB. A simple blue/green heuristic is a fallback for
    # RGB-only SpaceNet tiles where NIR is unavailable.
    rgb = pre_image
    green = rgb[1]
    blue = rgb[2]
    red = rgb[0]
    score = (blue + green) * 0.5 - red
    threshold = torch.quantile(score.flatten(), 0.8)
    return ((score > threshold) & (road_mask < 0.5)).float()


def match_dim(emb: torch.Tensor, dim: int) -> torch.Tensor:
    emb = emb.float()
    if emb.shape[-1] == dim:
        return F.normalize(emb, dim=-1)
    if emb.shape[-1] > dim:
        emb = emb[..., :dim]
    else:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return F.normalize(emb, dim=-1)
