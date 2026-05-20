from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ccrl import SegmentPolicyNet, build_segment_states, mask_from_segment_actions, rollout_policy
from .dca import DifferentialConceptAnchor
from .lora import apply_lora, lora_parameters
from .rgstm import RoadGraphTokenMerging
from .sam3_baseline import SAM3Adapter, build_sam3_adapter


class FloodRoadSAM3(nn.Module):
    def __init__(self, sam_cfg: Dict, ours_cfg: Dict, use_token_merging: bool = False) -> None:
        super().__init__()
        self.sam = build_sam3_adapter(sam_cfg)
        self.use_token_merging = use_token_merging
        for param in self.sam.parameters():
            param.requires_grad = False
        lora_cfg = ours_cfg.get("lora", {})
        self.lora_layers: List[str] = []
        if lora_cfg.get("enabled", True):
            self.lora_layers = apply_lora(
                self.sam.image_encoder,
                target_keywords=lora_cfg.get("target_keywords", ["q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj"]),
                rank=int(lora_cfg.get("rank", 8)),
                alpha=int(lora_cfg.get("alpha", 16)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
            )
        dca_cfg = ours_cfg.get("dca", {})
        self.dca = DifferentialConceptAnchor(
            self.sam,
            k_pos=int(dca_cfg.get("k_pos", 8)),
            k_neg=int(dca_cfg.get("k_neg", 4)),
            patch_size=int(dca_cfg.get("patch_size", 64)),
            text_pos=dca_cfg.get("text_pos", "road obstructed by water"),
            text_neg=dca_cfg.get("text_neg", "road still passable"),
        )
        rgstm_cfg = ours_cfg.get("rgstm", {})
        self.rgstm = RoadGraphTokenMerging(
            merge_ratio=float(rgstm_cfg.get("merge_ratio", 0.2)),
            laplacian_k=int(rgstm_cfg.get("laplacian_k", 16)),
            weights=rgstm_cfg.get("preserve_weights"),
        )
        ccrl_cfg = ours_cfg.get("ccrl", {})
        self.policy = SegmentPolicyNet(
            state_dim=int(ccrl_cfg.get("state_dim", 12)),
            hidden_dim=int(ccrl_cfg.get("hidden_dim", 256)),
            num_actions=int(ccrl_cfg.get("num_actions", 3)),
        )

    def trainable_lora_parameters(self):
        return lora_parameters(self.sam)

    def forward_segmentation(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        post = batch.get("post_raw", batch["post"])
        pre = batch.get("pre_raw", batch["pre"])
        road_mask = batch["road_mask"]
        road_buffer = batch["road_buffer"]
        features = self.sam.encode_image(post)
        if self.use_token_merging:
            if post.shape[0] != 1:
                raise ValueError("Token merging path expects batch size 1")
            features, merge_state = self.rgstm(features, batch["segment_map"], batch["graph"][0], batch["post_raw"])
        else:
            merge_state = None
        logits, aux = self.dca(post, pre, road_mask, road_buffer, post_features=features)
        aux["merge_state"] = merge_state
        return logits, aux

    def forward_policy_supervised(self, batch: Dict, score_map: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        if score_map.shape[0] != 1:
            raise ValueError("CC-RL policy expects batch size 1")
        states, node_ids, _ = build_segment_states(
            score_map[0, 0],
            batch["segment_map"][0],
            batch["graph"][0],
            batch["post_raw"][0],
            batch["pre_raw"][0],
        )
        return self.policy(states), node_ids

    def rollout(self, batch: Dict, score_map: torch.Tensor, sample: bool = True):
        if score_map.shape[0] != 1:
            raise ValueError("CC-RL rollout expects batch size 1")
        return rollout_policy(
            self.policy,
            score_map[0, 0],
            batch["segment_map"][0],
            batch["graph"][0],
            batch["post_raw"][0],
            batch["pre_raw"][0],
            sample=sample,
        )

    @torch.no_grad()
    def predict_mask(self, batch: Dict, threshold: float = 0.5) -> torch.Tensor:
        logits, _ = self.forward_segmentation(batch)
        score = torch.sigmoid(logits)
        if "graph" not in batch:
            return (score >= threshold).float()
        masks = []
        for i, graph in enumerate(batch["graph"]):
            single = {
                "segment_map": batch["segment_map"][i : i + 1],
                "post_raw": batch["post_raw"][i : i + 1],
                "pre_raw": batch["pre_raw"][i : i + 1],
                "graph": [graph],
            }
            rollout = rollout_policy(
                self.policy,
                score[i, 0],
                single["segment_map"][0],
                graph,
                single["post_raw"][0],
                single["pre_raw"][0],
                sample=False,
            )
            mask = mask_from_segment_actions(single["segment_map"][0], rollout.node_ids, rollout.actions, score[i, 0])
            masks.append(mask.unsqueeze(0))
        return torch.stack(masks, dim=0)


def build_floodroad_sam3(cfg: Dict, use_token_merging: bool) -> FloodRoadSAM3:
    return FloodRoadSAM3(cfg.get("sam3", {}), cfg.get("ours", {}), use_token_merging=use_token_merging)
