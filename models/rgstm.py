from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MergeState:
    assignment: torch.Tensor
    original_hw: Tuple[int, int]
    keep_mask: torch.Tensor


class RoadGraphTokenMerging(nn.Module):
    """Road-graph preserving spatial token merging for feature maps.

    This module exposes merge/unmerge operations. Hooking into exact SAM3 transformer
    layers depends on the official backend; the integrated model applies it to the
    encoded feature map as a conservative approximation while preserving the same
    road/Laplacian scoring logic and measuring the method variant separately.
    """

    def __init__(self, merge_ratio: float = 0.2, laplacian_k: int = 16, weights: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.merge_ratio = float(merge_ratio)
        self.laplacian_k = int(laplacian_k)
        self.weights = weights or {"energy": 0.4, "laplacian": 0.4, "edge": 0.2}

    def forward(
        self,
        features: torch.Tensor,
        segment_map: torch.Tensor,
        graph: nx.Graph,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, MergeState]:
        if self.merge_ratio <= 0:
            n = features.shape[-2] * features.shape[-1]
            assignment = torch.arange(n, device=features.device)
            keep_mask = torch.ones(n, dtype=torch.bool, device=features.device)
            return features, MergeState(assignment=assignment, original_hw=features.shape[-2:], keep_mask=keep_mask)

        b, c, h, w = features.shape
        if b != 1:
            # Keep batch handling explicit because segment graphs are per tile.
            raise ValueError("RG-STM currently expects batch size 1")
        preserve = self.preserve_scores(features[0], segment_map[0], graph, image[0])
        n = h * w
        merge_n = int(round(n * self.merge_ratio))
        keep_n = max(1, n - merge_n)
        flat = features[0].flatten(1).transpose(0, 1)  # N, C
        scores = preserve.flatten()
        keep_idx = torch.topk(scores, k=keep_n, largest=True).indices
        keep_mask = torch.zeros(n, dtype=torch.bool, device=features.device)
        keep_mask[keep_idx] = True
        merge_idx = torch.nonzero(~keep_mask, as_tuple=False).flatten()

        kept = flat[keep_idx]
        assignment = torch.empty(n, dtype=torch.long, device=features.device)
        assignment[keep_idx] = torch.arange(keep_n, device=features.device)
        if merge_idx.numel() > 0:
            sim = F.normalize(flat[merge_idx], dim=-1) @ F.normalize(kept, dim=-1).t()
            assigned = sim.argmax(dim=1)
            assignment[merge_idx] = assigned
            accum = kept.clone()
            counts = torch.ones(keep_n, device=features.device)
            accum.index_add_(0, assigned, flat[merge_idx])
            counts.index_add_(0, assigned, torch.ones_like(assigned, dtype=torch.float32))
            kept = accum / counts.unsqueeze(1)

        unmerged = kept[assignment].transpose(0, 1).view(1, c, h, w)
        return unmerged, MergeState(assignment=assignment, original_hw=(h, w), keep_mask=keep_mask)

    def preserve_scores(self, feature: torch.Tensor, segment_map: torch.Tensor, graph: nx.Graph, image: torch.Tensor) -> torch.Tensor:
        _, h, w = feature.shape
        energy = feature.norm(dim=0)
        energy = normalize01(energy)

        lap = laplacian_pixel_score(segment_map, graph, h, w, self.laplacian_k).to(feature.device)
        lap = normalize01(lap)

        edge = sobel_edge_score(image, h, w).to(feature.device)
        edge = normalize01(edge)

        return (
            self.weights.get("energy", 0.4) * energy
            + self.weights.get("laplacian", 0.4) * lap
            + self.weights.get("edge", 0.2) * edge
        )


def normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def laplacian_pixel_score(segment_map: torch.Tensor, graph: nx.Graph, h: int, w: int, k: int) -> torch.Tensor:
    seg_np = segment_map.detach().cpu().numpy().astype(np.int64)
    if seg_np.shape != (h, w):
        seg_np = cv2.resize(seg_np, (w, h), interpolation=cv2.INTER_NEAREST)
    nodes = list(graph.nodes())
    if not nodes:
        return torch.zeros((h, w), dtype=torch.float32)
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}
    if graph.number_of_edges() == 0 or len(nodes) == 1:
        values = np.ones(len(nodes), dtype=np.float32)
    else:
        lap = nx.normalized_laplacian_matrix(graph, nodelist=nodes).astype(np.float32).toarray()
        eigvals, eigvecs = np.linalg.eigh(lap)
        kk = min(k, eigvecs.shape[1])
        vals = eigvals[:kk]
        vecs = eigvecs[:, :kk]
        values = (np.abs(vecs) / np.maximum(vals[None, :], 1e-3)).sum(axis=1).astype(np.float32)
    score = np.zeros((h, w), dtype=np.float32)
    for sid in np.unique(seg_np):
        if sid <= 0:
            continue
        idx = node_to_idx.get(int(sid))
        if idx is not None:
            score[seg_np == sid] = values[idx]
    return torch.from_numpy(score)


def sobel_edge_score(image: torch.Tensor, h: int, w: int) -> torch.Tensor:
    img = image.detach().cpu().float()
    if img.ndim == 3:
        gray = img.mean(dim=0).numpy()
    else:
        gray = img.numpy()
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    if edge.shape != (h, w):
        edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(edge.astype(np.float32))

