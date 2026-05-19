from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import read_jsonl


class FloodRoadDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        ids: Optional[Sequence[str]] = None,
        max_items: Optional[int] = None,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
        road_buffer_px: int = 16,
        load_graph: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        rows = read_jsonl(self.manifest_path)
        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        if ids is not None:
            wanted = set(ids)
            rows = [r for r in rows if r["id"] in wanted]
        if max_items is not None:
            rows = rows[:max_items]
        self.rows = rows
        self.mean = torch.tensor(normalize_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalize_std, dtype=torch.float32).view(3, 1, 1)
        self.road_buffer_px = int(road_buffer_px)
        self.load_graph = load_graph

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        pre = np.load(row["pre_path"])
        post = np.load(row["post_path"])
        road_mask = np.load(row["road_mask_path"]).astype(np.uint8)
        mask = np.load(row["mask_path"]).astype(np.uint8)
        segment_map = np.load(row["segment_map_path"]).astype(np.int32)
        road_buffer = dilate_binary(road_mask, self.road_buffer_px)

        post_t = image_to_tensor(post, self.mean, self.std)
        pre_t = image_to_tensor(pre, self.mean, self.std)
        raw_post_t = torch.from_numpy(post.astype(np.float32) / 255.0).permute(2, 0, 1)
        raw_pre_t = torch.from_numpy(pre.astype(np.float32) / 255.0).permute(2, 0, 1)

        item = {
            "id": row["id"],
            "post": post_t,
            "pre": pre_t,
            "post_raw": raw_post_t,
            "pre_raw": raw_pre_t,
            "mask": torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            "road_mask": torch.from_numpy(road_mask.astype(np.float32)).unsqueeze(0),
            "road_buffer": torch.from_numpy(road_buffer.astype(np.float32)).unsqueeze(0),
            "segment_map": torch.from_numpy(segment_map.astype(np.int64)),
            "meta": row,
        }
        if self.load_graph:
            item["graph"] = load_graph(row["graph_path"])
        return item


def image_to_tensor(img: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - mean) / std


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.uint8)


def load_graph(path: str | Path) -> nx.Graph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def collate_floodroad(batch: List[Dict]) -> Dict:
    tensor_keys = [
        "post",
        "pre",
        "post_raw",
        "pre_raw",
        "mask",
        "road_mask",
        "road_buffer",
        "segment_map",
    ]
    out: Dict = {}
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["id"] = [b["id"] for b in batch]
    out["meta"] = [b["meta"] for b in batch]
    if "graph" in batch[0]:
        out["graph"] = [b["graph"] for b in batch]
    return out


def segment_labels_from_graph(graph: nx.Graph) -> Dict[int, int]:
    return {int(n): int(d.get("label", 0)) for n, d in graph.nodes(data=True)}

