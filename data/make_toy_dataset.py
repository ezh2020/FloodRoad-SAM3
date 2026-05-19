from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny processed FloodRoad dataset for pipeline smoke tests.")
    parser.add_argument("--processed-root", default="/content/spacenet8/toy_processed")
    parser.add_argument("--num-tiles", type=int, default=24)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def draw_segment(mask: np.ndarray, segment_map: np.ndarray, sid: int, y: int, x0: int, x1: int, width: int) -> None:
    half = max(1, width // 2)
    y0 = max(0, y - half)
    y1 = min(mask.shape[0], y + half + 1)
    xa = max(0, min(x0, x1))
    xb = min(mask.shape[1], max(x0, x1) + 1)
    mask[y0:y1, xa:xb] = 1
    segment_map[y0:y1, xa:xb] = sid


def make_tile(tile_id: str, out_dir: Path, size: int, rng: random.Random) -> Dict[str, object]:
    pre = np.zeros((size, size, 3), dtype=np.uint8)
    post = np.zeros((size, size, 3), dtype=np.uint8)
    pre[..., :] = np.array([70, 95, 80], dtype=np.uint8)
    post[..., :] = np.array([65, 90, 78], dtype=np.uint8)

    road_mask = np.zeros((size, size), dtype=np.uint8)
    flood_mask = np.zeros((size, size), dtype=np.uint8)
    segment_map = np.zeros((size, size), dtype=np.int32)
    graph = nx.Graph()

    widths = [6, 7, 8, 6]
    ys = [int(size * f) for f in (0.25, 0.42, 0.60, 0.76)]
    flooded_segments = set(rng.sample(range(1, len(ys) + 1), k=2))
    for sid, (y, width) in enumerate(zip(ys, widths), start=1):
        x0 = rng.randint(4, 12)
        x1 = size - rng.randint(8, 18)
        draw_segment(road_mask, segment_map, sid, y, x0, x1, width)
        pix = segment_map == sid
        pre[pix] = np.array([120, 120, 115], dtype=np.uint8)
        post[pix] = np.array([118, 118, 112], dtype=np.uint8)
        if sid in flooded_segments:
            flooded_slice = pix.copy()
            flood_mask[flooded_slice] = 1
            post[flooded_slice] = np.array([40, 95, 155], dtype=np.uint8)
        graph.add_node(
            sid,
            label=int(sid in flooded_segments),
            flooded_ratio=float(sid in flooded_segments),
            length=float(x1 - x0),
            width=float(width),
            pixel_count=int(pix.sum()),
        )

    for sid in range(1, len(ys)):
        graph.add_edge(sid, sid + 1)

    flooded_road_mask = (road_mask & flood_mask).astype(np.uint8)
    tile_dir = out_dir / "tiles" / tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "pre_path": tile_dir / "pre.npy",
        "post_path": tile_dir / "post.npy",
        "road_mask_path": tile_dir / "road_mask.npy",
        "flood_mask_path": tile_dir / "flood_mask.npy",
        "mask_path": tile_dir / "flooded_road_mask.npy",
        "segment_map_path": tile_dir / "segment_map.npy",
    }
    arrays = {
        "pre_path": pre,
        "post_path": post,
        "road_mask_path": road_mask,
        "flood_mask_path": flood_mask,
        "mask_path": flooded_road_mask,
        "segment_map_path": segment_map,
    }
    for key, arr in arrays.items():
        np.save(paths[key], arr)
    graph_path = tile_dir / "graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(graph), f, ensure_ascii=True)

    return {
        "id": tile_id,
        "scene_id": "toy",
        "x": 0,
        "y": 0,
        "graph_path": str(graph_path),
        **{key: str(path) for key, path in paths.items()},
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = ensure_dir(args.processed_root)
    rows: List[Dict[str, object]] = []
    for idx in range(args.num_tiles):
        row = make_tile(f"toy_{idx:03d}", out_dir, args.tile_size, rng)
        row["split"] = "train" if idx < int(args.num_tiles * 0.8) else "val"
        rows.append(row)
    manifest = out_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    with open(out_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump({"num_tiles": len(rows), "manifest": str(manifest), "toy": True}, f, indent=2)
    print(f"Wrote {len(rows)} toy tiles to {manifest}")


if __name__ == "__main__":
    main()
