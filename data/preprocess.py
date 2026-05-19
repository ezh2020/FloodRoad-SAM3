from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import ensure_dir, load_config, save_json, write_jsonl


@dataclass
class SceneRecord:
    scene_id: str
    pre_path: Path
    post_path: Path
    road_geojson_path: Path
    flood_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SpaceNet 8 tiles for FloodRoad-SAM3.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--pairs-csv", default=None, help="CSV with id,pre_path,post_path,road_geojson_path,flood_path")
    parser.add_argument("--limit-scenes", type=int, default=None)
    return parser.parse_args()


def _as_path(root: Path, value: str | os.PathLike[str] | None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def read_pairs_csv(path: Path, raw_root: Path) -> List[SceneRecord]:
    records: List[SceneRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene_id = row.get("id") or row.get("scene_id") or Path(row["post_path"]).stem
            flood = _as_path(raw_root, row.get("flood_path"))
            records.append(
                SceneRecord(
                    scene_id=scene_id,
                    pre_path=_as_path(raw_root, row["pre_path"]),
                    post_path=_as_path(raw_root, row["post_path"]),
                    road_geojson_path=_as_path(raw_root, row["road_geojson_path"]),
                    flood_path=flood,
                )
            )
    return records


def discover_records(raw_root: Path, cfg: Dict) -> List[SceneRecord]:
    inputs = cfg.get("inputs", {})
    pre_files = sorted(raw_root.glob(inputs.get("pre_glob", "**/*pre*.tif")))
    post_files = sorted(raw_root.glob(inputs.get("post_glob", "**/*post*.tif")))
    road_files = sorted(raw_root.glob(inputs.get("road_geojson_glob", "**/*road*.geojson")))
    flood_files = sorted(raw_root.glob(inputs.get("flood_label_glob", "**/*flood*.tif")))

    if not post_files:
        raise FileNotFoundError(f"No post-disaster rasters found under {raw_root}")
    if not pre_files:
        raise FileNotFoundError(f"No pre-disaster rasters found under {raw_root}")
    if not road_files:
        raise FileNotFoundError(f"No road GeoJSON files found under {raw_root}")

    def key(path: Path) -> str:
        stem = path.stem.lower()
        for token in ["pre", "post", "roads", "road", "flood", "label", "mask"]:
            stem = stem.replace(token, "")
        return "".join(ch for ch in stem if ch.isalnum())

    pre_by_key = {key(p): p for p in pre_files}
    road_by_key = {key(p): p for p in road_files}
    flood_by_key = {key(p): p for p in flood_files}

    records: List[SceneRecord] = []
    for post in post_files:
        k = key(post)
        pre = pre_by_key.get(k) or min(pre_files, key=lambda p: _name_distance(post.name, p.name))
        road = road_by_key.get(k) or min(road_files, key=lambda p: _name_distance(post.name, p.name))
        flood = flood_by_key.get(k)
        if flood is None and flood_files:
            flood = min(flood_files, key=lambda p: _name_distance(post.name, p.name))
        records.append(SceneRecord(k or post.stem, pre, post, road, flood))
    return records


def _name_distance(a: str, b: str) -> int:
    aset = set(a.lower().replace("_", "").replace("-", ""))
    bset = set(b.lower().replace("_", "").replace("-", ""))
    return len(aset.symmetric_difference(bset))


def read_rgb_window(dataset: rasterio.DatasetReader, window: Window, tile_size: int) -> np.ndarray:
    channels = min(3, dataset.count)
    arr = dataset.read(indexes=list(range(1, channels + 1)), window=window, boundless=True, fill_value=0)
    if channels == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.shape[0] < 3:
        arr = np.concatenate([arr, np.zeros((3 - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)
    arr = np.transpose(arr[:3], (1, 2, 0))
    if arr.shape[:2] != (tile_size, tile_size):
        padded = np.zeros((tile_size, tile_size, 3), dtype=arr.dtype)
        padded[: arr.shape[0], : arr.shape[1]] = arr
        arr = padded
    return normalize_uint8(arr)


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def rasterize_roads(
    gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform,
    default_width_px: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict]]:
    road_mask = np.zeros(out_shape, dtype=np.uint8)
    segment_map = np.zeros(out_shape, dtype=np.int32)
    segment_meta: Dict[int, Dict] = {}

    for idx, row in enumerate(gdf.itertuples(), start=1):
        geom = getattr(row, "geometry")
        if geom is None or geom.is_empty:
            continue
        width = getattr(row, "width", None) or getattr(row, "road_width", None) or default_width_px
        try:
            width_i = max(1, int(round(float(width))))
        except (TypeError, ValueError):
            width_i = default_width_px
        burned = features.rasterize(
            [(geom, idx)], out_shape=out_shape, transform=transform, fill=0, dtype="int32", all_touched=True
        )
        if width_i > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_i, width_i))
            burned = cv2.dilate((burned > 0).astype(np.uint8), kernel, iterations=1).astype(np.int32) * idx
        road_mask[burned > 0] = 1
        segment_map[burned > 0] = idx
        segment_meta[idx] = {
            "geometry_wkt": geom.wkt,
            "width": float(width_i),
            "length": float(geom.length),
            "properties": {k: _jsonable(getattr(row, k)) for k in getattr(row, "_fields", []) if k != "geometry"},
        }
    return road_mask, segment_map, segment_meta


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        if np.isscalar(value):
            return value.item()
    except Exception:
        pass
    return str(value)


def rasterize_flood(
    flood_path: Optional[Path],
    out_shape: Tuple[int, int],
    transform,
    threshold: float,
    road_gdf: gpd.GeoDataFrame,
    window: Optional[Window] = None,
) -> np.ndarray:
    if flood_path is not None and flood_path.exists():
        if flood_path.suffix.lower() in {".geojson", ".json"}:
            flood_gdf = gpd.read_file(flood_path)
            return features.rasterize(
                [(geom, 1) for geom in flood_gdf.geometry if geom is not None and not geom.is_empty],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=True,
            )
        with rasterio.open(flood_path) as src:
            if window is not None:
                arr = src.read(1, window=window, boundless=True, fill_value=0, out_shape=out_shape, resampling=Resampling.nearest)
            else:
                arr = src.read(1, out_shape=out_shape, resampling=Resampling.nearest)
            return (arr.astype(np.float32) > threshold).astype(np.uint8)

    flooded_shapes = []
    for row in road_gdf.itertuples():
        props = {k.lower(): getattr(row, k) for k in getattr(row, "_fields", []) if k != "geometry"}
        value = None
        for key in ["flooded", "flood", "inundated", "is_flooded"]:
            if key in props:
                value = props[key]
                break
        if str(value).lower() in {"1", "true", "yes", "flooded", "inundated"}:
            flooded_shapes.append((getattr(row, "geometry"), 1))
    if not flooded_shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return features.rasterize(flooded_shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8", all_touched=True)


def build_segment_graph(segment_map: np.ndarray, segment_meta: Dict[int, Dict], flooded_mask: np.ndarray) -> Dict:
    graph = nx.Graph()
    ids = sorted(int(i) for i in np.unique(segment_map) if i > 0)
    for sid in ids:
        pix = segment_map == sid
        flooded_ratio = float(flooded_mask[pix].mean()) if pix.any() else 0.0
        meta = segment_meta.get(sid, {})
        graph.add_node(
            int(sid),
            label=int(flooded_ratio > 0.5),
            flooded_ratio=flooded_ratio,
            length=float(meta.get("length", pix.sum())),
            width=float(meta.get("width", 8.0)),
            pixel_count=int(pix.sum()),
        )

    # Adjacency by 8-neighborhood contact in the rasterized segment map.
    padded = np.pad(segment_map, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        neigh = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
        mask = (center > 0) & (neigh > 0) & (center != neigh)
        if not mask.any():
            continue
        pairs = np.stack([center[mask], neigh[mask]], axis=1)
        for a, b in np.unique(np.sort(pairs, axis=1), axis=0):
            graph.add_edge(int(a), int(b))

    return nx.node_link_data(graph)


def tile_windows(width: int, height: int, tile_size: int, overlap: int) -> Iterable[Tuple[int, int, Window]]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    for y in ys:
        for x in xs:
            yield x, y, Window(x, y, tile_size, tile_size)


def save_tile_array(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return str(path)


def process_scene(record: SceneRecord, processed_root: Path, cfg: Dict) -> List[Dict]:
    data_cfg = cfg["data"]
    tile_size = int(data_cfg["tile_size"])
    overlap = int(data_cfg["overlap"])
    default_width = int(data_cfg["default_road_width_px"])
    flood_threshold = float(data_cfg.get("flood_threshold", 0.5))
    rows: List[Dict] = []

    with rasterio.open(record.post_path) as post_src, rasterio.open(record.pre_path) as pre_src:
        road_gdf = gpd.read_file(record.road_geojson_path)
        if road_gdf.crs is not None and post_src.crs is not None and road_gdf.crs != post_src.crs:
            road_gdf = road_gdf.to_crs(post_src.crs)
        for x, y, window in tqdm(list(tile_windows(post_src.width, post_src.height, tile_size, overlap)), desc=record.scene_id):
            transform = post_src.window_transform(window)
            left, bottom, right, top = rasterio.windows.bounds(window, post_src.transform)
            clipped = road_gdf.cx[left:right, bottom:top]
            if clipped.empty:
                continue

            pre = read_rgb_window(pre_src, window, tile_size)
            post = read_rgb_window(post_src, window, tile_size)
            road_mask, segment_map, segment_meta = rasterize_roads(clipped, (tile_size, tile_size), transform, default_width)
            if road_mask.sum() == 0:
                continue
            flood_mask = rasterize_flood(record.flood_path, (tile_size, tile_size), transform, flood_threshold, clipped, window=window)
            flooded_road_mask = (road_mask & flood_mask).astype(np.uint8)
            graph = build_segment_graph(segment_map, segment_meta, flooded_road_mask)

            tile_id = f"{record.scene_id}_{y}_{x}"
            tile_dir = processed_root / "tiles" / tile_id
            paths = {
                "pre_path": save_tile_array(tile_dir / "pre.npy", pre),
                "post_path": save_tile_array(tile_dir / "post.npy", post),
                "road_mask_path": save_tile_array(tile_dir / "road_mask.npy", road_mask),
                "flood_mask_path": save_tile_array(tile_dir / "flood_mask.npy", flood_mask),
                "mask_path": save_tile_array(tile_dir / "flooded_road_mask.npy", flooded_road_mask),
                "segment_map_path": save_tile_array(tile_dir / "segment_map.npy", segment_map),
            }
            graph_path = tile_dir / "graph.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=True)
            rows.append(
                {
                    "id": tile_id,
                    "scene_id": record.scene_id,
                    "x": int(x),
                    "y": int(y),
                    "graph_path": str(graph_path),
                    **paths,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw_root = Path(args.raw_root or cfg["paths"]["raw_root"])
    processed_root = ensure_dir(args.processed_root or cfg["paths"]["processed_root"])

    if args.pairs_csv:
        records = read_pairs_csv(Path(args.pairs_csv), raw_root)
    else:
        records = discover_records(raw_root, cfg)
    if args.limit_scenes:
        records = records[: args.limit_scenes]

    all_rows: List[Dict] = []
    for record in records:
        all_rows.extend(process_scene(record, processed_root, cfg))

    if not all_rows:
        raise RuntimeError("No usable tiles were produced. Check raw paths, CRS alignment, and road labels.")

    seed = int(cfg["data"].get("split_seed", 42))
    train_ratio = float(cfg["data"].get("train_ratio", 0.8))
    indices = list(range(len(all_rows)))
    train_idx, val_idx = train_test_split(indices, train_size=train_ratio, random_state=seed, shuffle=True)
    split_by_idx = {i: "train" for i in train_idx}
    split_by_idx.update({i: "val" for i in val_idx})
    for i, row in enumerate(all_rows):
        row["split"] = split_by_idx[i]

    manifest = processed_root / "manifest.jsonl"
    write_jsonl(manifest, all_rows)
    save_json(processed_root / "preprocess_summary.json", {"num_tiles": len(all_rows), "manifest": str(manifest)})
    print(f"Wrote {len(all_rows)} tiles to {manifest}")


if __name__ == "__main__":
    main()
