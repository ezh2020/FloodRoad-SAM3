from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocess import build_segment_graph, normalize_uint8, save_tile_array, tile_windows
from utils import ensure_dir, load_config, save_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official SpaceNet 8 training data from mapping/reference CSV files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-root", default="/content/spacenet8/raw")
    parser.add_argument("--processed-root", default="/content/spacenet8/processed")
    parser.add_argument("--location", default="Louisiana-East_Training_Public")
    parser.add_argument("--limit-records", type=int, default=None, help="Limit source SN8 image records; positives are selected first.")
    return parser.parse_args()


def find_location_root(raw_root: Path, location: str) -> Path:
    """Find the extracted SpaceNet 8 location root.

    Public SN8 tarballs are not perfectly consistent about their top-level
    folder. Some extract into ``raw_root/location/...`` while others place the
    location files directly under ``raw_root`` or add an extra nesting layer.
    Prefer a directory that contains the official mapping/reference CSV files
    and both PRE/POST event image folders, but search ancestors of discovered
    CSVs too so all released layouts are accepted.
    """
    raw_root = raw_root.resolve()
    candidates: List[Path] = [raw_root, raw_root / location]
    candidates.extend(raw_root.glob(f"**/{location}"))

    marker_files = list(raw_root.rglob("*_label_image_mapping.csv")) + list(raw_root.rglob("*_reference.csv"))
    for marker in marker_files:
        current = marker.parent
        while True:
            candidates.append(current)
            if current == raw_root or raw_root not in current.parents:
                break
            current = current.parent

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except FileNotFoundError:
            continue
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if find_event_dir(candidate, "PRE-event") and find_event_dir(candidate, "POST-event"):
            if find_first(candidate, "*_label_image_mapping.csv") and find_first(candidate, "*_reference.csv"):
                return candidate
    raise FileNotFoundError(f"Could not find extracted SpaceNet 8 location {location!r} under {raw_root}")


def find_event_dir(root: Path, name: str) -> Optional[Path]:
    direct = root / name
    if direct.exists():
        return direct
    target = name.lower()
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == target:
            return candidate
    return None


def find_first(root: Path, pattern: str) -> Optional[Path]:
    direct = sorted(root.glob(pattern))
    if direct:
        return direct[0]
    nested = sorted(root.rglob(pattern))
    return nested[0] if nested else None


def resolve_image_path(event_dir: Path, image_name: str) -> Optional[Path]:
    candidates = [event_dir / image_name, event_dir / Path(image_name).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(event_dir.rglob(Path(image_name).name))
    return matches[0] if matches else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def image_id(path_or_name: str) -> str:
    return Path(path_or_name).stem


def select_mapping_rows(rows: List[Dict[str, str]], road_refs: Dict[str, List[Dict[str, str]]], limit: Optional[int], seed: int) -> List[Dict[str, str]]:
    if limit is None or len(rows) <= limit:
        return rows
    positives, negatives = [], []
    for row in rows:
        refs = road_refs.get(image_id(row["pre-event image"]), [])
        if any(str(r.get("Flooded", "")).lower() == "true" for r in refs):
            positives.append(row)
        else:
            negatives.append(row)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return (positives + negatives)[:limit]


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        channels = min(3, src.count)
        arr = src.read(indexes=list(range(1, channels + 1)))
        if channels == 1:
            arr = np.repeat(arr, 3, axis=0)
        if arr.shape[0] < 3:
            arr = np.concatenate([arr, np.zeros((3 - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)], axis=0)
    return normalize_uint8(np.transpose(arr[:3], (1, 2, 0)))


def draw_lines(mask: np.ndarray, geom, value: int, width: int) -> None:
    if geom.is_empty:
        return
    lines: Iterable[LineString]
    if isinstance(geom, LineString):
        lines = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        return
    for line in lines:
        coords = np.asarray(line.coords, dtype=np.float32)
        if coords.shape[0] < 2:
            continue
        pts = np.rint(coords).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=int(value), thickness=int(width), lineType=cv2.LINE_8)


def build_masks_from_reference(refs: List[Dict[str, str]], shape: Tuple[int, int], default_width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict]]:
    h, w = shape
    road_mask = np.zeros((h, w), dtype=np.uint8)
    flooded_road_mask = np.zeros((h, w), dtype=np.uint8)
    segment_map = np.zeros((h, w), dtype=np.int32)
    segment_meta: Dict[int, Dict] = {}
    sid = 1
    for row in refs:
        if row.get("Object") != "Road":
            continue
        wkt_text = row.get("Wkt_Pix", "")
        if not wkt_text or "EMPTY" in wkt_text.upper():
            continue
        try:
            geom = wkt.loads(wkt_text)
        except Exception:
            continue
        draw_lines(road_mask, geom, 1, default_width)
        draw_lines(segment_map, geom, sid, default_width)
        flooded = str(row.get("Flooded", "")).lower() == "true"
        if flooded:
            draw_lines(flooded_road_mask, geom, 1, default_width)
        length = float(row.get("length_m") or 0.0) if row.get("length_m") != "Null" else 0.0
        segment_meta[sid] = {"length": length, "width": float(default_width), "properties": row}
        sid += 1
    return road_mask, flooded_road_mask, segment_map, segment_meta


def crop_with_pad(arr: np.ndarray, x: int, y: int, size: int, fill: int = 0) -> np.ndarray:
    if arr.ndim == 3:
        out = np.full((size, size, arr.shape[2]), fill, dtype=arr.dtype)
        crop = arr[y : y + size, x : x + size]
        out[: crop.shape[0], : crop.shape[1], :] = crop
    else:
        out = np.full((size, size), fill, dtype=arr.dtype)
        crop = arr[y : y + size, x : x + size]
        out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def process_record(
    row: Dict[str, str],
    pre_dir: Path,
    post_dir: Path,
    processed_root: Path,
    refs_by_image: Dict[str, List[Dict[str, str]]],
    cfg: Dict,
) -> List[Dict]:
    data_cfg = cfg["data"]
    tile_size = int(data_cfg["tile_size"])
    overlap = int(data_cfg["overlap"])
    default_width = int(data_cfg.get("default_road_width_px", 8))
    pre_name = row["pre-event image"]
    post_name = row.get("post-event image 1") or row.get("post-event image 2")
    if not post_name:
        return []
    pre_path = resolve_image_path(pre_dir, pre_name)
    post_path = resolve_image_path(post_dir, post_name)
    if pre_path is None or post_path is None:
        return []
    pre = read_rgb(pre_path)
    post = read_rgb(post_path)
    h, w = post.shape[:2]
    refs = refs_by_image.get(image_id(pre_name), [])
    road_mask, flooded_road_mask, segment_map, segment_meta = build_masks_from_reference(refs, (h, w), default_width)
    if road_mask.sum() == 0:
        return []

    rows: List[Dict] = []
    for x, y, _window in tile_windows(w, h, tile_size, overlap):
        road_tile = crop_with_pad(road_mask, x, y, tile_size)
        if road_tile.sum() == 0:
            continue
        segment_tile = crop_with_pad(segment_map, x, y, tile_size)
        flood_tile = crop_with_pad(flooded_road_mask, x, y, tile_size)
        pre_tile = crop_with_pad(pre, x, y, tile_size)
        post_tile = crop_with_pad(post, x, y, tile_size)
        graph = build_segment_graph(segment_tile, segment_meta, flood_tile)
        tile_id = f"{image_id(pre_name)}_{y}_{x}"
        tile_dir = processed_root / "tiles" / tile_id
        paths = {
            "pre_path": save_tile_array(tile_dir / "pre.npy", pre_tile),
            "post_path": save_tile_array(tile_dir / "post.npy", post_tile),
            "road_mask_path": save_tile_array(tile_dir / "road_mask.npy", road_tile),
            "flood_mask_path": save_tile_array(tile_dir / "flood_mask.npy", flood_tile),
            "mask_path": save_tile_array(tile_dir / "flooded_road_mask.npy", flood_tile),
            "segment_map_path": save_tile_array(tile_dir / "segment_map.npy", segment_tile),
        }
        graph_path = tile_dir / "graph.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=True)
        rows.append(
            {
                "id": tile_id,
                "scene_id": image_id(pre_name),
                "x": int(x),
                "y": int(y),
                "graph_path": str(graph_path),
                "flooded_pixels": int(flood_tile.sum()),
                "road_pixels": int(road_tile.sum()),
                "source_pre": str(pre_path),
                "source_post": str(post_path),
                **paths,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["data"].get("split_seed", 42))
    loc_root = find_location_root(Path(args.raw_root), args.location)
    pre_dir = find_event_dir(loc_root, "PRE-event")
    post_dir = find_event_dir(loc_root, "POST-event")
    if pre_dir is None or post_dir is None:
        raise FileNotFoundError(f"Could not find PRE-event/POST-event folders under {loc_root}")
    processed_root = ensure_dir(args.processed_root)
    mapping_path = find_first(loc_root, "*_label_image_mapping.csv")
    reference_path = find_first(loc_root, "*_reference.csv")
    if mapping_path is None or reference_path is None:
        raise FileNotFoundError(f"Could not find SpaceNet 8 mapping/reference CSV files under {loc_root}")
    print(f"SN8 root: {loc_root}", flush=True)
    print(f"PRE-event dir: {pre_dir}", flush=True)
    print(f"POST-event dir: {post_dir}", flush=True)
    print(f"Mapping CSV: {mapping_path}", flush=True)
    print(f"Reference CSV: {reference_path}", flush=True)
    mapping_rows = read_csv(mapping_path)
    reference_rows = read_csv(reference_path)
    road_refs: Dict[str, List[Dict[str, str]]] = {}
    for ref in reference_rows:
        if ref.get("Object") == "Road":
            road_refs.setdefault(ref["ImageId"], []).append(ref)
    mapping_rows = select_mapping_rows(mapping_rows, road_refs, args.limit_records, seed)

    all_rows: List[Dict] = []
    for row in tqdm(mapping_rows, desc=f"SN8 {args.location}"):
        all_rows.extend(process_record(row, pre_dir, post_dir, processed_root, road_refs, cfg))
    if not all_rows:
        raise RuntimeError("No usable SpaceNet 8 tiles were produced.")

    train_ratio = float(cfg["data"].get("train_ratio", 0.8))
    indices = list(range(len(all_rows)))
    train_idx, val_idx = train_test_split(indices, train_size=train_ratio, random_state=seed, shuffle=True)
    split_by_idx = {i: "train" for i in train_idx}
    split_by_idx.update({i: "val" for i in val_idx})
    for i, out_row in enumerate(all_rows):
        out_row["split"] = split_by_idx[i]
    manifest = processed_root / "manifest.jsonl"
    write_jsonl(manifest, all_rows)
    save_json(
        processed_root / "preprocess_summary.json",
        {
            "num_tiles": len(all_rows),
            "positive_tiles": sum(1 for r in all_rows if r.get("flooded_pixels", 0) > 0),
            "manifest": str(manifest),
            "location": args.location,
            "mapping_rows": len(mapping_rows),
        },
    )
    print(f"Wrote {len(all_rows)} SN8 tiles to {manifest}")


if __name__ == "__main__":
    main()
