from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patheffects
from matplotlib.colors import BoundaryNorm, ListedColormap

from data.dataset import FloodRoadDataset, collate_floodroad
from evaluate import display_name_for_cfg, load_method, sample_ids_for_eval
from utils import ensure_dir, load_config, move_to_device, read_jsonl, resolve_device, save_json


DEFAULT_METHODS = ["deeplab", "sam_text", "ours_no_tm", "ours_tm"]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize per-road segment scores for one evaluation sample.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-index", type=int, default=1, help="One-based index into rl_samples.json.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def make_single_batch(cfg: Dict, sample_id: str):
    data_cfg = cfg["data"]
    manifest = Path(cfg["paths"]["processed_root"]) / "manifest.jsonl"
    dataset = FloodRoadDataset(
        manifest,
        ids=[sample_id],
        normalize_mean=data_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=data_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        road_buffer_px=int(data_cfg.get("road_buffer_px", 16)),
        load_graph=True,
    )
    if len(dataset) != 1:
        available = {row["id"] for row in read_jsonl(manifest)}
        raise RuntimeError(f"Sample id {sample_id!r} resolved to {len(dataset)} rows; present={sample_id in available}")
    item = dataset[0]
    return item, collate_floodroad([item])


@torch.no_grad()
def method_score_map(method: str, model, batch: Dict, cfg: Dict) -> torch.Tensor:
    if method == "deeplab":
        return torch.sigmoid(model(batch["post"])).detach()
    if method == "sam_text":
        out = model.text_prompt_predict(batch["post"], cfg.get("sam3", {}).get("prompt", "flooded road"))
        return torch.sigmoid(out.logits).detach()
    if method in {"ours_no_tm", "ours_tm"}:
        logits, _ = model.forward_segmentation(batch)
        return torch.sigmoid(logits).detach()
    raise ValueError(method)


def score_segments(item: Dict, score_maps: Dict[str, np.ndarray], cfg: Dict, threshold: float) -> pd.DataFrame:
    segment_map = item["segment_map"].numpy()
    graph = item["graph"]
    rows = []
    for node, data in sorted(graph.nodes(data=True), key=lambda x: int(x[0])):
        sid = int(node)
        pix = segment_map == sid
        if not pix.any():
            continue
        row = {
            "segment_id": sid,
            "gt_flooded": int(data.get("label", 0)),
            "road_pixels": int(pix.sum()),
        }
        for method, score_map in score_maps.items():
            display_name = display_name_for_cfg(method, cfg)
            score = float(score_map[pix].mean())
            row[f"{display_name} score"] = round(score, 4)
            row[f"{display_name} pred"] = int(score >= threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def overlay_from_segments(segment_map: np.ndarray, values: Dict[int, float]) -> np.ma.MaskedArray:
    overlay = np.full(segment_map.shape, np.nan, dtype=np.float32)
    for sid, value in values.items():
        overlay[segment_map == int(sid)] = float(value)
    return np.ma.masked_invalid(overlay)


def annotate_segment_ids(ax, segment_map: np.ndarray, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        sid = int(row["segment_id"])
        ys, xs = np.nonzero(segment_map == sid)
        if xs.size == 0:
            continue
        text = ax.text(
            float(xs.mean()),
            float(ys.mean()),
            str(sid),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
        )
        text.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="black")])


def draw_visualization(item: Dict, df: pd.DataFrame, score_maps: Dict[str, np.ndarray], cfg: Dict, output_path: Path) -> None:
    post = item["post_raw"].permute(1, 2, 0).numpy().clip(0, 1)
    segment_map = item["segment_map"].numpy()
    road_mask = item["road_mask"][0].numpy() > 0.5
    target = item["mask"][0].numpy() > 0.5
    methods = list(score_maps.keys())

    panels = 2 + len(methods)
    cols = 3
    rows = int(np.ceil(panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.4 * rows), dpi=140)
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(post)
    axes[0].set_title(f"Post image\n{item['id']}")
    axes[0].axis("off")

    axes[1].imshow(post)
    gt_overlay = np.full(segment_map.shape, np.nan, dtype=np.float32)
    gt_overlay[road_mask] = 0
    gt_overlay[target] = 1
    gt_cmap = ListedColormap(["#22c55e", "#ef4444"])
    gt_norm = BoundaryNorm([-0.5, 0.5, 1.5], gt_cmap.N)
    axes[1].imshow(np.ma.masked_invalid(gt_overlay), cmap=gt_cmap, norm=gt_norm, alpha=0.72)
    axes[1].set_title("Ground truth\ngreen=passable, red=flooded")
    annotate_segment_ids(axes[1], segment_map, df)
    axes[1].axis("off")

    for ax, method in zip(axes[2:], methods):
        display_name = display_name_for_cfg(method, cfg)
        score_col = f"{display_name} score"
        values = {int(row["segment_id"]): float(row[score_col]) for _, row in df.iterrows()}
        ax.imshow(post)
        image = ax.imshow(overlay_from_segments(segment_map, values), cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.72)
        ax.set_title(display_name)
        annotate_segment_ids(ax, segment_map, df)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[panels:]:
        ax.axis("off")

    fig.suptitle("Per-road flooded score by method", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or cfg["paths"].get("output_dir", "runs/default"))
    threshold = float(args.threshold if args.threshold is not None else cfg.get("eval", {}).get("threshold", 0.5))
    sample_ids = sample_ids_for_eval(cfg, output_dir, use_rl_samples=True)
    if not 1 <= args.sample_index <= len(sample_ids):
        raise ValueError(f"sample-index must be in [1, {len(sample_ids)}], got {args.sample_index}")
    sample_id = sample_ids[args.sample_index - 1]

    item, batch = make_single_batch(cfg, sample_id)
    device = resolve_device(cfg.get("sam3", {}).get("device", "auto"))
    batch = move_to_device(batch, device)

    score_maps: Dict[str, np.ndarray] = {}
    for method in args.methods:
        model = load_method(method, cfg, device, output_dir)
        model.eval()
        score = method_score_map(method, model, batch, cfg)[0, 0].float().cpu().numpy()
        score_maps[method] = score
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = score_segments(item, score_maps, cfg, threshold)
    vis_dir = ensure_dir(output_dir / "segment_visualizations")
    stem = f"sample_{args.sample_index:02d}_{sample_id}"
    csv_path = vis_dir / f"{stem}_segment_scores.csv"
    md_path = vis_dir / f"{stem}_segment_scores.md"
    png_path = vis_dir / f"{stem}_segment_scores.png"
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df) + "\n", encoding="utf-8")
    draw_visualization(item, df, score_maps, cfg, png_path)

    summary = {
        "sample_index": args.sample_index,
        "sample_count": len(sample_ids),
        "sample_id": sample_id,
        "threshold": threshold,
        "figure": str(png_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    summary_path = Path(args.summary_json) if args.summary_json else vis_dir / "latest_sample.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
