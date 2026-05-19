from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FloodRoadDataset, collate_floodroad
from metrics import accumulate_confusion, binary_confusion, f1_iou_precision_recall, format_metric, markdown_table, segment_confusion_from_mask
from models.deeplabv3_baseline import build_deeplab
from models.floodroad_sam3 import build_floodroad_sam3
from models.sam3_baseline import build_sam3_adapter
from utils import checkpoint_path, ensure_dir, load_config, move_to_device, read_jsonl, resolve_device, save_json


ACCURACY_COLUMNS = ["Method", "Pixel F1", "Pixel IoU", "Segment-F1", "Precision", "Recall"]
EFFICIENCY_COLUMNS = ["Method", "Inference time (ms)", "FLOPs (G)", "Peak VRAM (GB)", "GPU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FloodRoad-SAM3 experiment methods.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["deeplab", "sam_text", "ours_tm"],
        choices=["deeplab", "sam_text", "ours_no_tm", "ours_tm"],
    )
    parser.add_argument("--use-rl-samples", action="store_true", help="Evaluate on saved 20 RL sample ids")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--efficiency-only", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def sample_ids_for_eval(cfg: Dict, output_dir: Path, use_rl_samples: bool) -> Optional[List[str]]:
    if not use_rl_samples:
        return None
    sample_path = output_dir / "rl_samples.json"
    if not sample_path.exists():
        manifest = Path(cfg["paths"]["processed_root"]) / "manifest.jsonl"
        rows = [r for r in read_jsonl(manifest) if r.get("split") == "train"]
        limit = int(cfg["data"].get("rl_sample_limit", 20))
        rng = random.Random(int(cfg["data"].get("split_seed", 42)))
        rng.shuffle(rows)
        ids = [r["id"] for r in rows[:limit]]
        save_json(sample_path, {"ids": ids, "note": "Generated for requested RL-sample evaluation subset."})
        return ids
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)["ids"]


def make_eval_loader(cfg: Dict, ids: Optional[List[str]], split: str) -> DataLoader:
    manifest = Path(cfg["paths"]["processed_root"]) / "manifest.jsonl"
    data_cfg = cfg["data"]
    dataset = FloodRoadDataset(
        manifest,
        split=None if ids is not None or split == "all" else split,
        ids=ids,
        normalize_mean=data_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=data_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        road_buffer_px=int(data_cfg.get("road_buffer_px", 16)),
        load_graph=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty.")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_floodroad,
    )


def load_method(method: str, cfg: Dict, device: torch.device, output_dir: Path):
    if method == "deeplab":
        model = build_deeplab(cfg.get("deeplab", {})).to(device)
        ckpt = checkpoint_path(output_dir, "deeplab")
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        else:
            print(f"Warning: {ckpt} not found; evaluating randomly initialized DeepLab.")
        model.eval()
        return model
    if method == "sam_text":
        model = build_sam3_adapter(cfg.get("sam3", {})).to(device)
        model.eval()
        return model
    if method in {"ours_no_tm", "ours_tm"}:
        model = build_floodroad_sam3(cfg, use_token_merging=(method == "ours_tm")).to(device)
        ckpt = checkpoint_path(output_dir, method)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device)["model"], strict=False)
        else:
            print(f"Warning: {ckpt} not found; evaluating randomly initialized {method}.")
        model.eval()
        return model
    raise ValueError(method)


@torch.no_grad()
def predict(method: str, model, batch: Dict, cfg: Dict, threshold: float) -> torch.Tensor:
    if method == "deeplab":
        return (torch.sigmoid(model(batch["post"])) >= threshold).float()
    if method == "sam_text":
        out = model.text_prompt_predict(batch["post"], cfg.get("sam3", {}).get("prompt", "flooded road"))
        return (torch.sigmoid(out.logits) >= threshold).float()
    if method in {"ours_no_tm", "ours_tm"}:
        return model.predict_mask(batch, threshold=threshold)
    raise ValueError(method)


def evaluate_accuracy(method: str, model, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, object]:
    pixel_total: Dict[str, float] = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    segment_total: Dict[str, float] = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    for batch in tqdm(loader, desc=f"accuracy {method}"):
        batch = move_to_device(batch, device)
        pred = predict(method, model, batch, cfg, threshold)
        target = batch["mask"]
        pixel_total = accumulate_confusion(pixel_total, binary_confusion(pred, target))
        segment_total = accumulate_confusion(
            segment_total,
            segment_confusion_from_mask(pred[0, 0], batch["graph"][0], batch["segment_map"][0]),
        )
    pixel_metrics = f1_iou_precision_recall(pixel_total)
    segment_metrics = f1_iou_precision_recall(segment_total)
    return {
        "Method": display_name(method, table="accuracy"),
        "Pixel F1": format_metric(pixel_metrics["Pixel F1"]),
        "Pixel IoU": format_metric(pixel_metrics["Pixel IoU"]),
        "Segment-F1": format_metric(segment_metrics["Pixel F1"]),
        "Precision": format_metric(pixel_metrics["Precision"]),
        "Recall": format_metric(pixel_metrics["Recall"]),
    }


def measure_efficiency(method: str, model, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, object]:
    batch = next(iter(loader))
    batch = move_to_device(batch, device)
    warmup = int(cfg.get("eval", {}).get("warmup", 10))
    iterations = int(cfg.get("eval", {}).get("iterations", 50))
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = predict(method, model, batch, cfg, threshold)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = predict(method, model, batch, cfg, threshold)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    peak_gb = ""
    gpu = "CPU"
    if device.type == "cuda":
        peak_gb = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 3):.3f}"
        gpu = torch.cuda.get_device_name(device)

    flops = estimate_flops(method, model, batch, device)
    return {
        "Method": display_name(method, table="efficiency"),
        "Inference time (ms)": f"{elapsed * 1000.0 / max(iterations, 1):.3f}",
        "FLOPs (G)": flops,
        "Peak VRAM (GB)": peak_gb,
        "GPU": gpu,
    }


def estimate_flops(method: str, model, batch: Dict, device: torch.device) -> str:
    try:
        from thop import profile
    except Exception:
        return ""
    try:
        if method == "deeplab":
            macs, _ = profile(model, inputs=(batch["post"],), verbose=False)
            return f"{2 * macs / 1e9:.3f}"
        # SAM3/FloodRoad wrappers take dicts and external graph objects; THOP is
        # not reliable there without backend-specific adapters.
        return "backend-specific"
    except Exception:
        return ""


def display_name(method: str, table: str = "accuracy") -> str:
    if table == "efficiency" and method == "ours_tm":
        return "Ours-TM"
    return {
        "deeplab": "DeepLabV3+",
        "sam_text": "SAM3 text-only",
        "ours_no_tm": "Ours-noTM",
        "ours_tm": "FloodRoad-SAM3 (Ours)",
    }[method]


def write_outputs(output_dir: Path, name: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    ensure_dir(output_dir)
    csv_path = output_dir / f"{name}.csv"
    md_path = output_dir / f"{name}.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    md = markdown_table(rows, columns)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"Wrote {csv_path} and {md_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or cfg["paths"].get("output_dir", "runs/default"))
    device = resolve_device(cfg.get("sam3", {}).get("device", "auto"))
    if cfg.get("eval", {}).get("use_rl_samples", True):
        use_rl = True
    else:
        use_rl = False
    use_rl = args.use_rl_samples or use_rl
    ids = sample_ids_for_eval(cfg, output_dir, use_rl)
    loader = make_eval_loader(cfg, ids=ids, split=args.split)

    accuracy_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []
    for method in args.methods:
        model = load_method(method, cfg, device, output_dir)
        if not args.skip_accuracy and not args.efficiency_only:
            accuracy_rows.append(evaluate_accuracy(method, model, loader, cfg, device))
        if not args.skip_efficiency:
            efficiency_rows.append(measure_efficiency(method, model, loader, cfg, device))

    if accuracy_rows:
        write_outputs(output_dir, "accuracy_table", accuracy_rows, ACCURACY_COLUMNS)
    if efficiency_rows:
        write_outputs(output_dir, "efficiency_table", efficiency_rows, EFFICIENCY_COLUMNS)


if __name__ == "__main__":
    main()
