from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FloodRoadDataset, collate_floodroad, segment_labels_from_graph
from metrics import binary_confusion, f1_iou_precision_recall
from models.ccrl import compute_reward, supervised_segment_targets
from models.deeplabv3_baseline import build_deeplab
from models.floodroad_sam3 import build_floodroad_sam3
from models.losses import BCEDiceLoss
from utils import checkpoint_path, ensure_dir, load_config, move_to_device, read_jsonl, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FloodRoad-SAM3 experiment methods.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["deeplab", "ours_no_tm", "ours_tm"], required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one short epoch for syntax/integration checks")
    return parser.parse_args()


def make_loader(cfg: Dict, split: str, ids: Optional[List[str]] = None, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    processed_root = Path(cfg["paths"]["processed_root"])
    manifest = processed_root / "manifest.jsonl"
    data_cfg = cfg["data"]
    dataset = FloodRoadDataset(
        manifest,
        split=split,
        ids=ids,
        normalize_mean=data_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=data_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        road_buffer_px=int(data_cfg.get("road_buffer_px", 16)),
        load_graph=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No dataset items found for split={split}, ids={ids}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_floodroad,
    )


def select_rl_sample_ids(cfg: Dict, output_dir: Path) -> List[str]:
    sample_path = output_dir / "rl_samples.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)["ids"]
    manifest = Path(cfg["paths"]["processed_root"]) / "manifest.jsonl"
    rows = [r for r in read_jsonl(manifest) if r.get("split") == "train"]
    limit = int(cfg["data"].get("rl_sample_limit", 20))
    rng = random.Random(int(cfg["data"].get("split_seed", 42)))
    rng.shuffle(rows)
    ids = [r["id"] for r in rows[:limit]]
    ensure_dir(output_dir)
    save_json(sample_path, {"ids": ids, "note": "Used for CC-RL training and requested evaluation subset."})
    return ids


def train_deeplab(cfg: Dict, args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    model = build_deeplab(cfg.get("deeplab", {})).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
    epochs = 1 if args.smoke else int(cfg["deeplab"].get("epochs", 100))
    batch_size = 2 if args.smoke else int(cfg["deeplab"].get("batch_size", cfg["data"].get("batch_size_deeplab", 4)))
    loader = make_loader(cfg, split="train", batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["deeplab"].get("lr", 1e-4)), weight_decay=float(cfg["deeplab"].get("weight_decay", 1e-5)))
    loss_fn = BCEDiceLoss()
    ckpt = checkpoint_path(output_dir, "deeplab")
    ensure_dir(ckpt.parent)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"deeplab epoch {epoch+1}/{epochs}")):
            batch = move_to_device(batch, device)
            logits = model(batch["post"])
            loss = loss_fn(logits, batch["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if args.smoke and step >= 1:
                break
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": cfg}, ckpt)
        print(f"epoch={epoch+1} loss={running / max(1, len(loader)):.4f} saved={ckpt}")


def train_ours(cfg: Dict, args: argparse.Namespace, device: torch.device, output_dir: Path, use_tm: bool) -> None:
    method = "ours_tm" if use_tm else "ours_no_tm"
    model = build_floodroad_sam3(cfg, use_token_merging=use_tm).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"], strict=False)
    rl_ids = select_rl_sample_ids(cfg, output_dir)
    batch_size = 1
    loader = make_loader(cfg, split="train", ids=rl_ids, batch_size=batch_size, shuffle=True)
    ours_cfg = cfg["ours"]
    phase1_epochs = 1 if args.smoke else int(ours_cfg.get("phase1_epochs", 30))
    phase2_epochs = 1 if args.smoke else int(ours_cfg.get("phase2_epochs", 20))

    seg_loss_fn = BCEDiceLoss()
    lora_params = list(model.trainable_lora_parameters())
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "policy" not in n and "lora_" not in n]
    policy_params = list(model.policy.parameters())
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(ours_cfg.get("lr_lora", 1e-5))})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(ours_cfg.get("lr_head", 1e-4))})
    param_groups.append({"params": policy_params, "lr": float(ours_cfg.get("lr_policy", 1e-4))})
    opt = torch.optim.Adam(param_groups, weight_decay=float(ours_cfg.get("weight_decay", 1e-5)))
    ckpt = checkpoint_path(output_dir, method)
    ensure_dir(ckpt.parent)

    for epoch in range(phase1_epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"{method} phase I {epoch+1}/{phase1_epochs}")):
            batch = move_to_device(batch, device)
            logits, _ = model.forward_segmentation(batch)
            loss_seg = seg_loss_fn(logits, batch["mask"])
            score = torch.sigmoid(logits.detach())
            policy_logits, node_ids = model.forward_policy_supervised(batch, score)
            targets = supervised_segment_targets(batch["graph"][0], node_ids, device)
            loss_policy = F.cross_entropy(policy_logits, targets) if len(node_ids) else torch.tensor(0.0, device=device)
            loss = loss_seg + loss_policy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            if args.smoke and step >= 1:
                break
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "phase": 1, "config": cfg}, ckpt)
        print(f"phase=I epoch={epoch+1} loss={running / max(1, len(loader)):.4f} saved={ckpt}")

    if bool(ours_cfg.get("freeze_lora_in_rl", True)):
        for p in model.trainable_lora_parameters():
            p.requires_grad = False
    opt_rl = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(ours_cfg.get("lr_policy", 1e-4)))
    baseline = None
    reward_cfg = ours_cfg.get("ccrl", {})
    for epoch in range(phase2_epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"{method} phase II {epoch+1}/{phase2_epochs}")):
            batch = move_to_device(batch, device)
            logits, _ = model.forward_segmentation(batch)
            score = torch.sigmoid(logits)
            rollout = model.rollout(batch, score, sample=True)
            reward, reward_details = compute_reward(
                batch["graph"][0],
                rollout.node_ids,
                rollout.actions,
                reachability_pairs=int(reward_cfg.get("reachability_pairs", 50)),
                weights=reward_cfg.get("reward_weights"),
            )
            baseline = reward.detach() if baseline is None else 0.9 * baseline + 0.1 * reward.detach()
            advantage = reward - baseline
            loss_rl = -(advantage.detach() * rollout.log_probs).mean() if rollout.log_probs.numel() else torch.tensor(0.0, device=device)
            loss_seg = seg_loss_fn(logits, batch["mask"])
            loss = float(ours_cfg.get("loss_seg_weight_phase2", 0.1)) * loss_seg + loss_rl
            opt_rl.zero_grad(set_to_none=True)
            loss.backward()
            opt_rl.step()
            running += loss.item()
            if args.smoke and step >= 1:
                break
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "phase": 2, "config": cfg}, ckpt)
        print(f"phase=II epoch={epoch+1} loss={running / max(1, len(loader)):.4f} saved={ckpt}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["data"].get("split_seed", 42)))
    output_dir = ensure_dir(cfg["paths"].get("output_dir", "runs/default"))
    device = resolve_device(cfg.get("sam3", {}).get("device", "auto"))
    print(f"Using device: {device}")
    if args.method == "deeplab":
        train_deeplab(cfg, args, device, output_dir)
    elif args.method == "ours_no_tm":
        train_ours(cfg, args, device, output_dir, use_tm=False)
    elif args.method == "ours_tm":
        train_ours(cfg, args, device, output_dir, use_tm=True)


if __name__ == "__main__":
    main()
