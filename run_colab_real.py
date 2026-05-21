from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real FloodRoad-SAM3 Colab experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--raw-root", default="/content/spacenet8/raw")
    parser.add_argument("--processed-root", default="/content/spacenet8/processed")
    parser.add_argument("--output-dir", default="/content/floodroad_runs/default")
    parser.add_argument("--pairs-csv", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-deeplab", action="store_true")
    parser.add_argument("--skip-ours-no-tm", action="store_true")
    parser.add_argument("--skip-ours-tm", action="store_true")
    parser.add_argument("--skip-sam3-install", action="store_true")
    parser.add_argument(
        "--deeplab-checkpoint-url",
        default=None,
        help="Optional URL for a pretrained deeplab.pt. If set, download it and skip DeepLab training.",
    )
    parser.add_argument(
        "--force-deeplab-checkpoint-download",
        action="store_true",
        help="Redownload --deeplab-checkpoint-url even when checkpoints/deeplab.pt already exists.",
    )
    parser.add_argument(
        "--ours-no-tm-checkpoint-url",
        default=None,
        help="Optional URL for a pretrained ours_no_tm.pt. If set, download it and skip Ours-noTM training.",
    )
    parser.add_argument(
        "--ours-tm-checkpoint-url",
        default=None,
        help="Optional URL for a pretrained ours_tm.pt. If set, download it and skip Ours-TM training.",
    )
    parser.add_argument(
        "--force-ours-checkpoint-download",
        action="store_true",
        help="Redownload provided Ours checkpoint URLs even when checkpoint files already exist.",
    )
    parser.add_argument("--download-prefix", default="spacenet/SN8_floods/")
    parser.add_argument("--sn8-location", default="Louisiana-East_Training_Public")
    parser.add_argument("--sn8-tarball", default="Louisiana-East_Training_Public.tar.gz")
    parser.add_argument("--limit-records", type=int, default=32, help="Limit official SN8 source images while keeping real data.")
    parser.add_argument(
        "--deeplab-source-raw-root",
        default=None,
        help="Optional raw root for training DeepLab on a different source domain.",
    )
    parser.add_argument(
        "--deeplab-source-processed-root",
        default=None,
        help="Optional processed root for training DeepLab on a different source domain.",
    )
    parser.add_argument(
        "--deeplab-source-location",
        default=None,
        help="Optional SN8 location for source-domain DeepLab training, e.g. Germany_Training_Public.",
    )
    parser.add_argument(
        "--deeplab-source-tarball",
        default=None,
        help="Optional SN8 tarball for source-domain DeepLab training, e.g. Germany_Training_Public.tar.gz.",
    )
    parser.add_argument(
        "--deeplab-source-limit-records",
        type=int,
        default=None,
        help="Optional source-domain record limit for DeepLab training. Defaults to --limit-records.",
    )
    parser.add_argument("--deeplab-epochs", type=int, default=None)
    parser.add_argument("--ours-phase1-epochs", type=int, default=None)
    parser.add_argument("--ours-phase2-epochs", type=int, default=None)
    return parser.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_cfg(path: str | os.PathLike[str]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_deeplab_checkpoint_url(args: argparse.Namespace) -> bool:
    return bool((args.deeplab_checkpoint_url or os.environ.get("DEEPLAB_CHECKPOINT_URL") or "").strip())


def write_runtime_config(cfg: dict, args: argparse.Namespace) -> Path:
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg.get("paths", {}))
    cfg["paths"]["raw_root"] = args.raw_root
    cfg["paths"]["processed_root"] = args.processed_root
    cfg["paths"]["output_dir"] = args.output_dir
    cfg["sam3"] = dict(cfg.get("sam3", {}))
    cfg["sam3"]["allow_mock"] = False
    cfg["sam3"]["device"] = "cuda"
    cfg["eval"] = dict(cfg.get("eval", {}))
    cfg["eval"]["use_rl_samples"] = True
    cfg["deeplab"] = dict(cfg.get("deeplab", {}))
    cfg["ours"] = dict(cfg.get("ours", {}))
    if args.deeplab_epochs is not None:
        cfg["deeplab"]["epochs"] = args.deeplab_epochs
    if using_deeplab_source_domain(args) and not has_deeplab_checkpoint_url(args):
        cfg["deeplab"]["source_domain"] = args.deeplab_source_location or "custom"
    if args.ours_phase1_epochs is not None:
        cfg["ours"]["phase1_epochs"] = args.ours_phase1_epochs
    if args.ours_phase2_epochs is not None:
        cfg["ours"]["phase2_epochs"] = args.ours_phase2_epochs
    out = Path(args.output_dir) / "real_run.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(out.read_text(), flush=True)
    return out


def write_deeplab_source_config(cfg: dict, args: argparse.Namespace) -> Path:
    source_raw_root = args.deeplab_source_raw_root or "/content/spacenet8/deeplab_source_raw"
    source_processed_root = args.deeplab_source_processed_root or "/content/spacenet8/deeplab_source_processed"
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg.get("paths", {}))
    cfg["paths"]["raw_root"] = source_raw_root
    cfg["paths"]["processed_root"] = source_processed_root
    cfg["paths"]["output_dir"] = args.output_dir
    cfg["sam3"] = dict(cfg.get("sam3", {}))
    cfg["sam3"]["allow_mock"] = False
    cfg["sam3"]["device"] = "cuda"
    cfg["deeplab"] = dict(cfg.get("deeplab", {}))
    if args.deeplab_epochs is not None:
        cfg["deeplab"]["epochs"] = args.deeplab_epochs
    out = Path(args.output_dir) / "deeplab_source_run.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("DeepLab source-domain config:", flush=True)
    print(out.read_text(), flush=True)
    return out


def ensure_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. In Colab, connect a GPU runtime before running the real experiment.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def ensure_hf_auth() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return
    try:
        from huggingface_hub import HfFolder

        if HfFolder.get_token():
            return
    except Exception:
        pass
    raise RuntimeError(
        "SAM3 checkpoints are gated on Hugging Face. Set HF_TOKEN/HUGGING_FACE_HUB_TOKEN "
        "or run `hf auth login` in Colab after your account has access to facebook/sam3."
    )


def install_sam3(cfg: dict) -> None:
    try:
        import sam3  # noqa: F401
        return
    except Exception:
        pass
    url = cfg.get("sam3", {}).get("install_url", "git+https://github.com/facebookresearch/sam3.git")
    run([sys.executable, "-m", "pip", "install", "-q", url])


def processed_manifest_exists(processed_root: str) -> bool:
    manifest = Path(processed_root) / "manifest.jsonl"
    return manifest.exists() and manifest.stat().st_size > 0


def raw_data_exists(raw_root: str) -> bool:
    root = Path(raw_root)
    return root.exists() and any(root.rglob("*.tif")) and (any(root.rglob("*.geojson")) or any(root.rglob("*_reference.csv")))


def deeplab_checkpoint_path(output_dir: str) -> Path:
    return model_checkpoint_path(output_dir, "deeplab")


def model_checkpoint_path(output_dir: str, method: str) -> Path:
    return Path(output_dir) / "checkpoints" / f"{method}.pt"


def checkpoint_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download_checkpoint(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    if "drive.google.com" in url:
        try:
            import gdown  # noqa: F401
        except Exception:
            run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        run([sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(tmp_path)])
    else:
        print(f"Downloading {label} checkpoint from {url} to {tmp_path}", flush=True)
        urllib.request.urlretrieve(url, tmp_path)
    if not checkpoint_exists(tmp_path):
        raise RuntimeError(f"Downloaded checkpoint is missing or empty: {tmp_path}")
    tmp_path.replace(destination)
    print(f"{label} checkpoint ready: {destination}", flush=True)


def download_deeplab_checkpoint(url: str, destination: Path) -> None:
    download_checkpoint(url, destination, "DeepLab")


def ensure_deeplab_checkpoint(args: argparse.Namespace) -> bool:
    url = args.deeplab_checkpoint_url or os.environ.get("DEEPLAB_CHECKPOINT_URL") or ""
    url = url.strip()
    if not url:
        return False
    destination = deeplab_checkpoint_path(args.output_dir)
    if checkpoint_exists(destination) and not args.force_deeplab_checkpoint_download:
        print(f"Found existing DeepLab checkpoint at {destination}; using it.", flush=True)
    else:
        download_deeplab_checkpoint(url, destination)
    return True


def ensure_ours_checkpoint(args: argparse.Namespace, method: str, url_arg: str, env_name: str, label: str) -> bool:
    url = url_arg or os.environ.get(env_name) or ""
    url = url.strip()
    if not url:
        return False
    destination = model_checkpoint_path(args.output_dir, method)
    if checkpoint_exists(destination) and not args.force_ours_checkpoint_download:
        print(f"Found existing {label} checkpoint at {destination}; using it.", flush=True)
    else:
        download_checkpoint(url, destination, label)
    return True


def ensure_data(args: argparse.Namespace, config_path: Path) -> None:
    if processed_manifest_exists(args.processed_root):
        print(f"Found processed manifest under {args.processed_root}; reusing it.", flush=True)
        return
    if not raw_data_exists(args.raw_root):
        if args.skip_download:
            raise RuntimeError(f"No processed manifest and no raw SpaceNet 8 data found at {args.raw_root}.")
        if shutil.which("aws") is None:
            run([sys.executable, "-m", "pip", "install", "-q", "awscli"])
        if args.sn8_tarball:
            run([sys.executable, "data/download.py", "--output", args.raw_root, "--tarball", args.sn8_tarball, "--extract"])
        else:
            run([sys.executable, "data/download.py", "--output", args.raw_root, "--prefix", args.download_prefix])
    if args.skip_preprocess:
        raise RuntimeError(f"Preprocess was skipped but {args.processed_root}/manifest.jsonl does not exist.")
    if args.pairs_csv:
        cmd = [
            sys.executable,
            "data/preprocess.py",
            "--config",
            str(config_path),
            "--raw-root",
            args.raw_root,
            "--processed-root",
            args.processed_root,
            "--pairs-csv",
            args.pairs_csv,
        ]
    else:
        cmd = [
            sys.executable,
            "data/preprocess_sn8.py",
            "--config",
            str(config_path),
            "--raw-root",
            args.raw_root,
            "--processed-root",
            args.processed_root,
            "--location",
            args.sn8_location,
            "--limit-records",
            str(args.limit_records),
        ]
    run(cmd)


def using_deeplab_source_domain(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in [
            args.deeplab_source_raw_root,
            args.deeplab_source_processed_root,
            args.deeplab_source_location,
            args.deeplab_source_tarball,
            args.deeplab_source_limit_records,
        ]
    )


def ensure_deeplab_source_data(args: argparse.Namespace, config_path: Path) -> None:
    source_args = argparse.Namespace(**vars(args))
    source_args.raw_root = args.deeplab_source_raw_root or "/content/spacenet8/deeplab_source_raw"
    source_args.processed_root = args.deeplab_source_processed_root or "/content/spacenet8/deeplab_source_processed"
    source_args.sn8_location = args.deeplab_source_location or args.sn8_location
    source_args.sn8_tarball = args.deeplab_source_tarball or args.sn8_tarball
    source_args.limit_records = args.deeplab_source_limit_records or args.limit_records
    print(
        "Preparing DeepLab source domain: "
        f"location={source_args.sn8_location} raw_root={source_args.raw_root} processed_root={source_args.processed_root}",
        flush=True,
    )
    ensure_data(source_args, config_path)


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    cfg = load_cfg(args.config)
    runtime_config = write_runtime_config(cfg, args)
    deeplab_source_config = write_deeplab_source_config(cfg, args) if using_deeplab_source_domain(args) else None
    ensure_gpu()
    if not args.skip_sam3_install:
        install_sam3(cfg)
    ensure_hf_auth()
    deeplab_checkpoint_ready = ensure_deeplab_checkpoint(args)
    ours_no_tm_checkpoint_ready = ensure_ours_checkpoint(
        args,
        "ours_no_tm",
        args.ours_no_tm_checkpoint_url,
        "OURS_NO_TM_CHECKPOINT_URL",
        "Ours-noTM",
    )
    ours_tm_checkpoint_ready = ensure_ours_checkpoint(
        args,
        "ours_tm",
        args.ours_tm_checkpoint_url,
        "OURS_TM_CHECKPOINT_URL",
        "Ours-TM",
    )
    if deeplab_source_config is not None and not deeplab_checkpoint_ready and not args.skip_deeplab:
        ensure_deeplab_source_data(args, deeplab_source_config)
    ensure_data(args, runtime_config)

    if deeplab_checkpoint_ready:
        print("DeepLab checkpoint URL was provided; skipping DeepLab training.", flush=True)
    elif not args.skip_deeplab:
        train_config = deeplab_source_config or runtime_config
        run([sys.executable, "train.py", "--config", str(train_config), "--method", "deeplab"])
    if ours_no_tm_checkpoint_ready:
        print("Ours-noTM checkpoint URL was provided; skipping Ours-noTM training.", flush=True)
    elif not args.skip_ours_no_tm:
        run([sys.executable, "train.py", "--config", str(runtime_config), "--method", "ours_no_tm"])
    if ours_tm_checkpoint_ready:
        print("Ours-TM checkpoint URL was provided; skipping Ours-TM training.", flush=True)
    elif not args.skip_ours_tm:
        run([sys.executable, "train.py", "--config", str(runtime_config), "--method", "ours_tm"])

    run([sys.executable, "evaluate.py", "--config", str(runtime_config), "--skip-efficiency"])
    run([sys.executable, "evaluate.py", "--config", str(runtime_config), "--efficiency-only"])


if __name__ == "__main__":
    main()
