from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SN8_TARBALL_PREFIX = "s3://spacenet-dataset/spacenet/SN8_floods/tarballs/"
DEFAULT_SN8_TARBALL = "Louisiana-East_Training_Public.tar.gz"
SAM3_INSTALL_URL = "git+https://github.com/facebookresearch/sam3.git"

NUMERIC_STACK = [
    "numpy>=1.26,<2",
    "pandas>=2.2,<2.4",
    "scipy>=1.11,<1.17",
    "scikit-learn>=1.4,<1.8",
    "rasterio>=1.4,<1.5",
    "geopandas>=0.14,<1.2",
    "shapely>=2.0,<2.2",
    "pyogrio>=0.7,<0.12",
    "pyproj>=3.6,<3.8",
    "opencv-python-headless>=4.8,<4.11",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Colab runtime for the real FloodRoad-SAM3 run.")
    parser.add_argument("--output", default="/content/floodroad_runs/default/colab_validation.json")
    parser.add_argument("--skip-sam3-install", action="store_true")
    parser.add_argument("--sam3-install-url", default=SAM3_INSTALL_URL)
    return parser.parse_args()


def run(cmd: list[str], *, check: bool = True, capture: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        text=True,
        check=False,
        capture_output=capture,
        env=env or os.environ.copy(),
    )
    if not capture:
        pass
    elif result.stdout:
        print(result.stdout, end="")
    if capture and result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def run_python(code: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run([sys.executable, "-c", code], check=check, capture=capture)


def aws_command() -> list[str] | None:
    venv_aws = Path(sys.executable).resolve().parent / "aws"
    if venv_aws.exists():
        return [str(venv_aws)]
    system_aws = shutil.which("aws")
    if system_aws:
        return [system_aws]
    return None


def ensure_gpu() -> dict[str, Any]:
    print("===== GPU =====", flush=True)
    print("Python executable:", sys.executable, flush=True)
    run(["nvidia-smi", "-L"], check=False)
    run(["nvidia-smi"], check=False)
    code = r'''
import json
import sys
import torch
info = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(info))
if not info["cuda_available"]:
    raise SystemExit("CUDA GPU is not available. Switch Colab Runtime > Change runtime type > GPU.")
'''
    result = run_python(code, capture=True)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    print("Python:", info["python"], flush=True)
    print("Torch:", info["torch"], flush=True)
    print("CUDA available:", info["cuda_available"], flush=True)
    print("Device:", info["device"], flush=True)
    return info


def list_sn8_tarballs() -> str:
    cmd = aws_command()
    if cmd is None:
        print("aws command was not found; installing awscli into the active Python environment.", flush=True)
        pip_install(["-q", "awscli"])
        cmd = aws_command()
    if cmd is None:
        raise RuntimeError(f"awscli installed, but no aws executable was found next to {sys.executable}")
    result = run([*cmd, "s3", "ls", SN8_TARBALL_PREFIX, "--no-sign-request"], capture=True, check=False)
    if result.returncode != 0:
        print("awscli failed; reinstalling awscli and retrying.", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q", "awscli"])
        cmd = aws_command()
        if cmd is None:
            raise RuntimeError(f"awscli reinstall finished, but no aws executable was found next to {sys.executable}")
        result = run([*cmd, "s3", "ls", SN8_TARBALL_PREFIX, "--no-sign-request"], capture=True)
    return result.stdout


def select_sn8_tarball() -> str:
    print("\n===== SpaceNet 8 public tarballs =====", flush=True)
    listing = list_sn8_tarballs()
    lines = [line for line in listing.splitlines() if line.strip()]
    for line in lines:
        if "Louisiana" in line or "louisiana" in line:
            print(line, flush=True)
    names = [line.split()[-1] for line in lines]
    tarballs = [name for name in names if name.endswith(".tar.gz")]
    if not tarballs:
        raise RuntimeError("No .tar.gz files were listed under the SpaceNet 8 tarballs prefix.")
    if DEFAULT_SN8_TARBALL in tarballs:
        selected = DEFAULT_SN8_TARBALL
    else:
        candidates = [name for name in tarballs if "Louisiana" in name and "Training" in name]
        east = [name for name in candidates if "East" in name]
        selected = (east or candidates or tarballs)[0]
    print("Selected SN8 tarball:", selected, flush=True)
    return selected


def pip_install(args: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    cmd = [sys.executable, "-m", "pip", "install"]
    if "--no-warn-conflicts" not in args:
        cmd.append("--no-warn-conflicts")
    run([*cmd, *args], env=env)


def ensure_sam3_installed(install_url: str, skip_install: bool) -> bool:
    print("\n===== SAM3 package install =====", flush=True)
    probe = run_python("import sam3; print('sam3 import ok')", check=False, capture=True)
    if probe.returncode == 0:
        return False
    print("Initial sam3 import failed; installing official package.", flush=True)
    if probe.stderr:
        print(probe.stderr, flush=True)
    if skip_install:
        raise RuntimeError("SAM3 is not importable and --skip-sam3-install was requested.")
    pip_install(["-q", install_url])
    return True


def package_probe() -> subprocess.CompletedProcess[str]:
    code = r'''
import importlib
import json

names = ["numpy", "pandas", "scipy", "sklearn", "rasterio", "cv2", "networkx", "geopandas", "shapely", "yaml", "thop"]
versions = {}
for name in names:
    mod = importlib.import_module(name)
    versions[name] = getattr(mod, "__version__", "ok")
print(json.dumps(versions, sort_keys=True))
'''
    return run_python(code, check=False, capture=True)


def repair_numeric_stack(reason: str) -> None:
    print("\n===== Repair NumPy/binary package stack =====", flush=True)
    print(reason, flush=True)
    pip_install(["-q", "--upgrade", "--force-reinstall", "--no-cache-dir", *NUMERIC_STACK])


def ensure_numeric_stack(installed_sam3: bool) -> dict[str, str]:
    print("\n===== Key package imports =====", flush=True)
    probe = package_probe()
    needs_repair = installed_sam3 or probe.returncode != 0
    reason = "SAM3 was just installed; refreshing NumPy-linked wheels for a consistent Colab ABI."
    if probe.returncode != 0:
        reason = "Package import probe failed, usually due to a NumPy ABI mismatch. Refreshing NumPy-linked wheels."
        if probe.stdout:
            print(probe.stdout, flush=True)
        if probe.stderr:
            print(probe.stderr, flush=True)
    if not needs_repair:
        versions = json.loads(probe.stdout.strip().splitlines()[-1])
        if str(versions.get("numpy", "")).startswith("2."):
            needs_repair = True
            reason = "NumPy 2.x is active; this run pins NumPy <2 for the SAM3/geospatial stack."
    if needs_repair:
        repair_numeric_stack(reason)
        probe = package_probe()
    if probe.returncode != 0:
        print(probe.stdout, flush=True)
        print(probe.stderr, flush=True)
        raise RuntimeError("Package import probe still failed after dependency repair.")
    versions = json.loads(probe.stdout.strip().splitlines()[-1])
    for name, version in versions.items():
        print(name, version, flush=True)
    return versions


def probe_sam3_entrypoints() -> list[list[str]]:
    print("\n===== SAM3 import/API probe =====", flush=True)
    code = r'''
import importlib
import json

candidates = [
    ("sam3", "build_sam3_image_model"),
    ("sam3.model_builder", "build_sam3_image_model"),
    ("sam3", "build_sam3"),
    ("segment_anything_3", "build_sam3"),
    ("segment_anything_3", "sam_model_registry"),
]
hits = []
for module_name, attr_name in candidates:
    try:
        mod = importlib.import_module(module_name)
        attr = getattr(mod, attr_name)
        print("OK", module_name, attr_name, attr)
        hits.append([module_name, attr_name])
    except Exception as exc:
        print("NO", module_name, attr_name, repr(exc))
print("SAM3_HITS_JSON=" + json.dumps(hits))
if not hits:
    raise SystemExit("SAM3 installed, but none of the adapter's known entry points were found.")
'''
    result = run_python(code, capture=True)
    hits: list[list[str]] = []
    for line in result.stdout.splitlines():
        if line.startswith("SAM3_HITS_JSON="):
            hits = json.loads(line.split("=", 1)[1])
    if not hits:
        raise RuntimeError("SAM3 API probe produced no compatible entry points.")
    return hits


def write_output(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print("\nWrote validation metadata:", out, flush=True)


def main() -> None:
    args = parse_args()
    gpu_info = ensure_gpu()
    sn8_tarball = select_sn8_tarball()
    installed_sam3 = ensure_sam3_installed(args.sam3_install_url, args.skip_sam3_install)
    package_versions = ensure_numeric_stack(installed_sam3)
    sam3_hits = probe_sam3_entrypoints()
    payload = {
        "sn8_tarball": sn8_tarball,
        "sam3_hits": sam3_hits,
        "gpu": gpu_info,
        "package_versions": package_versions,
    }
    write_output(args.output, payload)
    print("\nValidation passed. The formal real-data run can start.", flush=True)


if __name__ == "__main__":
    main()
