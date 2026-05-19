from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SpaceNet 8 files from the public AWS bucket.")
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument("--prefix", default="spacenet8/", help="S3 prefix inside s3://spacenet-dataset")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws",
        "s3",
        "sync",
        f"s3://spacenet-dataset/{args.prefix}",
        str(out),
        "--no-sign-request",
    ]
    if args.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

