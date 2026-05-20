from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SpaceNet 8 files from the public AWS bucket.")
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument("--prefix", default="spacenet/SN8_floods/", help="S3 prefix inside s3://spacenet-dataset")
    parser.add_argument("--tarball", default=None, help="Optional SN8 tarball name under spacenet/SN8_floods/tarballs")
    parser.add_argument("--extract", action="store_true", help="Extract a downloaded .tar.gz file")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.tarball:
        key = f"spacenet/SN8_floods/tarballs/{args.tarball}"
        dest = out / args.tarball
        cmd = ["aws", "s3", "cp", f"s3://spacenet-dataset/{key}", str(dest), "--no-sign-request"]
        if args.dry_run:
            print(" ".join(cmd))
            return
        subprocess.run(cmd, check=True)
        if args.extract:
            with tarfile.open(dest, "r:gz") as tar:
                tar.extractall(out)
        return
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
