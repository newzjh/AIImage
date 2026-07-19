#!/usr/bin/env python
"""Download one checkpoint from the DeepFillV2 PyTorch reference project.

The upstream `download_files.py` downloads every pretrained checkpoint.  For
baseline reproduction we usually need just one candidate, so this helper keeps
the download explicit and repeatable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED_DIR = PROJECT_ROOT / "ref" / "deepfillv2" / "deepfillv2-pytorch-master" / "pretrained"

CHECKPOINTS = {
    "states_tf_places2.pth": {
        "id": "1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt",
        "url": "https://drive.google.com/file/d/1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt/view?usp=drive_link",
        "family": "networks_tf.py",
    },
    "states_tf_celebahq.pth": {
        "id": "1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm",
        "url": "https://drive.google.com/file/d/1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm/view?usp=drive_link",
        "family": "networks_tf.py",
    },
    "states_pt_places2.pth": {
        "id": "1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa",
        "url": "https://drive.google.com/file/d/1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa/view?usp=drive_link",
        "family": "networks.py",
    },
    "states_pt_celebahq.pth": {
        "id": "17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7",
        "url": "https://drive.google.com/file/d/17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7/view?usp=drive_link",
        "family": "networks.py",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one DeepFillV2 PyTorch checkpoint.")
    parser.add_argument("name", choices=sorted(CHECKPOINTS.keys()))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PRETRAINED_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.name
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} already exists ({out_path.stat().st_size} bytes)")
        return

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("gdown is required: python -m pip install gdown") from exc

    spec = CHECKPOINTS[args.name]
    print(f"Downloading {args.name} ({spec['family']}) -> {out_path}")

    # gdown 6.x no longer accepts the old fuzzy=True in some installs.  Passing
    # the Drive file-id is the most stable route for both 5.x and 6.x.
    try:
        result = gdown.download(id=spec["id"], output=str(out_path), quiet=args.quiet, use_cookies=False)
    except TypeError:
        result = gdown.download(url=spec["url"], output=str(out_path), quiet=args.quiet)

    if not result:
        raise SystemExit(f"Download failed for {args.name}")
    if not out_path.exists() or out_path.stat().st_size < 1024 * 1024:
        raise SystemExit(f"Downloaded file is suspiciously small: {out_path}")
    print(f"Downloaded {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
