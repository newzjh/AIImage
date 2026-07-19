"""Download the pinned Qwen3.5 ncnn model with resume and size checks."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from model_manifest import BASE_URL, ModelFile, files_for_group


CHUNK_SIZE = 4 * 1024 * 1024


def human_size(value: int) -> str:
    size = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or suffix == "GiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    raise AssertionError("unreachable")


def validate_manifest_item(item: ModelFile) -> None:
    if "/" in item.name or "\\" in item.name or item.name in {"", ".", ".."}:
        raise ValueError(f"unsafe manifest filename: {item.name!r}")
    if not item.url.startswith(BASE_URL):
        raise ValueError(f"model URL escaped the pinned mirror: {item.url}")


def download_one(item: ModelFile, output_dir: Path) -> str:
    validate_manifest_item(item)
    destination = output_dir / item.name
    partial = output_dir / f"{item.name}.part"

    if destination.exists():
        actual = destination.stat().st_size
        if actual == item.size:
            return "verified"
        raise RuntimeError(
            f"existing file has the wrong size: {destination} "
            f"({actual} != {item.size}); move it aside before retrying"
        )

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item.size:
        partial.unlink()
        offset = 0
    if offset == item.size:
        os.replace(partial, destination)
        return "completed"

    headers = {"User-Agent": "AIImage-Qwen35NcnnBaseline/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item.url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"download failed for {item.url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"download failed for {item.url}: {exc.reason}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
        if offset and response.headers.get("Content-Range", "").split(" ")[-1].split("-")[0] != str(offset):
            raise RuntimeError(f"server returned an unexpected Content-Range for {item.name}")

        mode = "ab" if offset else "wb"
        downloaded = offset
        started = time.monotonic()
        last_report = started
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2.0:
                    elapsed = max(now - started, 1e-6)
                    speed = (downloaded - offset) / elapsed
                    print(
                        f"  {item.name}: {human_size(downloaded)} / "
                        f"{human_size(item.size)} ({human_size(int(speed))}/s)",
                        flush=True,
                    )
                    last_report = now
            stream.flush()
            os.fsync(stream.fileno())

    actual = partial.stat().st_size
    if actual != item.size:
        raise RuntimeError(
            f"size check failed for {item.name}: got {actual}, expected {item.size}; "
            "the .part file was kept for resume"
        )
    os.replace(partial, destination)
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "_models" / "qwen3.5_0.8b",
        help="model output directory",
    )
    parser.add_argument(
        "--group",
        choices=("metadata", "text", "vision", "all"),
        default="metadata",
        help="metadata is the small default; text/vision include their weight files",
    )
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="download all 3.18 GiB of text and vision weights (same as --group all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print selected files without downloading")
    parser.add_argument("--verify-only", action="store_true", help="check existing file sizes only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    group = "all" if args.with_weights else args.group
    selected = files_for_group(group)
    total = sum(item.size for item in selected)

    print(f"Pinned source: {BASE_URL}")
    print(f"Selection: {group}, {len(selected)} files, {human_size(total)}")
    for item in selected:
        print(f"  {item.name:<44} {human_size(item.size):>10}  {item.url}")

    if args.dry_run:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for item in selected:
        destination = args.output / item.name
        if args.verify_only:
            actual = destination.stat().st_size if destination.exists() else -1
            if actual != item.size:
                failures.append(f"{item.name}: got {actual}, expected {item.size}")
            continue
        try:
            status = download_one(item, args.output)
            print(f"[{status}] {item.name}")
        except Exception as exc:  # Keep independent downloads diagnosable.
            failures.append(f"{item.name}: {exc}")
            print(f"[failed] {item.name}: {exc}", file=sys.stderr)

    if failures:
        print("Model verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 2
    print(f"Model files are ready under {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

