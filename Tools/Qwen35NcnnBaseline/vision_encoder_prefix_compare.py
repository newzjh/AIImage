"""Build and compare the Qwen3.5 vision encoder add_0 (blob 52) gold output."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import ncnn
import numpy as np

from vision_reference import reorder_patches_for_merge


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_position_reference(model_dir: Path, grid_width: int, grid_height: int) -> np.ndarray:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_packing_layout = False
    if net.load_param(str(model_dir / "qwen3.5_vision_embed_pos.ncnn.param")) != 0:
        raise RuntimeError("failed to load vision position param")
    if net.load_model(str(model_dir / "qwen3.5_vision_embed_pos.ncnn.bin")) != 0:
        raise RuntimeError("failed to load vision position weights")
    extractor = net.create_extractor()
    grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    if extractor.input("in0", ncnn.Mat(grid)) != 0:
        raise RuntimeError("failed to feed vision position grid")
    status, output = extractor.extract("out0")
    if status != 0:
        raise RuntimeError(f"failed to extract vision position output: {status}")
    values = np.asarray(output, dtype=np.float32).reshape(grid_height * grid_width, 768)
    return reorder_patches_for_merge(values, grid_height, grid_width, 2)


def build_report(
    model_dir: Path,
    patch_reference_dump: Path,
    unity_dump: Path,
    unity_report: Path,
    reference_dump: Path,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    grid_width = 64
    grid_height = 48
    patch_count = grid_width * grid_height
    patch = np.fromfile(patch_reference_dump, dtype="<f4").reshape(patch_count, 768)
    position = run_position_reference(model_dir, grid_width, grid_height)
    reference = np.ascontiguousarray(patch + position).reshape(-1)
    unity = np.fromfile(unity_dump, dtype="<f4")
    if unity.size != reference.size:
        raise RuntimeError(f"Unity/reference count mismatch: {unity.size} != {reference.size}")
    reference_dump.parent.mkdir(parents=True, exist_ok=True)
    reference.astype("<f4", copy=False).tofile(reference_dump)

    absolute = np.abs(unity - reference)
    allowed = np.float32(atol) + np.float32(rtol) * np.abs(reference)
    failures = np.flatnonzero(absolute > allowed)
    first_failure = None
    if failures.size:
        index = int(failures[0])
        first_failure = {
            "flat_index": index,
            "patch_index": index // 768,
            "feature_index": index % 768,
            "unity": float(unity[index]),
            "reference": float(reference[index]),
            "abs_error": float(absolute[index]),
            "allowed_error": float(allowed[index]),
        }
    worst = int(np.argmax(absolute))
    return {
        "schema": "qwen35.compare.vision-encoder-prefix/v1",
        "checkpoint": "vision_encoder/0006_add_0/52",
        "model_directory": str(model_dir.resolve()),
        "grid_width": grid_width,
        "grid_height": grid_height,
        "patch_count": patch_count,
        "element_count": int(reference.size),
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": bool(np.isfinite(unity).all() and np.isfinite(reference).all()),
        "failure_count": int(failures.size),
        "first_failure": first_failure,
        "max_abs_error": float(absolute[worst]),
        "max_abs_error_location": {
            "flat_index": worst,
            "patch_index": worst // 768,
            "feature_index": worst % 768,
            "unity": float(unity[worst]),
            "reference": float(reference[worst]),
        },
        "mean_abs_error": float(np.mean(absolute, dtype=np.float64)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "patch_reference_dump": str(patch_reference_dump.resolve()),
        "patch_reference_dump_sha256": sha256_file(patch_reference_dump),
        "unity_report": str(unity_report.resolve()),
        "unity_report_sha256": sha256_file(unity_report),
        "unity_dump": str(unity_dump.resolve()),
        "unity_dump_sha256": sha256_file(unity_dump),
        "reference_dump": str(reference_dump.resolve()),
        "reference_dump_sha256": sha256_file(reference_dump),
        "valid": bool(not failures.size and np.isfinite(unity).all() and np.isfinite(reference).all()),
    }


def main() -> int:
    tool_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=tool_dir / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--patch-reference-dump", type=Path, default=tool_dir / "reports" / "ncnn_vision_patch_atlas_reference.f32")
    parser.add_argument("--unity-dump", type=Path, default=tool_dir / "reports" / "unity_vision_encoder_prefix_probe.f32")
    parser.add_argument("--unity-report", type=Path, default=tool_dir / "reports" / "unity_vision_encoder_prefix_probe.json")
    parser.add_argument("--reference-dump", type=Path, default=tool_dir / "reports" / "ncnn_vision_encoder_blob52_reference.f32")
    parser.add_argument("--output", type=Path, default=tool_dir / "reports" / "vision_encoder_blob52_compare.json")
    parser.add_argument("--atol", type=float, default=0.00002)
    parser.add_argument("--rtol", type=float, default=0.00002)
    args = parser.parse_args()
    report = build_report(
        args.model_dir,
        args.patch_reference_dump,
        args.unity_dump,
        args.unity_report,
        args.reference_dump,
        args.atol,
        args.rtol,
    )
    report["command"] = "py -3.10 " + " ".join(shlex.quote(part) for part in sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid", "failure_count", "max_abs_error", "mean_abs_error")}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
