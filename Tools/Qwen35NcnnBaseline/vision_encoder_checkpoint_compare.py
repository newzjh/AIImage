"""Compare a Unity Qwen3.5 vision-encoder checkpoint with ncnn CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import ncnn
import numpy as np

from vision_encoder_prefix_compare import run_position_reference
from vision_reference import vision_rope_2d


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checkpoint(
    model_dir: Path,
    patch_reference_dump: Path,
    checkpoint: str,
    input_blob52_dump: Path | None = None,
) -> np.ndarray:
    grid_width = 64
    grid_height = 48
    patch_count = grid_width * grid_height
    if input_blob52_dump is None:
        patch = np.fromfile(patch_reference_dump, dtype="<f4").reshape(patch_count, 768)
        position = run_position_reference(model_dir, grid_width, grid_height)
    else:
        patch = np.fromfile(input_blob52_dump, dtype="<f4").reshape(patch_count, 768)
        position = np.zeros_like(patch)
    cos, sin = vision_rope_2d(grid_height, grid_width, merge=2, theta=10_000.0, section=(16, 16))

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_packing_layout = False
    if net.load_param(str(model_dir / "qwen3.5_vision_encoder.ncnn.param")) != 0:
        raise RuntimeError("failed to load vision encoder param")
    if net.load_model(str(model_dir / "qwen3.5_vision_encoder.ncnn.bin")) != 0:
        raise RuntimeError("failed to load vision encoder weights")
    extractor = net.create_extractor()
    if hasattr(extractor, "set_light_mode"):
        extractor.set_light_mode(False)
    for name, values in (("in0", patch), ("in1", position), ("in2", cos), ("in3", sin)):
        if extractor.input(name, ncnn.Mat(np.ascontiguousarray(values))) != 0:
            raise RuntimeError(f"failed to feed vision encoder {name}")
    status, output = extractor.extract(checkpoint)
    if status != 0:
        raise RuntimeError(f"failed to extract vision encoder checkpoint {checkpoint}: {status}")
    return np.ascontiguousarray(np.asarray(output, dtype=np.float32)).reshape(-1)


def compare(
    model_dir: Path,
    patch_reference_dump: Path,
    checkpoint: str,
    checkpoint_name: str,
    unity_dump: Path,
    unity_report: Path,
    reference_dump: Path,
    atol: float,
    rtol: float,
    input_blob52_dump: Path | None = None,
) -> dict[str, object]:
    reference = run_checkpoint(model_dir, patch_reference_dump, checkpoint, input_blob52_dump)
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
            "unity": float(unity[index]),
            "reference": float(reference[index]),
            "abs_error": float(absolute[index]),
            "allowed_error": float(allowed[index]),
        }
    worst = int(np.argmax(absolute))
    return {
        "schema": "qwen35.compare.vision-encoder-checkpoint/v1",
        "checkpoint": checkpoint_name,
        "blob": checkpoint,
        "model_directory": str(model_dir.resolve()),
        "ncnn_options": {"use_vulkan_compute": False, "use_packing_layout": False},
        "gold_mode": "canonical_fp32_logical_unpacked",
        "element_count": int(reference.size),
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": bool(np.isfinite(unity).all() and np.isfinite(reference).all()),
        "failure_count": int(failures.size),
        "first_failure": first_failure,
        "max_abs_error": float(absolute[worst]),
        "max_abs_error_location": {
            "flat_index": worst,
            "unity": float(unity[worst]),
            "reference": float(reference[worst]),
        },
        "mean_abs_error": float(np.mean(absolute, dtype=np.float64)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "patch_reference_dump": str(patch_reference_dump.resolve()),
        "patch_reference_dump_sha256": sha256_file(patch_reference_dump),
        "input_mode": "unity_blob52_operator_isolation" if input_blob52_dump is not None else "end_to_end_ncnn_frontend",
        "input_blob52_dump": str(input_blob52_dump.resolve()) if input_blob52_dump is not None else None,
        "input_blob52_dump_sha256": sha256_file(input_blob52_dump) if input_blob52_dump is not None else None,
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--unity-dump", type=Path, required=True)
    parser.add_argument("--unity-report", type=Path, required=True)
    parser.add_argument("--reference-dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--input-blob52-dump", type=Path)
    args = parser.parse_args()
    report = compare(
        args.model_dir,
        args.patch_reference_dump,
        args.checkpoint,
        args.checkpoint_name,
        args.unity_dump,
        args.unity_report,
        args.reference_dump,
        args.atol,
        args.rtol,
        args.input_blob52_dump,
    )
    report["command"] = "py -3.10 " + " ".join(shlex.quote(part) for part in sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid", "failure_count", "max_abs_error", "mean_abs_error")}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
