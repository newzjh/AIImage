"""Compare the Unity Qwen3.5 patch-atlas output with an ncnn CPU gold run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import ncnn
import numpy as np

from vision_patch_reference import build_all_patches


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ncnn_atlas(model_dir: Path, image_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    patches, source_shape, target_shape = build_all_patches(image_path)
    target_width, target_height = target_shape
    grid_width = target_width // 16
    grid_height = target_height // 16
    if patches.shape != (grid_width * grid_height, 3, 2, 16, 16):
        raise RuntimeError(f"unexpected patch shape: {patches.shape}")

    atlas = (
        patches.reshape(grid_height, grid_width, 3, 2, 16, 16)
        .transpose(2, 3, 0, 4, 1, 5)
        .reshape(3, 2, target_height, target_width)
    )
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_packing_layout = False
    if net.load_param(str(model_dir / "qwen3.5_vision_embed_patch.ncnn.param")) != 0:
        raise RuntimeError("failed to load vision patch param")
    if net.load_model(str(model_dir / "qwen3.5_vision_embed_patch.ncnn.bin")) != 0:
        raise RuntimeError("failed to load vision patch weights")
    extractor = net.create_extractor()
    if extractor.input("in0", ncnn.Mat(np.ascontiguousarray(atlas))) != 0:
        raise RuntimeError("failed to feed vision patch atlas")
    status, output = extractor.extract("out0")
    if status != 0:
        raise RuntimeError(f"failed to extract vision patch atlas output: {status}")

    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0, :, :]
    expected_shape = (768, grid_height, grid_width)
    if raw.shape != expected_shape:
        raise RuntimeError(f"unexpected ncnn atlas output shape {raw.shape}, expected {expected_shape}")
    token_major = np.ascontiguousarray(raw.transpose(1, 2, 0).reshape(-1))
    return token_major, {
        "source_width": source_shape[0],
        "source_height": source_shape[1],
        "target_width": target_width,
        "target_height": target_height,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "patch_count": grid_width * grid_height,
    }


def compare(
    model_dir: Path,
    image_path: Path,
    unity_dump: Path,
    unity_report: Path,
    reference_dump: Path,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    reference, shape = run_ncnn_atlas(model_dir, image_path)
    unity = np.fromfile(unity_dump, dtype="<f4")
    reference_dump.parent.mkdir(parents=True, exist_ok=True)
    reference.astype("<f4", copy=False).tofile(reference_dump)
    if unity.size != reference.size:
        raise RuntimeError(f"Unity/reference count mismatch: {unity.size} != {reference.size}")

    absolute = np.abs(unity - reference)
    allowed = np.float32(atol) + np.float32(rtol) * np.abs(reference)
    failures = np.flatnonzero(absolute > allowed)
    first_failure = None
    if failures.size:
        flat_index = int(failures[0])
        first_failure = {
            "flat_index": flat_index,
            "patch_index": flat_index // 768,
            "feature_index": flat_index % 768,
            "unity": float(unity[flat_index]),
            "reference": float(reference[flat_index]),
            "abs_error": float(absolute[flat_index]),
            "allowed_error": float(allowed[flat_index]),
        }

    worst_index = int(np.argmax(absolute))
    selected = {}
    for patch_index in (0, 1, 2, 3, 1024, shape["patch_count"] - 1):
        start = patch_index * 768
        selected[str(patch_index)] = {
            "unity": unity[start : start + 16].tolist(),
            "reference": reference[start : start + 16].tolist(),
            "max_abs_error": float(np.max(absolute[start : start + 768])),
        }

    return {
        "schema": "qwen35.compare.vision-patch-atlas/v1",
        "model_directory": str(model_dir.resolve()),
        "image": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        **shape,
        "element_count": int(reference.size),
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": bool(np.isfinite(unity).all() and np.isfinite(reference).all()),
        "failure_count": int(failures.size),
        "first_failure": first_failure,
        "max_abs_error": float(absolute[worst_index]),
        "max_abs_error_location": {
            "flat_index": worst_index,
            "patch_index": worst_index // 768,
            "feature_index": worst_index % 768,
            "unity": float(unity[worst_index]),
            "reference": float(reference[worst_index]),
        },
        "mean_abs_error": float(np.mean(absolute, dtype=np.float64)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "selected_patch_previews": selected,
        "unity_report": str(unity_report.resolve()),
        "unity_report_sha256": sha256_file(unity_report),
        "unity_dump": str(unity_dump.resolve()),
        "unity_dump_sha256": sha256_file(unity_dump),
        "reference_dump": str(reference_dump.resolve()),
        "reference_dump_sha256": sha256_file(reference_dump),
        "valid": bool(not failures.size and np.isfinite(unity).all() and np.isfinite(reference).all()),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    tool_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=tool_dir / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--image", type=Path, default=root / "ref" / "ncnn_llm-main" / "test.jpg")
    parser.add_argument("--unity-dump", type=Path, default=tool_dir / "reports" / "unity_vision_patch_atlas_probe.f32")
    parser.add_argument("--unity-report", type=Path, default=tool_dir / "reports" / "unity_vision_patch_atlas_probe.json")
    parser.add_argument("--reference-dump", type=Path, default=tool_dir / "reports" / "ncnn_vision_patch_atlas_reference.f32")
    parser.add_argument("--output", type=Path, default=tool_dir / "reports" / "vision_patch_atlas_compare.json")
    parser.add_argument("--atol", type=float, default=0.0005)
    parser.add_argument("--rtol", type=float, default=0.0008)
    args = parser.parse_args()
    report = compare(
        args.model_dir,
        args.image,
        args.unity_dump,
        args.unity_report,
        args.reference_dump,
        args.atol,
        args.rtol,
    )
    report["command"] = "python " + " ".join(shlex.quote(part) for part in __import__("sys").argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid", "failure_count", "max_abs_error", "mean_abs_error")}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
