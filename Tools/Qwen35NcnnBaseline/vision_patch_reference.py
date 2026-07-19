"""Generate an auditable ncnn CPU reference for Qwen3.5 vision patch zero."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

from vision_reference import reorder_patches_for_merge, target_image_size


def _saturate_short(value: float) -> np.int16:
    rounded = int(value + (0.5 if value >= 0.0 else -0.5))
    return np.int16(max(-32768, min(32767, rounded)))


def resize_bilinear_c3_ncnn(source: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Match ncnn::resize_bilinear_c3's 11-bit fixed-point byte path."""
    source = np.ascontiguousarray(source, dtype=np.uint8)
    source_height, source_width, channels = source.shape
    if channels != 3:
        raise ValueError("source must be uint8 RGB")
    if source_width == target_width and source_height == target_height:
        return source.copy()
    if source_width < 2 or source_height < 2:
        y = np.minimum(np.arange(target_height) * source_height // target_height, source_height - 1)
        x = np.minimum(np.arange(target_width) * source_width // target_width, source_width - 1)
        return np.ascontiguousarray(source[y[:, None], x[None, :]])

    coefficient_scale = 1 << 11
    x_offsets = np.empty(target_width, dtype=np.int32)
    y_offsets = np.empty(target_height, dtype=np.int32)
    alpha = np.empty((target_width, 2), dtype=np.int16)
    beta = np.empty((target_height, 2), dtype=np.int16)
    scale_x = source_width / target_width
    scale_y = source_height / target_height
    for x in range(target_width):
        fx = np.float32((x + 0.5) * scale_x - 0.5)
        sx = math.floor(float(fx))
        fx = np.float32(fx - sx)
        if sx < 0:
            sx, fx = 0, np.float32(0.0)
        if sx >= source_width - 1:
            sx, fx = source_width - 2, np.float32(1.0)
        x_offsets[x] = sx
        alpha[x, 0] = _saturate_short(float(np.float32((1.0 - fx) * coefficient_scale)))
        alpha[x, 1] = _saturate_short(float(np.float32(fx * coefficient_scale)))
    for y in range(target_height):
        fy = np.float32((y + 0.5) * scale_y - 0.5)
        sy = math.floor(float(fy))
        fy = np.float32(fy - sy)
        if sy < 0:
            sy, fy = 0, np.float32(0.0)
        if sy >= source_height - 1:
            sy, fy = source_height - 2, np.float32(1.0)
        y_offsets[y] = sy
        beta[y, 0] = _saturate_short(float(np.float32((1.0 - fy) * coefficient_scale)))
        beta[y, 1] = _saturate_short(float(np.float32(fy * coefficient_scale)))

    target = np.empty((target_height, target_width, 3), dtype=np.uint8)
    for y in range(target_height):
        sy = y_offsets[y]
        b0, b1 = int(beta[y, 0]), int(beta[y, 1])
        for x in range(target_width):
            sx = x_offsets[x]
            a0, a1 = int(alpha[x, 0]), int(alpha[x, 1])
            for channel in range(3):
                row0 = (int(source[sy, sx, channel]) * a0 + int(source[sy, sx + 1, channel]) * a1) >> 4
                row1 = (int(source[sy + 1, sx, channel]) * a0 + int(source[sy + 1, sx + 1, channel]) * a1) >> 4
                value = (((b0 * row0) >> 16) + ((b1 * row1) >> 16) + 2) >> 2
                target[y, x, channel] = np.uint8(max(0, min(255, value)))
    return target


def build_all_patches(image_path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    source_shape = (int(rgb.shape[1]), int(rgb.shape[0]))
    target_height, target_width = target_image_size(rgb.shape[0], rgb.shape[1])
    resized = resize_bilinear_c3_ncnn(rgb, target_width, target_height)
    normalized = (resized.astype(np.float32) / np.float32(255.5) - np.float32(0.5)) / np.float32(0.5)
    height_patches = target_height // 16
    width_patches = target_width // 16
    patches = (
        normalized.reshape(height_patches, 16, width_patches, 16, 3)
        .transpose(0, 2, 4, 1, 3)
        .reshape(height_patches * width_patches, 3, 16, 16)
    )
    patches = reorder_patches_for_merge(patches, height_patches, width_patches, 2)
    duplicated = np.ascontiguousarray(np.repeat(patches[:, :, None, :, :], 2, axis=2))
    return duplicated, source_shape, (target_width, target_height)


def build_patch_zero(image_path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    patches, source_shape, target_shape = build_all_patches(image_path)
    return np.ascontiguousarray(patches[0]), source_shape, target_shape


def run_reference(model_dir: Path, image_path: Path) -> dict[str, object]:
    patch, source_shape, target_shape = build_patch_zero(image_path)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    if net.load_param(str(model_dir / "qwen3.5_vision_embed_patch.ncnn.param")) != 0:
        raise RuntimeError("failed to load vision patch param")
    if net.load_model(str(model_dir / "qwen3.5_vision_embed_patch.ncnn.bin")) != 0:
        raise RuntimeError("failed to load vision patch weights")
    extractor = net.create_extractor()
    if extractor.input("in0", ncnn.Mat(patch)) != 0:
        raise RuntimeError("failed to feed vision patch input")
    status, output = extractor.extract("out0")
    if status != 0:
        raise RuntimeError(f"failed to extract vision patch output: {status}")
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    flat_patch = patch.reshape(-1)
    return {
        "schema": "qwen35.ncnn.vision-patch-reference/v1",
        "model_directory": str(model_dir.resolve()),
        "image": str(image_path.resolve()),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "source_width": source_shape[0],
        "source_height": source_shape[1],
        "target_width": target_shape[0],
        "target_height": target_shape[1],
        "patch_index": 0,
        "input_shape_cdhw": [3, 2, 16, 16],
        "input_values": flat_patch.tolist(),
        "output_count": int(values.size),
        "output_values": values.tolist(),
        "finite": bool(np.isfinite(values).all()),
        "nonzero_count": int(np.count_nonzero(values)),
        "max_abs": float(np.max(np.abs(values))),
        "valid": bool(values.size == 768 and np.isfinite(values).all() and np.count_nonzero(values) > 0),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--image", type=Path, default=root / "ref" / "ncnn_llm-main" / "test.jpg")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports" / "ncnn_vision_patch_reference.json")
    args = parser.parse_args()
    report = run_reference(args.model_dir, args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("valid", "output_count", "nonzero_count", "max_abs")}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
