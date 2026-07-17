#!/usr/bin/env python3
"""Run the normalized DeepFillV2 model on an RGB image and a white-hole mask."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_WIDTH = 1920
MODEL_HEIGHT = 1080


def load_rgb(path: Path, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size, resample), dtype=np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "Tools/DeepFillV2/output/deepfillv2.standard.onnx")
    parser.add_argument("--image", type=Path, default=root / "Documents/ClipCompareInput/03.jpg")
    parser.add_argument("--mask", type=Path, required=True, help="YOLO person mask: white denotes pixels to inpaint.")
    parser.add_argument("--output-dir", type=Path, default=root / "Tools/DeepFillV2/output/baseline_03")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = load_rgb(args.image, (MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)
    mask_rgb = load_rgb(args.mask, (MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.NEAREST)
    binary_mask = np.where(mask_rgb.max(axis=2, keepdims=True) >= 128.0, 255.0, 0.0).astype(np.float32)
    packed = np.concatenate((original, np.repeat(binary_mask, 3, axis=2)), axis=1)[None, ...]

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    output = session.run(None, {session.get_inputs()[0].name: packed})[0]
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    repaired = Image.fromarray(output[0].astype(np.uint8), "RGB")
    repaired.save(args.output_dir / "03_deepfillv2_1920x1080.png")
    repaired.resize(Image.open(args.image).size, Image.Resampling.LANCZOS).save(args.output_dir / "03_deepfillv2.png")
    Image.fromarray(binary_mask[..., 0].astype(np.uint8), "L").save(args.output_dir / "03_yolo_mask_1920x1080.png")
    report = {
        "model": str(args.model.resolve()), "image": str(args.image.resolve()), "mask": str(args.mask.resolve()),
        "input_shape": list(packed.shape), "output_shape": list(output.shape), "elapsed_ms": elapsed_ms,
        "mask_coverage": float((binary_mask >= 128.0).mean()), "mask_semantics": "white=inpaint",
    }
    (args.output_dir / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
