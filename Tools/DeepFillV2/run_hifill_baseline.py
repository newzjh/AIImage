#!/usr/bin/env python3
"""Run the 512x512 DeepFillV2 HiFill reference graph with an image and YOLO mask."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def load(path: Path, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size, resample), dtype=np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "Tools/DeepFillV2/output/hifill.standard.onnx")
    parser.add_argument("--image", type=Path, default=root / "Documents/ClipCompareInput/03.jpg")
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "Tools/DeepFillV2/output/hifill_baseline_03")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(args.image) as original_image:
        original_size = original_image.size
    image = load(args.image, (512, 512), Image.Resampling.LANCZOS)
    raw_mask = load(args.mask, (512, 512), Image.Resampling.NEAREST)
    mask = (raw_mask.max(axis=2, keepdims=True) >= 128.0).astype(np.float32)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    output = session.run(None, {"img:0": image[None], "mask:0": mask[None]})[0][0]
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    result = Image.fromarray(output, "RGB")
    result.save(args.output_dir / "03_hifill_512.png")
    result.resize(original_size, Image.Resampling.LANCZOS).save(args.output_dir / "03_hifill.png")
    Image.fromarray((mask[..., 0] * 255.0).astype(np.uint8), "L").save(args.output_dir / "03_yolo_mask_512.png")
    report = {
        "model": str(args.model.resolve()), "image": str(args.image.resolve()), "mask": str(args.mask.resolve()),
        "input_shape": [1, 512, 512, 3], "output_shape": list(output.shape), "elapsed_ms": elapsed_ms,
        "mask_coverage": float(mask.mean()), "mask_semantics": "white=inpaint", "status": "passed",
    }
    (args.output_dir / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
