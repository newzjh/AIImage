#!/usr/bin/env python3
"""Retarget the fixed-shape official DeepFillV2 ONNX to a smaller HxW canvas.

The official TensorFlow-exported graph in ref/deepfillv2/deepfillv2.onnx is
frozen for [1, 1080, 3840, 3] input and [1, 1080, 1920, 3] output.  Its
contextual-attention branch materializes tensors over a 135x240 grid, which
requires several GB on CPU ORT.  The PyTorch official sample set is 512x680,
so this helper rewrites only the static shape constants that encode H/W and
the H/8 x W/8 attention grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


PATCHED_INT_INITIALIZERS = {
    "const_fold_opt__250": lambda h, w: [1, h // 8, w // 8, (h // 8) * (w // 8)],
    "const_fold_opt__251": lambda h, w: [h // 4, w // 4],
    "const_fold_opt__252": lambda h, w: [h, w],
    "const_fold_opt__253": lambda h, w: [1, w // 8, h // 8, w // 8, h // 8],
    "const_fold_opt__255": lambda h, w: [h // 2, w // 2],
    "const_fold_opt__258": lambda h, w: [1, h // 8, w // 8, h // 8, w // 8],
    "const_fold_opt__263": lambda h, w: [h // 8, w // 8],
    "new_shape__239": lambda h, w: [1, 1, (h // 8) * (w // 8), (h // 8) * (w // 8)],
    "new_shape__243": lambda h, w: [1, 1, h, w],
    "new_shape__244": lambda h, w: [1, 1, 1, (h // 8) * (w // 8)],
}

PATCHED_FLOAT_INITIALIZERS = {
    "strided_slice_1:0": lambda h, w: np.ones((1, h, w, 1), dtype=np.float32),
}


def set_tensor_shape(value_info: onnx.ValueInfoProto, dims: list[int]) -> None:
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for value in dims:
        dim = shape.dim.add()
        dim.dim_value = int(value)


def replace_initializer(model: onnx.ModelProto, name: str, values: list[int]) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        original = numpy_helper.to_array(initializer)
        replacement = np.asarray(values, dtype=original.dtype).reshape(original.shape)
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(replacement, name))
        return
    raise KeyError(f"initializer not found: {name}")


def replace_initializer_array(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value, name))
        return
    raise KeyError(f"initializer not found: {name}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "ref/deepfillv2/deepfillv2.onnx")
    parser.add_argument("--output", type=Path, default=root / "Tools/DeepFillV2/output/deepfillv2.official_512x680.onnx")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=680)
    parser.add_argument("--check", action="store_true", help="Run ONNX checker; the stock official graph contains TensorFlow ExtractImagePatches and usually fails this.")
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0 or args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must be positive and divisible by 8")

    model = onnx.load(args.input)
    if len(model.graph.input) != 1:
        raise ValueError("expected one packed image+mask input")
    if len(model.graph.output) != 1:
        raise ValueError("expected one RGB output")

    set_tensor_shape(model.graph.input[0], [1, args.height, args.width * 2, 3])
    set_tensor_shape(model.graph.output[0], [1, args.height, args.width, 3])

    for name, builder in PATCHED_INT_INITIALIZERS.items():
        replace_initializer(model, name, builder(args.height, args.width))
    for name, builder in PATCHED_FLOAT_INITIALIZERS.items():
        replace_initializer_array(model, name, builder(args.height, args.width))

    del model.graph.value_info[:]
    if args.check:
        onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        {
            "status": "passed",
            "input": str(args.input),
            "output": str(args.output),
            "height": args.height,
            "width": args.width,
            "attention_grid": [args.height // 8, args.width // 8],
            "attention_cells": (args.height // 8) * (args.width // 8),
        }
    )


if __name__ == "__main__":
    main()
