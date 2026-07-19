#!/usr/bin/env python3
"""Inject states_tf_celebahq.pth weights into the official DeepFillV2 ONNX.

The checkpoint stores a dict with G/D OrderedDicts and the G keys match the
TensorFlow-style official ONNX conv layer names, for example:
  conv1.conv.weight -> inpaint_net/conv1/Conv2D/ReadVariableOp:0
  conv13_upsample.conv.conv.weight -> .../conv13_upsample_conv/Conv2D/...
Attention fuse Conv2D_1/2 kernels are fixed graph constants and are not
present in the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def pth_keys_for_layer(layer: str) -> tuple[str, str]:
    if layer.endswith("_upsample"):
        return f"{layer}.conv.conv.weight", f"{layer}.conv.conv.bias"
    return f"{layer}.conv.weight", f"{layer}.conv.bias"


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))
        return
    raise KeyError(name)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "Tools/DeepFillV2/output/deepfillv2.official_512x680.onnx")
    parser.add_argument("--checkpoint", type=Path, default=root / "ref/deepfillv2/states_tf_celebahq.pth")
    parser.add_argument("--output", type=Path, default=root / "Tools/DeepFillV2/output/deepfillv2.official_512x680.states_tf_celebahq.onnx")
    args = parser.parse_args()

    model = onnx.load(args.input)
    init_names = {initializer.name for initializer in model.graph.initializer}
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["G"] if isinstance(checkpoint, dict) and "G" in checkpoint else checkpoint

    patched = []
    skipped = []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in init_names:
            continue
        if "Conv2D_1" in node.name or "Conv2D_2" in node.name:
            skipped.append(node.name)
            continue
        parts = node.name.split("/")
        if len(parts) < 3 or parts[0] != "inpaint_net":
            skipped.append(node.name)
            continue
        layer = parts[1]
        weight_key, bias_key = pth_keys_for_layer(layer)
        if weight_key not in state or bias_key not in state:
            skipped.append(node.name)
            continue
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy()
        replace_initializer(model, node.input[1], weight)
        if len(node.input) > 2 and node.input[2] in init_names:
            replace_initializer(model, node.input[2], bias)
        patched.append((node.name, weight_key))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        {
            "status": "passed",
            "input": str(args.input),
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "patched_conv_layers": len(patched),
            "skipped_conv_layers": skipped,
        }
    )


if __name__ == "__main__":
    main()
