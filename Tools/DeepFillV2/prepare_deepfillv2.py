#!/usr/bin/env python3
"""Normalize the supplied TensorFlow-exported DeepFillV2 ONNX for ORT and pnnx.

The reference model uses TensorFlow's non-standard ExtractImagePatches and
ReverseSequence nodes.  This tool lowers those fixed-shape nodes to standard
ONNX before either runtime is asked to load the graph.  It deliberately keeps
the original model read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def _attr(node: onnx.NodeProto, name: str) -> onnx.AttributeProto:
    for value in node.attribute:
        if value.name == name:
            return value
    raise ValueError(f"{node.name}: missing {name}")


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(model)
    result: dict[str, list[int]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value))
        result[value.name] = dims
    return result


def _constant(name: str, values: Iterable[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(list(values), dtype=np.int64), name)


def _patch_weights(name: str, channels: int, kernel: int) -> onnx.TensorProto:
    # TensorFlow ExtractImagePatches uses [ky, kx, channel] as its trailing
    # dimension.  A standard Conv with this sparse kernel emits that exact
    # channel ordering after NCHW -> NHWC transpose.
    weights = np.zeros((channels * kernel * kernel, channels, kernel, kernel), dtype=np.float32)
    out = 0
    for ky in range(kernel):
        for kx in range(kernel):
            for channel in range(channels):
                weights[out, channel, ky, kx] = 1.0
                out += 1
    return numpy_helper.from_array(weights, name)


def lower_extract_image_patches(model: onnx.ModelProto) -> int:
    shapes = _shape_map(model)
    initializer_values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    initializers = set(initializer_values)
    rewritten: list[onnx.NodeProto] = []
    count = 0
    packed_input_hw: tuple[int, int] | None = None
    if model.graph.input:
        input_shape = shapes.get(model.graph.input[0].name)
        if input_shape and len(input_shape) == 4 and input_shape[1] > 0 and input_shape[2] > 0:
            packed_input_hw = (input_shape[1], input_shape[2] // 2)

    for node in model.graph.node:
        # The TensorFlow export omits several intermediate shapes, but its
        # Reshape targets are fixed initializers.  Record them while walking
        # the topologically ordered graph so chained patch nodes stay static.
        if node.op_type == "Reshape" and len(node.input) > 1 and node.input[1] in initializer_values:
            target = [int(value) for value in initializer_values[node.input[1]].reshape(-1)]
            if target and all(value > 0 for value in target):
                shapes[node.output[0]] = target
        if node.op_type != "ExtractImagePatches":
            rewritten.append(node)
            continue

        input_shape = shapes.get(node.input[0])
        # The original TensorFlow export leaves Resize output dimensions
        # symbolic.  Their immediately following fixed Reshape constants make
        # the two attention branches unambiguous for this fixed 1920x1080
        # model.
        if not input_shape or not all(input_shape):
            if packed_input_hw and node.name == "inpaint_net/ExtractImagePatches":
                height, width = packed_input_hw
                input_shape = [1, height // 4, width // 4, 96]
            elif packed_input_hw and node.name == "inpaint_net/ExtractImagePatches_1":
                height, width = packed_input_hw
                input_shape = [1, height // 8, width // 8, 96]
            elif packed_input_hw and node.name == "inpaint_net/ExtractImagePatches_2":
                height, width = packed_input_hw
                input_shape = [1, height // 8, width // 8, 1]
            elif node.name.endswith("ExtractImagePatches_1"):
                input_shape = [1, 135, 240, 96]
            elif node.name.endswith("ExtractImagePatches_2"):
                input_shape = [1, 135, 240, 1]
        if not input_shape or len(input_shape) != 4 or not input_shape[-1]:
            raise ValueError(f"{node.name}: static NHWC input shape is required, got {input_shape}")
        if _attr(node, "padding").s != b"SAME":
            raise ValueError(f"{node.name}: only SAME padding is supported")
        if list(_attr(node, "rates").ints) != [1, 1, 1, 1]:
            raise ValueError(f"{node.name}: dilated patches are not supported")

        kernel = list(_attr(node, "ksizes").ints)
        strides = list(_attr(node, "strides").ints)
        if kernel[0] != 1 or kernel[3] != 1 or strides[0] != 1 or strides[3] != 1:
            raise ValueError(f"{node.name}: unsupported patch layout")
        kh, kw, sh, sw = kernel[1], kernel[2], strides[1], strides[2]
        if kh != kw or sh != sw:
            raise ValueError(f"{node.name}: only square patches/strides are supported")

        _, height, width, channels = input_shape
        out_height = (height + sh - 1) // sh
        out_width = (width + sw - 1) // sw
        total_h = max((out_height - 1) * sh + kh - height, 0)
        total_w = max((out_width - 1) * sw + kw - width, 0)
        pad_top, pad_bottom = total_h // 2, total_h - total_h // 2
        pad_left, pad_right = total_w // 2, total_w - total_w // 2
        prefix = f"deepfill_norm_{count}"
        pads_name = prefix + "_pads"
        weight_name = prefix + "_weights"
        if pads_name in initializers or weight_name in initializers:
            raise ValueError("normalization name collision")
        model.graph.initializer.extend([
            _constant(pads_name, [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]),
            _patch_weights(weight_name, channels, kh),
        ])
        initializers.update((pads_name, weight_name))

        nchw = prefix + "_nchw"
        padded = prefix + "_padded"
        conv = prefix + "_conv"
        rewritten.extend([
            helper.make_node("Transpose", [node.input[0]], [nchw], name=prefix + "_to_nchw", perm=[0, 3, 1, 2]),
            helper.make_node("Pad", [nchw, pads_name], [padded], name=prefix + "_same_pad", mode="constant"),
            helper.make_node("Conv", [padded, weight_name], [conv], name=prefix + "_patch_conv", strides=[sh, sw]),
            helper.make_node("Transpose", [conv], list(node.output), name=prefix + "_to_nhwc", perm=[0, 2, 3, 1]),
        ])
        shapes[node.output[0]] = [1, out_height, out_width, channels * kh * kw]
        count += 1

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def lower_reverse_sequence(model: onnx.ModelProto) -> int:
    rewritten: list[onnx.NodeProto] = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "ReverseSequence":
            rewritten.append(node)
            continue
        # The reference graph derives every sequence length from the RGB
        # channel count (3), so the operation is a full reversal on time axis.
        if _attr(node, "time_axis").i != 0 or _attr(node, "batch_axis").i != 1:
            raise ValueError(f"{node.name}: unexpected ReverseSequence axes")
        prefix = f"deepfill_reverse_{count}"
        starts = prefix + "_starts"
        ends = prefix + "_ends"
        axes = prefix + "_axes"
        steps = prefix + "_steps"
        model.graph.initializer.extend([
            _constant(starts, [-1]), _constant(ends, [-9223372036854775807]),
            _constant(axes, [0]), _constant(steps, [-1]),
        ])
        rewritten.append(helper.make_node("Slice", [node.input[0], starts, ends, axes, steps], list(node.output), name=prefix + "_slice"))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "ref/deepfillv2/deepfillv2.onnx")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output/deepfillv2.standard.onnx")
    args = parser.parse_args()
    model = onnx.load(args.input)
    patches = lower_extract_image_patches(model)
    reverses = lower_reverse_sequence(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"normalized={args.output} extract_image_patches={patches} reverse_sequence={reverses}")


if __name__ == "__main__":
    main()
