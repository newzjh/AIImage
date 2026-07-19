#!/usr/bin/env python
"""Export the checked-in DeepFillV2 case1 model for the Unity texture runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ref" / "deepfillv2" / "states_tf_places2.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Assets" / "StreamingAssets" / "DeepFileV2"


@dataclass(frozen=True)
class ConvSpec:
    name: str
    out_channels: int
    kernel: int = 3
    stride: int = 1
    dilation: int = 1
    activation: str = "elu"
    upsample: bool = False
    gated: bool = True


STAGE1 = [
    ConvSpec("conv1", 24, 5),
    ConvSpec("conv2_downsample", 48, stride=2),
    ConvSpec("conv3", 48),
    ConvSpec("conv4_downsample", 96, stride=2),
    ConvSpec("conv5", 96),
    ConvSpec("conv6", 96),
    ConvSpec("conv7_atrous", 96, dilation=2),
    ConvSpec("conv8_atrous", 96, dilation=4),
    ConvSpec("conv9_atrous", 96, dilation=8),
    ConvSpec("conv10_atrous", 96, dilation=16),
    ConvSpec("conv11", 96),
    ConvSpec("conv12", 96),
    ConvSpec("conv13_upsample", 48, upsample=True),
    ConvSpec("conv14", 48),
    ConvSpec("conv15_upsample", 24, upsample=True),
    ConvSpec("conv16", 12),
    ConvSpec("conv17", 3, activation="none", gated=False),
]

HALLUCINATION = [
    ConvSpec("xconv1", 24, 5),
    ConvSpec("xconv2_downsample", 24, stride=2),
    ConvSpec("xconv3", 48),
    ConvSpec("xconv4_downsample", 48, stride=2),
    ConvSpec("xconv5", 96),
    ConvSpec("xconv6", 96),
    ConvSpec("xconv7_atrous", 96, dilation=2),
    ConvSpec("xconv8_atrous", 96, dilation=4),
    ConvSpec("xconv9_atrous", 96, dilation=8),
    ConvSpec("xconv10_atrous", 96, dilation=16),
]

ATTENTION = [
    ConvSpec("pmconv1", 24, 5),
    ConvSpec("pmconv2_downsample", 24, stride=2),
    ConvSpec("pmconv3", 48),
    ConvSpec("pmconv4_downsample", 96, stride=2),
    ConvSpec("pmconv5", 96),
    ConvSpec("pmconv6", 96, activation="relu"),
]

ATTENTION_TAIL = [ConvSpec("pmconv9", 96), ConvSpec("pmconv10", 96)]

STAGE2_TAIL = [
    ConvSpec("allconv11", 96),
    ConvSpec("allconv12", 96),
    ConvSpec("allconv13_upsample", 48, upsample=True),
    ConvSpec("allconv14", 48),
    ConvSpec("allconv15_upsample", 24, upsample=True),
    ConvSpec("allconv16", 12),
    ConvSpec("allconv17", 3, activation="none", gated=False),
]


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExportGraph:
    def __init__(self, state: dict[str, torch.Tensor], width: int, height: int) -> None:
        self.state = state
        self.width = width
        self.height = height
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.layers: list[str] = []
        self.blobs: set[str] = {"image", "mask"}
        self.shapes: dict[str, tuple[int, int, int]] = {
            "image": (3, height, width),
            "mask": (1, height, width),
        }
        self.conv_order: list[str] = []
        self.bin_parts: list[bytes] = []
        self._constant_index = 0

    def add_layer(self, layer_type: str, name: str, bottoms: list[str], tops: list[str], params: list[str] | None = None) -> None:
        tokens = [f"{layer_type:<24}", f"{name:<25}", str(len(bottoms)), str(len(tops)), *bottoms, *tops]
        if params:
            tokens.extend(params)
        self.layers.append(" ".join(tokens))
        self.blobs.update(tops)

    def add_initializer(self, name: str, array: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(array)
        self.initializers.append(numpy_helper.from_array(contiguous, name=name))
        return name

    def add_constant(self, prefix: str, array: np.ndarray) -> str:
        name = f"{prefix}_{self._constant_index}"
        self._constant_index += 1
        return self.add_initializer(name, array)

    @staticmethod
    def same_padding(in_size: int, kernel: int, stride: int, dilation: int) -> tuple[int, int, int]:
        out_size = (in_size + stride - 1) // stride
        effective = (kernel - 1) * dilation + 1
        total = max((out_size - 1) * stride + effective - in_size, 0)
        before = total // 2
        return before, total - before, out_size

    def resize2x(self, input_blob: str, name: str) -> str:
        channels, in_h, in_w = self.shapes[input_blob]
        output = name + "_out"
        scales = self.add_constant(name + "_scales", np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
        roi = self.add_constant(name + "_roi", np.asarray([], dtype=np.float32))
        self.nodes.append(
            helper.make_node(
                "Resize",
                [input_blob, roi, scales],
                [output],
                name=name,
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        )
        self.add_layer(
            "Interp",
            name,
            [input_blob],
            [output],
            ["0=1", "1=2.0", "2=2.0", f"3={in_h * 2}", f"4={in_w * 2}", "6=0", "100=1"],
        )
        self.shapes[output] = (channels, in_h * 2, in_w * 2)
        return output

    def gated_conv(self, input_blob: str, spec: ConvSpec) -> str:
        if spec.upsample:
            input_blob = self.resize2x(input_blob, spec.name + "_resize")

        in_channels, in_h, in_w = self.shapes[input_blob]
        pad_top, pad_bottom, out_h = self.same_padding(in_h, spec.kernel, spec.stride, spec.dilation)
        pad_left, pad_right, out_w = self.same_padding(in_w, spec.kernel, spec.stride, spec.dilation)
        key_base = spec.name + (".conv.conv" if spec.upsample else ".conv")
        weight = self.state[key_base + ".weight"].detach().cpu().numpy().astype(np.float32, copy=False)
        bias = self.state[key_base + ".bias"].detach().cpu().numpy().astype(np.float32, copy=False)
        expected_outputs = spec.out_channels * (2 if spec.gated else 1)
        if tuple(weight.shape) != (expected_outputs, in_channels, spec.kernel, spec.kernel):
            raise ValueError(f"Unexpected weight shape for {spec.name}: {weight.shape}")
        if tuple(bias.shape) != (expected_outputs,):
            raise ValueError(f"Unexpected bias shape for {spec.name}: {bias.shape}")

        def emit_conv(suffix: str, part_weight: np.ndarray, part_bias: np.ndarray, fused_activation: int = 0) -> str:
            layer_name = f"{spec.name}_{suffix}"
            output = layer_name + "_out"
            weight_name = self.add_initializer(layer_name + ".weight", part_weight)
            bias_name = self.add_initializer(layer_name + ".bias", part_bias)
            self.nodes.append(
                helper.make_node(
                    "Conv",
                    [input_blob, weight_name, bias_name],
                    [output],
                    name=layer_name,
                    kernel_shape=[spec.kernel, spec.kernel],
                    strides=[spec.stride, spec.stride],
                    dilations=[spec.dilation, spec.dilation],
                    pads=[pad_top, pad_left, pad_bottom, pad_right],
                )
            )
            params = [
                f"0={part_weight.shape[0]}",
                f"1={spec.kernel}",
                f"11={spec.kernel}",
                f"2={spec.dilation}",
                f"12={spec.dilation}",
                f"3={spec.stride}",
                f"13={spec.stride}",
                f"4={pad_left}",
                f"14={pad_top}",
                f"15={pad_right}",
                f"16={pad_bottom}",
                "5=1",
                f"6={part_weight.size}",
            ]
            if fused_activation:
                params.append(f"9={fused_activation}")
            self.add_layer("Convolution", layer_name, [input_blob], [output], params)
            self.shapes[output] = (part_weight.shape[0], out_h, out_w)
            self.conv_order.append(layer_name)
            self.bin_parts.extend(
                [struct.pack("<I", 0), np.ascontiguousarray(part_weight).tobytes(), np.ascontiguousarray(part_bias).tobytes()]
            )
            return output

        feature = emit_conv("feature", weight[: spec.out_channels], bias[: spec.out_channels])
        if not spec.gated:
            return feature

        gate = emit_conv("gate", weight[spec.out_channels :], bias[spec.out_channels :])
        gate_sigmoid = gate + "_sigmoid"
        self.nodes.append(helper.make_node("Sigmoid", [gate], [gate_sigmoid], name=spec.name + "_sigmoid"))
        self.add_layer("Sigmoid", spec.name + "_sigmoid", [gate], [gate_sigmoid])
        self.shapes[gate_sigmoid] = self.shapes[gate]

        activated = feature + "_activated"
        if spec.activation == "relu":
            self.nodes.append(helper.make_node("Relu", [feature], [activated], name=spec.name + "_relu"))
            self.add_layer("ReLU", spec.name + "_relu", [feature], [activated], ["0=0.0"])
        else:
            self.nodes.append(helper.make_node("Elu", [feature], [activated], name=spec.name + "_elu", alpha=1.0))
            self.add_layer("ELU", spec.name + "_elu", [feature], [activated], ["0=1.0"])
        self.shapes[activated] = self.shapes[feature]

        output = spec.name + "_out"
        self.nodes.append(helper.make_node("Mul", [activated, gate_sigmoid], [output], name=spec.name + "_mul"))
        self.add_layer("BinaryOp", spec.name + "_mul", [activated, gate_sigmoid], [output], ["0=2"])
        self.shapes[output] = (spec.out_channels, out_h, out_w)
        return output

    def run_stack(self, input_blob: str, specs: list[ConvSpec]) -> str:
        value = input_blob
        for spec in specs:
            value = self.gated_conv(value, spec)
        return value

    def scalar_binary(self, input_blob: str, name: str, op_type: int, scalar: float, onnx_op: str, scalar_first: bool = False) -> str:
        output = name + "_out"
        constant = self.add_constant(name + "_scalar", np.asarray(scalar, dtype=np.float32))
        inputs = [constant, input_blob] if scalar_first else [input_blob, constant]
        self.nodes.append(helper.make_node(onnx_op, inputs, [output], name=name))
        self.add_layer("BinaryOp", name, [input_blob], [output], [f"0={op_type}", "1=1", f"2={scalar:.9g}"])
        self.shapes[output] = self.shapes[input_blob]
        return output

    def tanh(self, input_blob: str, name: str) -> str:
        output = name + "_out"
        self.nodes.append(helper.make_node("Tanh", [input_blob], [output], name=name))
        self.add_layer("TanH", name, [input_blob], [output])
        self.shapes[output] = self.shapes[input_blob]
        return output

    def build(self) -> str:
        self.add_layer("Input", "image", [], ["image"])
        self.add_layer("Input", "mask", [], ["mask"])

        image2 = self.scalar_binary("image", "prepare_image_mul2", 2, 2.0, "Mul")
        image_norm = self.scalar_binary(image2, "prepare_image_sub1", 1, 1.0, "Sub")
        one_minus_mask = self.scalar_binary("mask", "prepare_one_minus_mask", 7, 1.0, "Sub", scalar_first=True)
        self.nodes.append(helper.make_node("Mul", [image_norm, one_minus_mask], ["image_masked"], name="prepare_image_masked"))
        self.add_layer("BinaryOp", "prepare_image_masked", [image_norm, one_minus_mask], ["image_masked"], ["0=2"])
        self.shapes["image_masked"] = (3, self.height, self.width)
        mask_zero = self.scalar_binary("mask", "prepare_mask_zero", 2, 0.0, "Mul")
        ones = self.scalar_binary(mask_zero, "prepare_ones", 0, 1.0, "Add")
        self.nodes.append(helper.make_node("Concat", ["image_masked", ones, "mask"], ["x"], name="prepare_concat", axis=1))
        self.add_layer("Concat", "prepare_concat", ["image_masked", ones, "mask"], ["x"], ["0=0"])
        self.shapes["x"] = (5, self.height, self.width)

        stage1 = self.tanh(self.run_stack("x", STAGE1), "stage1_tanh")

        starts = self.add_constant("stage2_slice_starts", np.asarray([0], dtype=np.int64))
        ends = self.add_constant("stage2_slice_ends", np.asarray([3], dtype=np.int64))
        axes = self.add_constant("stage2_slice_axes", np.asarray([1], dtype=np.int64))
        steps = self.add_constant("stage2_slice_steps", np.asarray([1], dtype=np.int64))
        self.nodes.append(helper.make_node("Slice", ["x", starts, ends, axes, steps], ["x_rgb"], name="stage2_x_rgb"))
        self.add_layer("Slice", "stage2_x_rgb", ["x"], ["x_rgb", "x_aux"], ["-23300=2,3,2", "1=0"])
        self.shapes["x_rgb"] = (3, self.height, self.width)
        self.shapes["x_aux"] = (2, self.height, self.width)

        self.nodes.append(helper.make_node("Mul", [stage1, "mask"], ["stage1_masked"], name="stage1_masked"))
        self.add_layer("BinaryOp", "stage1_masked", [stage1, "mask"], ["stage1_masked"], ["0=2"])
        self.shapes["stage1_masked"] = (3, self.height, self.width)
        self.nodes.append(helper.make_node("Add", ["stage1_masked", "x_rgb"], ["stage2_input"], name="stage2_input"))
        self.add_layer("BinaryOp", "stage2_input", ["stage1_masked", "x_rgb"], ["stage2_input"], ["0=0"])
        self.shapes["stage2_input"] = (3, self.height, self.width)

        hallu = self.run_stack("stage2_input", HALLUCINATION)
        attention_features = self.run_stack("stage2_input", ATTENTION)
        attended = "contextual_attention_out"
        self.nodes.append(
            helper.make_node(
                "DeepFillV2ContextualAttention",
                [attention_features, "mask"],
                [attended],
                name="contextual_attention_case1_2021",
                domain="com.aiimage",
                ksize=3,
                rate=2,
                stride=1,
                softmax_scale=10.0,
                patch_epsilon=1.0e-4,
                mask_downsample=8,
            )
        )
        self.add_layer(
            "DeepFillV2ContextualAttention",
            "contextual_attention_case1_2021",
            [attention_features, "mask"],
            [attended],
            ["0=3", "1=2", "2=1", "3=10.0", "4=0.0001", "5=8"],
        )
        self.shapes[attended] = self.shapes[attention_features]
        attention_tail = self.run_stack(attended, ATTENTION_TAIL)

        combined = "stage2_combined"
        self.nodes.append(helper.make_node("Concat", [hallu, attention_tail], [combined], name="stage2_concat", axis=1))
        self.add_layer("Concat", "stage2_concat", [hallu, attention_tail], [combined], ["0=0"])
        hallu_shape = self.shapes[hallu]
        self.shapes[combined] = (hallu_shape[0] + self.shapes[attention_tail][0], hallu_shape[1], hallu_shape[2])

        stage2 = self.tanh(self.run_stack(combined, STAGE2_TAIL), "stage2_tanh")
        self.nodes.append(helper.make_node("Identity", [stage2], ["out0"], name="output_identity"))
        self.add_layer("Noop", "output_identity", [stage2], ["out0"])
        self.shapes["out0"] = self.shapes[stage2]
        return "out0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the DeepFillV2 case1 model for Unity.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8:
        raise ValueError("DeepFillV2 dimensions must be positive multiples of 8")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _torch_load(args.checkpoint.resolve())
    state = checkpoint["G"]
    graph = ExportGraph(state, args.width, args.height)
    graph.build()

    onnx_path = args.output_dir / "deepfillv2_case1.source.onnx"
    param_path = args.output_dir / "deepfillv2_case1.ncnn.param"
    bin_path = args.output_dir / "deepfillv2_case1.ncnn.bin"
    report_path = args.output_dir / "deepfillv2_case1.export.json"

    onnx_graph = helper.make_graph(
        graph.nodes,
        "AIImage.DeepFillV2.Case1.2021",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, args.height, args.width]),
            helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, args.height, args.width]),
        ],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, 3, args.height, args.width])],
        initializer=graph.initializers,
    )
    model = helper.make_model(
        onnx_graph,
        producer_name="AIImage.DeepFillV2Exporter",
        producer_version="1",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.aiimage", 1)],
    )
    model.ir_version = 9
    model.metadata_props.add(key="aiimage.model", value="deepfillv2-pytorch-case1")
    model.metadata_props.add(key="aiimage.attention_semantics", value="case1-2021")
    onnx.save_model(model, onnx_path)

    param_text = "\n".join(["7767517", f"{len(graph.layers)} {len(graph.blobs)}", *graph.layers, ""])
    param_path.write_text(param_text, encoding="utf-8", newline="\n")
    with bin_path.open("wb") as stream:
        for part in graph.bin_parts:
            stream.write(part)

    report = {
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "width": args.width,
        "height": args.height,
        "attention_semantics": "case1-2021",
        "conv_count": len(graph.conv_order),
        "layer_count": len(graph.layers),
        "blob_count": len(graph.blobs),
        "onnx": {"path": str(onnx_path), "bytes": onnx_path.stat().st_size, "sha256": _sha256(onnx_path)},
        "param": {"path": str(param_path), "bytes": param_path.stat().st_size, "sha256": _sha256(param_path)},
        "bin": {"path": str(bin_path), "bytes": bin_path.stat().st_size, "sha256": _sha256(bin_path)},
        "conv_order": graph.conv_order,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
