#!/usr/bin/env python3
"""Build a small DeepFillV2 HiFill NCNN package with native ExtractPatches.

The generic pnnx route can lower TensorFlow ExtractImagePatches into giant
one-hot Conv kernels.  For HiFill this inflates the final .bin to hundreds of
MB.  This tool keeps the pnnx/ncnn optimized topology but replaces the known
one-hot patch Conv layers with a repo-owned lightweight NCNN custom layer:

    ExtractPatches

The output .bin contains only learned weights.  ExtractPatches carries no
weight payload and is implemented by the Unity runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
from pathlib import Path
from typing import BinaryIO


PATCH_LAYERS = {
    # source ONNX node -> ncnn layer name:
    # ExtractImagePatches_2 / _3
    "padconv_3",
    "padconv_4",
    # ExtractImagePatches / _1
    "conv_76",
    "conv_77",
    # ExtractImagePatches_4 / _5
    "conv_88",
    "padconv_5",
    # ExtractImagePatches_6 / _7
    "conv_98",
    "padconv_6",
}

INTERP_COORDINATE_TRANSFORM_PARAM = 100
INTERP_COORDINATE_TRANSFORM_ASYMMETRIC = "1"

TAG_FP16 = 0x01306B47
TAG_INT8 = 0x000D4B38
TAG_FLOAT32_EXTRA_SCALE = 0x0002C056


@dataclasses.dataclass
class Layer:
    raw: str
    typ: str
    name: str
    bottoms: list[str]
    tops: list[str]
    params: dict[int, str]
    extra_tokens: list[str]


def parse_param(path: Path) -> tuple[str, int, list[Layer]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError(f"param too short: {path}")
    magic = lines[0]
    header = lines[1].split()
    blob_count = int(header[1])
    layers: list[Layer] = []
    for line in lines[2:]:
        tok = line.split()
        if len(tok) < 4:
            continue
        typ, name = tok[0], tok[1]
        bottoms = int(tok[2])
        tops = int(tok[3])
        idx = 4
        bottom_names = tok[idx : idx + bottoms]
        idx += bottoms
        top_names = tok[idx : idx + tops]
        idx += tops
        params: dict[int, str] = {}
        extra_tokens: list[str] = []
        for item in tok[idx:]:
            if "=" not in item:
                extra_tokens.append(item)
                continue
            key, value = item.split("=", 1)
            try:
                params[int(key)] = value
            except ValueError:
                extra_tokens.append(item)
        layers.append(Layer(line, typ, name, bottom_names, top_names, params, extra_tokens))
    return magic, blob_count, layers


def align4(n: int) -> int:
    return (n + 3) & ~3


def copy_exact(src: BinaryIO, dst: BinaryIO | None, count: int) -> bytes:
    data = src.read(count)
    if len(data) != count:
        raise EOFError(f"wanted {count} bytes, got {len(data)}")
    if dst is not None:
        dst.write(data)
    return data


def copy_or_skip_ncnn_array(src: BinaryIO, dst: BinaryIO | None, count: int) -> int:
    if count <= 0:
        return 0
    start = src.tell()
    flag_bytes = copy_exact(src, dst, 4)
    (flag,) = struct.unpack("<I", flag_bytes)
    f0 = flag & 0xFF
    f1 = (flag >> 8) & 0xFF
    f2 = (flag >> 16) & 0xFF
    f3 = (flag >> 24) & 0xFF
    flag_sum = f0 + f1 + f2 + f3
    if flag == TAG_FP16:
        copy_exact(src, dst, align4(count * 2))
    elif flag == TAG_INT8:
        copy_exact(src, dst, align4(count))
    elif flag == TAG_FLOAT32_EXTRA_SCALE:
        copy_exact(src, dst, count * 4)
    elif flag_sum != 0:
        copy_exact(src, dst, 256 * 4)
        copy_exact(src, dst, align4(count))
    elif f0 == 0:
        copy_exact(src, dst, count * 4)
    else:
        raise ValueError(f"unsupported ncnn weight flag 0x{flag:08X} at {start}")
    return src.tell() - start


def copy_or_skip_raw_f32(src: BinaryIO, dst: BinaryIO | None, count: int) -> int:
    if count <= 0:
        return 0
    start = src.tell()
    copy_exact(src, dst, count * 4)
    return src.tell() - start


def layer_weight_byte_count(layer: Layer, src: BinaryIO, dst: BinaryIO | None) -> int:
    start = src.tell()
    typ = layer.typ
    p = layer.params
    if typ in {"Convolution", "ConvolutionDepthWise", "Deconvolution", "DeconvolutionDepthWise", "Convolution1D"}:
        copy_or_skip_ncnn_array(src, dst, int(p.get(6, "0")))
        if int(p.get(5, "0")) != 0:
            copy_or_skip_raw_f32(src, dst, int(p.get(0, "0")))
    elif typ == "InnerProduct":
        copy_or_skip_ncnn_array(src, dst, int(p.get(2, "0")))
        if int(p.get(1, "0")) != 0:
            copy_or_skip_raw_f32(src, dst, int(p.get(0, "0")))
    elif typ == "Gemm":
        # NcnnRepro reads constant B and C as NCNN arrays when present.
        if int(p.get(5, "0")) != 0:
            copy_or_skip_ncnn_array(src, dst, int(p.get(9, "0")) * int(p.get(10, "0")))
        if int(p.get(6, "0")) != 0:
            copy_or_skip_ncnn_array(src, dst, int(p.get(11, "0")) * int(p.get(12, "0")))
    elif typ == "MemoryData":
        w = int(p.get(0, "0"))
        h = int(p.get(1, "0"))
        d = int(p.get(11, "0"))
        c = int(p.get(2, "0"))
        count = w * h * d * c if d else (w * h * c if c else (w * h if h else (w if w else 1)))
        copy_or_skip_ncnn_array(src, dst, count)
    elif typ == "BatchNorm":
        channels = int(p.get(0, "0"))
        for _ in range(4):
            copy_or_skip_ncnn_array(src, dst, channels)
    elif typ in {"InstanceNorm", "GroupNorm"}:
        channels = int(p.get(0 if typ == "InstanceNorm" else 1, "0"))
        affine = int(p.get(3, "1")) if typ == "GroupNorm" else 1
        if affine:
            copy_or_skip_ncnn_array(src, dst, channels)
            copy_or_skip_ncnn_array(src, dst, channels)
    elif typ == "LayerNorm":
        affine_size = int(p.get(0, "0"))
        affine = int(p.get(2, "1"))
        if affine and affine_size > 0:
            copy_or_skip_ncnn_array(src, dst, affine_size)
            copy_or_skip_ncnn_array(src, dst, affine_size)
    elif typ == "Embed":
        copy_or_skip_ncnn_array(src, dst, int(p.get(3, "0")))
        if int(p.get(2, "0")) != 0:
            copy_or_skip_ncnn_array(src, dst, int(p.get(0, "0")))
    elif typ in {"PReLU", "Scale", "Normalize", "Quantize", "Dequantize", "Requantize", "RMSNorm"}:
        # Not used by the current HiFill graph, but covered for safer reuse.
        size = int(p.get(0, p.get(1, "0")))
        if size > 0:
            copy_or_skip_ncnn_array(src, dst, size)
            if typ in {"Scale", "Requantize"} and int(p.get(1 if typ == "Scale" else 4, "0")) != 0:
                copy_or_skip_ncnn_array(src, dst, size)
    elif typ == "MultiHeadAttention":
        embed_dim = int(p.get(0, "0"))
        weight_data_size = int(p.get(2, "0"))
        kdim = int(p.get(3, str(embed_dim)))
        vdim = int(p.get(4, str(embed_dim)))
        qdim = weight_data_size // max(1, embed_dim) if embed_dim else 0
        copy_or_skip_ncnn_array(src, dst, embed_dim * qdim)
        copy_or_skip_ncnn_array(src, dst, embed_dim)
        copy_or_skip_ncnn_array(src, dst, embed_dim * kdim)
        copy_or_skip_ncnn_array(src, dst, embed_dim)
        copy_or_skip_ncnn_array(src, dst, embed_dim * vdim)
        copy_or_skip_ncnn_array(src, dst, embed_dim)
        copy_or_skip_ncnn_array(src, dst, qdim * embed_dim)
        copy_or_skip_ncnn_array(src, dst, qdim)
    return src.tell() - start


def rewrite_layer(layer: Layer) -> str:
    if layer.typ == "Interp":
        if INTERP_COORDINATE_TRANSFORM_PARAM in layer.params:
            return layer.raw
        return layer.raw + f" {INTERP_COORDINATE_TRANSFORM_PARAM}={INTERP_COORDINATE_TRANSFORM_ASYMMETRIC}"
    if layer.name not in PATCH_LAYERS:
        return layer.raw
    p = layer.params
    rewritten = {
        1: p.get(1, "0"),
        11: p.get(11, p.get(1, "0")),
        2: p.get(2, "1"),
        12: p.get(12, p.get(2, "1")),
        3: p.get(3, "1"),
        13: p.get(13, p.get(3, "1")),
        4: p.get(4, "0"),
        14: p.get(14, p.get(4, "0")),
        15: p.get(15, p.get(4, "0")),
        16: p.get(16, p.get(14, p.get(4, "0"))),
        18: "0.0",
    }
    tokens = [
        "ExtractPatches",
        layer.name,
        str(len(layer.bottoms)),
        str(len(layer.tops)),
        *layer.bottoms,
        *layer.tops,
    ]
    for key in [1, 11, 2, 12, 3, 13, 4, 14, 15, 16, 18]:
        tokens.append(f"{key}={rewritten[key]}")
    return " ".join(tokens)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_compact_weight_layers(layers: list[Layer]) -> list[Layer]:
    result: list[Layer] = []
    for layer in layers:
        if layer.name in PATCH_LAYERS:
            continue
        if layer.typ in {"Convolution", "ConvolutionDepthWise"}:
            result.append(layer)
    return result


def float32_tensor_bytes(tensor) -> bytes:
    import numpy as np

    if tensor.data_type != 1:
        raise ValueError(f"expected float32 tensor: {tensor.name} dtype={tensor.data_type}")
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    expected = count * 4
    if tensor.raw_data and len(tensor.raw_data) == expected:
        return bytes(tensor.raw_data)
    from onnx import numpy_helper

    arr = numpy_helper.to_array(tensor).astype("<f4", copy=False)
    data = arr.reshape(-1).tobytes(order="C")
    if len(data) != expected:
        raise ValueError(f"tensor byte size mismatch: {tensor.name} expected={expected} got={len(data)}")
    return data


def write_onnx_conv_weight(dst: BinaryIO, layer: Layer, node, initializers: dict[str, object]) -> int:
    start = dst.tell()
    if len(node.input) < 2:
        raise ValueError(f"ONNX Conv node has no weight input: {node.name}")
    weight = initializers.get(node.input[1])
    if weight is None:
        raise ValueError(f"ONNX Conv weight initializer not found: {node.name} -> {node.input[1]}")
    weight_data = float32_tensor_bytes(weight)
    expected_weight_bytes = int(layer.params.get(6, "0")) * 4
    if len(weight_data) != expected_weight_bytes:
        raise ValueError(
            f"weight size mismatch: layer={layer.name} node={node.name} expected={expected_weight_bytes} got={len(weight_data)}"
        )
    dst.write(struct.pack("<I", 0))
    dst.write(weight_data)

    if int(layer.params.get(5, "0")) != 0:
        if len(node.input) < 3:
            raise ValueError(f"NCNN layer expects bias but ONNX Conv has none: layer={layer.name} node={node.name}")
        bias = initializers.get(node.input[2])
        if bias is None:
            raise ValueError(f"ONNX Conv bias initializer not found: {node.name} -> {node.input[2]}")
        bias_data = float32_tensor_bytes(bias)
        expected_bias_bytes = int(layer.params.get(0, "0")) * 4
        if len(bias_data) != expected_bias_bytes:
            raise ValueError(
                f"bias size mismatch: layer={layer.name} node={node.name} expected={expected_bias_bytes} got={len(bias_data)}"
            )
        dst.write(bias_data)
    return dst.tell() - start


def build_from_source_onnx(args: argparse.Namespace) -> None:
    import onnx

    magic, blob_count, layers = parse_param(args.input_param)
    model = onnx.load(args.source_onnx)
    opset = next((opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")), 0)
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    patch_nodes = [node for node in model.graph.node if node.op_type == "ExtractImagePatches"]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    compact_weight_layers = iter_compact_weight_layers(layers)

    if opset != 13:
        raise ValueError(f"expected ONNX opset 13, got {opset}")
    if len(conv_nodes) != len(compact_weight_layers):
        raise ValueError(f"ONNX Conv count {len(conv_nodes)} != compact NCNN conv layer count {len(compact_weight_layers)}")
    if len(patch_nodes) != 8:
        raise ValueError(f"expected 8 ExtractImagePatches nodes, got {len(patch_nodes)}")

    args.output_param.parent.mkdir(parents=True, exist_ok=True)
    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    copied: dict[str, int] = {}
    with args.output_bin.open("wb") as dst:
        for layer, node in zip(compact_weight_layers, conv_nodes):
            copied[layer.name] = write_onnx_conv_weight(dst, layer, node, initializers)

    rewritten_lines = [magic, f"{len(layers)} {blob_count}"]
    rewritten_lines.extend(rewrite_layer(layer) for layer in layers)
    args.output_param.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    report = {
        "status": "passed",
        "mode": "source_onnx_direct",
        "source_onnx": str(args.source_onnx.resolve()),
        "input_param": str(args.input_param.resolve()),
        "output_param": str(args.output_param.resolve()),
        "output_bin": str(args.output_bin.resolve()),
        "source_onnx_bytes": args.source_onnx.stat().st_size,
        "output_bin_bytes": args.output_bin.stat().st_size,
        "output_param_bytes": args.output_param.stat().st_size,
        "opset": opset,
        "conv_nodes": len(conv_nodes),
        "extract_image_patches_nodes": len(patch_nodes),
        "interp_coordinate_transform": "asymmetric",
        "interp_coordinate_transform_param": INTERP_COORDINATE_TRANSFORM_PARAM,
        "patch_layers": sorted(PATCH_LAYERS),
        "copied_total_bytes": sum(copied.values()),
        "output_bin_sha256": sha256(args.output_bin),
        "output_param_sha256": sha256(args.output_param),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


def build_from_standard_bin(args: argparse.Namespace) -> None:
    magic, blob_count, layers = parse_param(args.input_param)
    args.output_param.parent.mkdir(parents=True, exist_ok=True)
    args.output_bin.parent.mkdir(parents=True, exist_ok=True)

    skipped: dict[str, int] = {}
    copied: dict[str, int] = {}
    with args.input_bin.open("rb") as src, args.output_bin.open("wb") as dst:
        for layer in layers:
            if layer.name in PATCH_LAYERS:
                skipped[layer.name] = layer_weight_byte_count(layer, src, None)
            else:
                copied[layer.name] = layer_weight_byte_count(layer, src, dst)
        remaining = src.read(1)
        if remaining:
            raise ValueError(f"input bin has trailing data at byte {src.tell() - 1}")

    rewritten_lines = [magic, f"{len(layers)} {blob_count}"]
    rewritten_lines.extend(rewrite_layer(layer) for layer in layers)
    args.output_param.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    report = {
        "status": "passed",
        "mode": "standard_bin_skip_sparse_extract_patches",
        "input_param": str(args.input_param.resolve()),
        "input_bin": str(args.input_bin.resolve()),
        "output_param": str(args.output_param.resolve()),
        "output_bin": str(args.output_bin.resolve()),
        "input_bin_bytes": args.input_bin.stat().st_size,
        "output_bin_bytes": args.output_bin.stat().st_size,
        "output_param_bytes": args.output_param.stat().st_size,
        "patch_layers": sorted(PATCH_LAYERS),
        "interp_coordinate_transform": "asymmetric",
        "interp_coordinate_transform_param": INTERP_COORDINATE_TRANSFORM_PARAM,
        "skipped_weight_bytes": skipped,
        "skipped_total_bytes": sum(skipped.values()),
        "copied_total_bytes": sum(copied.values()),
        "output_bin_sha256": sha256(args.output_bin),
        "output_param_sha256": sha256(args.output_param),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "source-onnx", "standard-bin"], default="auto")
    parser.add_argument("--source-onnx", type=Path, default=root / "Assets/StreamingAssets/DeepFileV2/deepfillv2_hifill.source.onnx")
    parser.add_argument("--input-param", type=Path, default=root / "Tools/DeepFillV2/output/hifill.standard.ncnn.param")
    parser.add_argument("--input-bin", type=Path, default=root / "Tools/DeepFillV2/output/hifill.standard.ncnn.bin")
    parser.add_argument("--output-param", type=Path, default=root / "Assets/StreamingAssets/DeepFileV2/deepfillv2_hifill.ncnn.param")
    parser.add_argument("--output-bin", type=Path, default=root / "Assets/StreamingAssets/DeepFileV2/deepfillv2_hifill.ncnn.bin")
    parser.add_argument("--report", type=Path, default=root / "Tools/DeepFillV2/output/deepfillv2_small_ncnn_report.json")
    args = parser.parse_args()
    if args.mode == "source-onnx" or (args.mode == "auto" and args.source_onnx.exists()):
        build_from_source_onnx(args)
    else:
        build_from_standard_bin(args)


if __name__ == "__main__":
    main()
