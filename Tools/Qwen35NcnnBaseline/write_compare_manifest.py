"""Generate a stable per-layer comparison manifest for Unity validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inspect_model import Layer, parse_param


VIEW_LAYERS = {"ExpandDims", "Permute", "Reshape", "Split"}
STRICT_LAYERS = {"BinaryOp", "Concat", "GELU", "Sigmoid", "Slice", "Swish", "Tile"}
REDUCTION_LAYERS = {"GatedDeltaRule", "LayerNorm", "RMSNorm", "RotaryEmbed", "SDPA", "Softmax"}
MATMUL_LAYERS = {"Convolution3D", "Embed", "Gemm"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tolerance(layer_type: str) -> dict[str, Any]:
    if layer_type in {"Input", "MemoryData"}:
        return {"mode": "contract_only"}
    return {
        "mode": "value",
        "atol": 0.012,
        "rtol": 0.002,
        "max_mean_abs": 0.003,
        "min_cosine": 0.99999,
    }


def layer_record(network: str, index: int, layer: Layer) -> dict[str, Any]:
    return {
        "index": index,
        "type": layer.type,
        "name": layer.name,
        "bottoms": layer.bottoms,
        "tops": layer.tops,
        "checkpoints": [f"{network}/{index:04d}_{layer.name}/{top}" for top in layer.tops],
        "comparison": tolerance(layer.type),
    }


def build_manifest(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    params = config["params"]
    vision = config["setting"]["vision"]
    network_files = {
        "embed_token": params["embed_token_param"],
        "decoder": params["decoder_param"],
        "proj_out": params["proj_out_param"],
        "vision_embed_patch": vision["vision_embed_patch_param"],
        "vision_embed_pos": vision["vision_embed_pos_param"],
        "vision_encoder": vision["vision_encoder_param"],
    }
    networks: dict[str, Any] = {}
    total_layers = 0
    total_checkpoints = 0
    for network, filename in network_files.items():
        path = model_dir / filename
        header, layers = parse_param(path)
        records = [layer_record(network, index, layer) for index, layer in enumerate(layers)]
        checkpoint_count = sum(len(record["checkpoints"]) for record in records)
        total_layers += len(records)
        total_checkpoints += checkpoint_count
        networks[network] = {
            "param_file": filename,
            "param_sha256": sha256(path),
            "declared_layers": header["declared_layers"],
            "declared_blobs": header["declared_blobs"],
            "checkpoint_count": checkpoint_count,
            "layers": records,
        }
    return {
        "schema_version": 1,
        "model": "qwen3.5_0.8b",
        "dtype": "float32 gold; compare logical unpacked values",
        "axis_order": "ncnn logical order from .param; dumps must include dims/w/h/d/c/elempack metadata",
        "comparison_policy": "FP32 Unity GPU cumulative envelope with absolute/relative, mean-error, and cosine guards",
        "total_layers": total_layers,
        "total_checkpoints": total_checkpoints,
        "networks": networks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.model_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {manifest['total_layers']} layers and {manifest['total_checkpoints']} checkpoints "
        f"to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
