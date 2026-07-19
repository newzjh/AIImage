"""Inspect ncnn param files and freeze the Qwen3.5 blob/cache contract."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAGIC = "7767517"
EXPECTED_INPUTS = ["in0", "in1", "in2", "in3"]


@dataclass
class Layer:
    type: str
    name: str
    bottoms: list[str]
    tops: list[str]
    params: list[str]


def parse_param(path: Path) -> tuple[dict[str, int], list[Layer]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2 or lines[0] != MAGIC:
        raise ValueError(f"{path} is not an ncnn text param")
    header_fields = lines[1].split()
    if len(header_fields) != 2:
        raise ValueError(f"invalid ncnn header in {path}")
    header = {"declared_layers": int(header_fields[0]), "declared_blobs": int(header_fields[1])}
    layers: list[Layer] = []
    for line_number, line in enumerate(lines[2:], start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid layer at {path}:{line_number}")
        bottom_count = int(fields[2])
        top_count = int(fields[3])
        names_end = 4 + bottom_count + top_count
        if len(fields) < names_end:
            raise ValueError(f"truncated layer at {path}:{line_number}")
        layers.append(
            Layer(
                type=fields[0],
                name=fields[1],
                bottoms=fields[4 : 4 + bottom_count],
                tops=fields[4 + bottom_count : names_end],
                params=fields[names_end:],
            )
        )
    return header, layers


def network_report(path: Path) -> dict[str, Any]:
    header, layers = parse_param(path)
    produced = {blob for layer in layers for blob in layer.tops}
    consumed = {blob for layer in layers for blob in layer.bottoms}
    input_blobs = [blob for layer in layers if layer.type == "Input" for blob in layer.tops]
    output_blobs = sorted(produced - consumed)
    type_counts = collections.Counter(layer.type for layer in layers)
    return {
        "file": path.name,
        **header,
        "parsed_layers": len(layers),
        "unique_blobs": len(produced | consumed),
        "input_blobs": input_blobs,
        "terminal_output_blobs": output_blobs,
        "layer_type_counts": dict(sorted(type_counts.items())),
    }


def decoder_contract(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    header, layers = parse_param(path)
    all_blobs = {blob for layer in layers for blob in layer.bottoms + layer.tops}
    inputs = {blob for layer in layers if layer.type == "Input" for blob in layer.tops}
    setting = config["setting"]
    attn = int(setting["attn_cnt"])
    conv = int(setting["sconv_cnt"])
    gdr = int(setting["gdr_cnt"])
    cache_inputs = (
        [name for index in range(attn) for name in (f"cache_k{index}", f"cache_v{index}")]
        + [f"cache_conv{index}" for index in range(conv)]
        + [f"cache_gdr{index}" for index in range(gdr)]
    )
    cache_outputs = (
        [name for index in range(attn) for name in (f"out_cache_k{index}", f"out_cache_v{index}")]
        + [f"out_cache_conv{index}" for index in range(conv)]
        + [f"out_cache_gdr{index}" for index in range(gdr)]
    )
    expected_inputs = EXPECTED_INPUTS + cache_inputs
    topology = [layer.type for layer in layers if layer.type in {"ShortConv", "GatedDeltaRule", "SDPA"}]
    return {
        **header,
        "expected_input_count": len(expected_inputs),
        "cache_input_count": len(cache_inputs),
        "cache_output_count": len(cache_outputs),
        "missing_input_layers": sorted(set(expected_inputs) - inputs),
        "missing_cache_outputs": sorted(set(cache_outputs) - all_blobs),
        "custom_layer_counts": {
            name: sum(layer.type == name for layer in layers) for name in ("ShortConv", "GatedDeltaRule", "SDPA")
        },
        "attention_topology": topology,
        "valid": (
            len(layers) == header["declared_layers"]
            and not (set(expected_inputs) - inputs)
            and not (set(cache_outputs) - all_blobs)
            and sum(layer.type == "ShortConv" for layer in layers) == conv
            and sum(layer.type == "GatedDeltaRule" for layer in layers) == gdr
        ),
    }


def build_report(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "model.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    param_paths = sorted(model_dir.glob("*.ncnn.param"))
    if not param_paths:
        raise FileNotFoundError(f"no .ncnn.param files found under {model_dir}")
    decoder_name = config["params"]["decoder_param"]
    report = {
        "source_model": "qwen3.5_0.8b",
        "model_type": config.get("type"),
        "setting": config["setting"],
        "weight_aliases": {
            "proj_out_bin": config["params"]["proj_out_bin"],
            "embed_token_bin": config["params"]["embed_token_bin"],
            "shared": config["params"]["proj_out_bin"] == config["params"]["embed_token_bin"],
        },
        "networks": [network_report(path) for path in param_paths],
        "decoder_contract": decoder_contract(model_dir / decoder_name, config),
    }
    report["valid"] = report["model_type"] == "qwen3.5" and report["decoder_contract"]["valid"]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, help="optional stable JSON report path")
    parser.add_argument("--strict", action="store_true", help="return non-zero when the contract is incomplete")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.model_dir.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    else:
        print(rendered, end="")
    return 2 if args.strict and not report["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

