"""Dump ncnn CPU references for all non-contract Qwen3.5 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ncnn
import numpy as np

from vision_reference import vision_rope_2d
from vision_patch_atlas_compare import build_all_patches


NETWORKS = ("embed_token", "proj_out", "vision_embed_patch", "vision_embed_pos", "vision_encoder")


def load_net(model_dir: Path, name: str) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_packing_layout = False
    if net.load_param(str(model_dir / f"qwen3.5_{name}.ncnn.param")) != 0:
        raise RuntimeError(f"failed to load {name} param")
    bin_name = "qwen3.5_embed_token.ncnn.bin" if name == "proj_out" else f"qwen3.5_{name}.ncnn.bin"
    if net.load_model(str(model_dir / bin_name)) != 0:
        raise RuntimeError(f"failed to load {name} weights")
    return net


def numeric_blobs(manifest: dict, network: str) -> list[str]:
    result: list[str] = []
    for layer in manifest["networks"][network]["layers"]:
        if layer["type"] in ("Input", "MemoryData"):
            continue
        result.extend(str(top) for top in layer["tops"])
    return result


def extract_and_dump(
    net: ncnn.Net,
    inputs: list[tuple[str, np.ndarray]],
    blobs: list[str],
    output_dir: Path,
    prefix: str = "reference",
) -> dict[str, dict[str, object]]:
    extractor = net.create_extractor()
    if hasattr(extractor, "set_light_mode"):
        extractor.set_light_mode(False)
    for name, values in inputs:
        status = extractor.input(name, ncnn.Mat(np.ascontiguousarray(values)))
        if status != 0:
            raise RuntimeError(f"failed to feed {name}: {status}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for blob in blobs:
        status, value = extractor.extract(blob)
        if status != 0:
            raise RuntimeError(f"failed to extract {blob}: {status}")
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        flat = array.reshape(-1)
        path = output_dir / f"{prefix}.blob_{blob}.f32"
        flat.astype("<f4", copy=False).tofile(path)
        records[blob] = {
            "path": str(path.resolve()),
            "count": int(flat.size),
            "shape": list(array.shape),
            "finite": bool(np.isfinite(flat).all()),
        }
    return records


def main() -> int:
    tool_dir = Path(__file__).resolve().parent
    root = tool_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=tool_dir / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--image", type=Path, default=root / "ref" / "ncnn_llm-main" / "test.jpg")
    parser.add_argument("--manifest", type=Path, default=tool_dir / "reports" / "qwen35_0_8b_compare_manifest.json")
    parser.add_argument("--output-root", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit")
    parser.add_argument("--report", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit" / "reference_full_audit_report.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report: dict[str, object] = {"schema": "qwen35.ncnn.full-checkpoint-reference/v1", "networks": {}, "valid": True}

    # Build the same 64x48 patch atlas used by the Unity preprocessor.
    patches, _, target_shape = build_all_patches(args.image)
    target_width, target_height = target_shape
    grid_width, grid_height = target_width // 16, target_height // 16
    atlas = np.ascontiguousarray(
        patches.reshape(grid_height, grid_width, 3, 2, 16, 16)
        .transpose(2, 3, 0, 4, 1, 5)
        .reshape(3, 2, target_height, target_width)
    )

    patch_net = load_net(args.model_dir, "vision_embed_patch")
    patch_records = extract_and_dump(
        patch_net,
        [("in0", atlas)],
        numeric_blobs(manifest, "vision_embed_patch"),
        args.output_root / "vision_embed_patch",
    )
    patch_net = None
    raw_patch_reference = np.fromfile(
        args.output_root / "vision_embed_patch" / "reference.blob_out0.f32", dtype="<f4"
    ).reshape(768, grid_height, grid_width)
    patch_reference = np.ascontiguousarray(
        raw_patch_reference.transpose(1, 2, 0).reshape(grid_width * grid_height, 768)
    )

    position_net = load_net(args.model_dir, "vision_embed_pos")
    position_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    position_records = extract_and_dump(
        position_net,
        [("in0", position_grid)],
        numeric_blobs(manifest, "vision_embed_pos"),
        args.output_root / "vision_embed_pos",
    )
    # The final position output is reordered only when it becomes an encoder input.
    position_extractor = position_net.create_extractor()
    position_extractor.input("in0", ncnn.Mat(position_grid))
    status, position_value = position_extractor.extract("out0")
    if status != 0:
        raise RuntimeError(f"failed to extract vision position out0: {status}")
    raw_position = np.asarray(position_value, dtype=np.float32).reshape(grid_height * grid_width, 768)
    position_for_encoder = np.ascontiguousarray(
        raw_position.reshape(grid_height, grid_width, 768)
        .reshape(grid_height // 2, 2, grid_width // 2, 2, 768)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_height * grid_width, 768)
    )
    position_net = None

    cos, sin = vision_rope_2d(grid_height, grid_width, merge=2, theta=10_000.0, section=(16, 16))
    # The Unity upload and RotaryEmbed contract consume the 32-value half dimension;
    # vision_rope_2d also returns the duplicated half used by other reference helpers.
    cos = np.ascontiguousarray(cos[:, :32])
    sin = np.ascontiguousarray(sin[:, :32])
    encoder_net = load_net(args.model_dir, "vision_encoder")
    encoder_records = extract_and_dump(
        encoder_net,
        [("in0", patch_reference), ("in1", position_for_encoder), ("in2", cos), ("in3", sin)],
        numeric_blobs(manifest, "vision_encoder"),
        args.output_root / "vision_encoder",
    )
    encoder_net = None

    embed_net = load_net(args.model_dir, "embed_token")
    embed_records = extract_and_dump(
        embed_net,
        [("in0", np.asarray([0], dtype=np.int32))],
        numeric_blobs(manifest, "embed_token"),
        args.output_root / "embed_token",
    )
    embed_extractor = embed_net.create_extractor()
    embed_extractor.input("in0", ncnn.Mat(np.asarray([0], dtype=np.int32)))
    status, embed_value = embed_extractor.extract("out0")
    if status != 0:
        raise RuntimeError(f"failed to extract embed out0: {status}")
    hidden = np.ascontiguousarray(np.asarray(embed_value, dtype=np.float32)).copy()
    embed_net = None

    projection_net = load_net(args.model_dir, "proj_out")
    projection_records = extract_and_dump(
        projection_net,
        [("in0", hidden)],
        numeric_blobs(manifest, "proj_out"),
        args.output_root / "proj_out",
    )
    projection_net = None

    records_by_network = {
        "vision_embed_patch": patch_records,
        "vision_embed_pos": position_records,
        "vision_encoder": encoder_records,
        "embed_token": embed_records,
        "proj_out": projection_records,
    }
    for network, records in records_by_network.items():
        report["networks"][network] = {"checkpoint_count": len(records), "checkpoints": records}
        report["valid"] = bool(report["valid"]) and all(item["finite"] for item in records.values())
    report["numeric_checkpoint_count"] = sum(item["checkpoint_count"] for item in report["networks"].values())
    report["valid"] = bool(report["valid"]) and report["numeric_checkpoint_count"] == 372
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "numeric_checkpoint_count": report["numeric_checkpoint_count"]}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
