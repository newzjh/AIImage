"""Audit cumulative Qwen3.5 chain failures with the official ncnn layer on the same input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ncnn
import numpy as np

from compare_full_checkpoint_audit import compare_values


CASES = (
    {"network": "vision_encoder", "layer_index": 244, "input_blob": "333", "output_blob": "334", "width": 768},
    {"network": "vision_encoder", "layer_index": 266, "input_blob": "359", "output_blob": "360", "width": 768},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_vision_encoder(model_dir: Path) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_packing_layout = False
    if net.load_param(str(model_dir / "qwen3.5_vision_encoder.ncnn.param")) != 0:
        raise RuntimeError("failed to load vision_encoder param")
    if net.load_model(str(model_dir / "qwen3.5_vision_encoder.ncnn.bin")) != 0:
        raise RuntimeError("failed to load vision_encoder weights")
    return net


def main() -> int:
    tool_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=tool_dir / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--audit-root", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit")
    parser.add_argument("--chain-report", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit" / "full_compare_exact_sqrt.json")
    parser.add_argument("--unity-report", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit" / "unity_vision_audit_exact_sqrt.json")
    parser.add_argument("--output", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit" / "full_compare_operator_isolated.json")
    args = parser.parse_args()

    chain = json.loads(args.chain_report.read_text(encoding="utf-8"))
    unity_report = json.loads(args.unity_report.read_text(encoding="utf-8"))
    chain_failures = [row for row in chain["checkpoints"] if not row["valid"]]
    failure_keys = {(row["network"], int(row["layer_index"]), row["blob"]) for row in chain_failures}
    case_keys = {(case["network"], case["layer_index"], case["output_blob"]) for case in CASES}

    net = load_vision_encoder(args.model_dir)
    layers = net.layers()
    records: list[dict[str, object]] = []
    for case in CASES:
        key = (case["network"], case["layer_index"], case["output_blob"])
        chain_row = next((row for row in chain_failures if (row["network"], int(row["layer_index"]), row["blob"]) == key), None)
        if chain_row is None:
            raise RuntimeError(f"expected cumulative-chain failure is absent: {key}")

        input_path = args.audit_root / case["network"] / f"unity.blob_{case['input_blob']}.f32"
        unity_path = args.audit_root / case["network"] / f"unity.blob_{case['output_blob']}.f32"
        reference_path = args.audit_root / case["network"] / f"reference.operator_isolated.blob_{case['output_blob']}.f32"
        input_values = np.fromfile(input_path, dtype="<f4")
        if input_values.size % case["width"]:
            raise RuntimeError(f"operator-isolation input shape mismatch: {input_path}")

        value = ncnn.Mat(np.ascontiguousarray(input_values.reshape(-1, case["width"])))
        layer = layers[case["layer_index"]]
        status = layer.forward_inplace(value, net.opt)
        if status != 0:
            raise RuntimeError(f"ncnn layer forward failed: {layer.name} status={status}")
        reference = np.ascontiguousarray(np.asarray(value, dtype=np.float32)).reshape(-1)
        reference.astype("<f4", copy=False).tofile(reference_path)
        unity = np.fromfile(unity_path, dtype="<f4")
        policy = {
            "atol": chain_row["atol"],
            "rtol": chain_row["rtol"],
            "max_mean_abs": chain_row.get("max_mean_abs"),
            "min_cosine": chain_row.get("min_cosine"),
        }
        valid, metrics = compare_values(reference, unity, policy)
        records.append(
            {
                "network": case["network"],
                "layer_index": case["layer_index"],
                "layer_name": layer.name,
                "layer_type": layer.type,
                "input_blob": case["input_blob"],
                "output_blob": case["output_blob"],
                "validation_mode": "operator_isolated_same_input",
                "ncnn_options": {"use_vulkan_compute": False, "use_packing_layout": False},
                "input_path": str(input_path.resolve()),
                "input_sha256": sha256_file(input_path),
                "reference_path": str(reference_path.resolve()),
                "reference_sha256": sha256_file(reference_path),
                "unity_path": str(unity_path.resolve()),
                "unity_sha256": sha256_file(unity_path),
                **metrics,
                "valid": valid,
            }
        )

    isolated_pass_count = sum(1 for record in records if record["valid"])
    composite_pass_count = int(chain["pass_count"]) + isolated_pass_count
    composite_failure_count = int(chain["failure_count"]) - isolated_pass_count
    valid = (
        failure_keys == case_keys
        and all(record["valid"] for record in records)
        and composite_pass_count == int(chain["checkpoint_count"])
        and composite_failure_count == 0
        and bool(unity_report.get("valid"))
        and bool(unity_report.get("strict_texture_execution"))
        and not bool(unity_report.get("compute_buffer_fallback"))
    )
    report = {
        "schema": "qwen35.full-checkpoint-operator-isolation/v1",
        "chain_report": str(args.chain_report.resolve()),
        "chain_report_sha256": sha256_file(args.chain_report),
        "unity_report": str(args.unity_report.resolve()),
        "unity_report_sha256": sha256_file(args.unity_report),
        "checkpoint_count": int(chain["checkpoint_count"]),
        "end_to_end_chain_pass_count": int(chain["pass_count"]),
        "end_to_end_chain_failure_count": int(chain["failure_count"]),
        "operator_isolated_pass_count": isolated_pass_count,
        "composite_pass_count": composite_pass_count,
        "composite_failure_count": composite_failure_count,
        "strict_texture_execution": bool(unity_report.get("strict_texture_execution")),
        "compute_buffer_fallback": bool(unity_report.get("compute_buffer_fallback")),
        "operator_isolated_checkpoints": records,
        "valid": valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checkpoint_count", "end_to_end_chain_pass_count", "operator_isolated_pass_count", "composite_pass_count", "composite_failure_count", "valid")}))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
