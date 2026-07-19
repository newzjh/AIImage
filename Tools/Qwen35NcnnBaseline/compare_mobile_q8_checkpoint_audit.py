"""Compare mobile Q8 Unity checkpoints with the FP32 ncnn/Python reference set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_full_checkpoint_audit import compare_values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q8_compare(reference: np.ndarray, unity: np.ndarray) -> tuple[bool, dict[str, object]]:
    if reference.size != unity.size or not reference.size:
        return False, {"reference_count": int(reference.size), "unity_count": int(unity.size)}
    finite = bool(np.isfinite(reference).all() and np.isfinite(unity).all())
    error = np.abs(unity - reference)
    reference64 = reference.astype(np.float64)
    unity64 = unity.astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    reference_max_abs = float(np.max(np.abs(reference)))
    max_abs = float(np.max(error))
    mean_abs = float(np.mean(error, dtype=np.float64))
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(unity64))
    cosine = float(np.dot(reference64, unity64) / denominator) if denominator else None

    # Match the repository's established W8 acceptance shape: bounded mean error
    # plus cosine preservation. Max error remains telemetry because isolated INT8
    # outliers do not represent vector-level drift and existing W8 tests do not gate it.
    mean_limit = max(0.03, 0.04 * reference_rms)
    cosine_valid = cosine is None or reference_rms < 1e-7 or cosine >= 0.98
    valid = finite and mean_abs <= mean_limit and cosine_valid
    return valid, {
        "reference_count": int(reference.size),
        "unity_count": int(unity.size),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(unity))),
        "reference_rms": reference_rms,
        "reference_max_abs": reference_max_abs,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "cosine": cosine,
        "q8_mean_abs_limit": mean_limit,
        "q8_min_cosine": 0.98,
        "valid": valid,
    }


def main() -> int:
    tool_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=tool_dir / "reports" / "qwen35_0_8b_compare_manifest.json")
    parser.add_argument("--reference-root", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit")
    parser.add_argument("--unity-root", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit_mobile_q8")
    parser.add_argument("--unity-report", type=Path, required=True)
    parser.add_argument("--ocr-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for network, spec in manifest["networks"].items():
        for layer in spec["layers"]:
            for blob in layer["tops"]:
                contract_only = layer["type"] in ("Input", "MemoryData")
                reference_name = f"reference.decode1.blob_{blob}.f32" if network == "decoder" else f"reference.blob_{blob}.f32"
                unity_name = f"unity.decode1.blob_{blob}.f32" if network == "decoder" else f"unity.blob_{blob}.f32"
                reference_path = args.reference_root / network / reference_name
                unity_path = args.unity_root / network / unity_name
                row: dict[str, object] = {
                    "network": network,
                    "layer_index": int(layer["index"]),
                    "layer_type": layer["type"],
                    "layer_name": layer["name"],
                    "blob": blob,
                    "validation_mode": "contract_only" if contract_only else "mobile_q8_numerical",
                    "reference_path": str(reference_path.resolve()),
                    "unity_path": str(unity_path.resolve()),
                }
                if contract_only:
                    row.update({"strict_fp32_valid": True, "valid": True})
                elif not reference_path.is_file() or not unity_path.is_file():
                    row.update({
                        "reference_count": int(np.fromfile(reference_path, dtype="<f4").size) if reference_path.is_file() else 0,
                        "unity_count": int(np.fromfile(unity_path, dtype="<f4").size) if unity_path.is_file() else 0,
                        "strict_fp32_valid": False,
                        "valid": False,
                    })
                else:
                    reference = np.fromfile(reference_path, dtype="<f4")
                    unity = np.fromfile(unity_path, dtype="<f4")
                    strict_valid, strict_metrics = compare_values(reference, unity, layer["comparison"])
                    valid, metrics = q8_compare(reference, unity)
                    row.update(metrics)
                    row["strict_fp32_valid"] = strict_valid
                    row["strict_fp32_max_abs"] = strict_metrics.get("max_abs")
                    row["strict_fp32_mean_abs"] = strict_metrics.get("mean_abs")
                    row["valid"] = valid
                rows.append(row)

    rows.sort(key=lambda row: (row["network"], row["layer_index"], row["blob"]))
    failures = [row for row in rows if not row["valid"]]
    strict_failures = [row for row in rows if not row["strict_fp32_valid"]]
    unity_report = json.loads(args.unity_report.read_text(encoding="utf-8"))
    ocr_report = json.loads(args.ocr_report.read_text(encoding="utf-8"))
    expected = int(manifest["total_checkpoints"])
    report = {
        "schema": "qwen35.mobile-q8-checkpoint-compare/v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "reference_root": str(args.reference_root.resolve()),
        "unity_root": str(args.unity_root.resolve()),
        "checkpoint_count": len(rows),
        "q8_pass_count": len(rows) - len(failures),
        "q8_failure_count": len(failures),
        "first_q8_failure": failures[0] if failures else None,
        "strict_fp32_pass_count": len(rows) - len(strict_failures),
        "strict_fp32_failure_count": len(strict_failures),
        "first_strict_fp32_deviation": strict_failures[0] if strict_failures else None,
        "tolerance_policy": {
            "mean_abs": "<= max(0.03, 0.04 * reference_rms)",
            "max_abs": "telemetry_only",
            "min_cosine": 0.98,
            "finite_required": True,
        },
        "strict_texture_execution": bool(unity_report.get("strict_texture_execution")),
        "compute_buffer_fallback": bool(unity_report.get("compute_buffer_fallback")),
        "ocr_valid": bool(ocr_report.get("valid")),
        "ocr_marker_hit_count": int(ocr_report.get("marker_hit_count", 0)),
        "ocr_marker_group_count": int(ocr_report.get("marker_group_count", 0)),
        "checkpoints": rows,
    }
    report["valid"] = (
        not failures
        and len(rows) == expected
        and bool(unity_report.get("valid"))
        and report["strict_texture_execution"]
        and not report["compute_buffer_fallback"]
        and report["ocr_valid"]
        and report["ocr_marker_hit_count"] == report["ocr_marker_group_count"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "checkpoint_count", "q8_pass_count", "q8_failure_count", "first_q8_failure",
        "strict_fp32_pass_count", "strict_fp32_failure_count", "first_strict_fp32_deviation", "valid")}, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
