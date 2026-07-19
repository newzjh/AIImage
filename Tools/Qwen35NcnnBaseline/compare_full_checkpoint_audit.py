"""Compare Unity and ncnn dumps for the complete Qwen3.5 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_values(reference: np.ndarray, unity: np.ndarray, policy: dict) -> tuple[bool, dict]:
    atol = float(policy.get("atol", 0.0))
    rtol = float(policy.get("rtol", 0.0))
    if reference.size != unity.size or not reference.size:
        return False, {"reference_count": int(reference.size), "unity_count": int(unity.size)}
    error = np.abs(unity - reference)
    allowed = np.float32(atol) + np.float32(rtol) * np.abs(reference)
    finite = bool(np.isfinite(reference).all() and np.isfinite(unity).all())
    exceed = int(np.count_nonzero(error > allowed))
    max_abs = float(np.max(error))
    mean_abs = float(np.mean(error, dtype=np.float64))
    denominator = float(np.linalg.norm(reference.astype(np.float64)) * np.linalg.norm(unity.astype(np.float64)))
    cosine = float(np.dot(reference.astype(np.float64), unity.astype(np.float64)) / denominator) if denominator else None
    valid = finite and exceed == 0
    if valid and policy.get("max_mean_abs") is not None:
        valid = mean_abs <= float(policy["max_mean_abs"])
    if valid and policy.get("min_cosine") is not None and cosine is not None:
        valid = cosine >= float(policy["min_cosine"])
    return valid, {
        "reference_count": int(reference.size),
        "unity_count": int(unity.size),
        "atol": atol,
        "rtol": rtol,
        "max_mean_abs": policy.get("max_mean_abs"),
        "min_cosine": policy.get("min_cosine"),
        "exceed_count": exceed,
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(unity))),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "valid": valid,
    }


def main() -> int:
    tool_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=tool_dir / "reports" / "qwen35_0_8b_compare_manifest.json")
    parser.add_argument("--audit-root", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit")
    parser.add_argument("--output", type=Path, default=tool_dir / "reports" / "full_checkpoint_audit" / "full_compare.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for network, spec in manifest["networks"].items():
        for layer in spec["layers"]:
            for blob in layer["tops"]:
                contract_only = layer["type"] in ("Input", "MemoryData")
                if network == "decoder":
                    reference_path = args.audit_root / "decoder" / f"reference.decode1.blob_{blob}.f32"
                    unity_path = args.audit_root / "decoder" / f"unity.decode1.blob_{blob}.f32"
                else:
                    reference_path = args.audit_root / network / f"reference.blob_{blob}.f32"
                    unity_path = args.audit_root / network / f"unity.blob_{blob}.f32"
                row = {
                    "network": network,
                    "layer_index": int(layer["index"]),
                    "layer_type": layer["type"],
                    "layer_name": layer["name"],
                    "blob": blob,
                    "validation_mode": "contract_only" if contract_only else "numerical",
                    "reference_path": str(reference_path.resolve()),
                    "unity_path": str(unity_path.resolve()),
                }
                if contract_only:
                    row.update({"reference_count": None, "unity_count": int(np.fromfile(unity_path, dtype="<f4").size) if unity_path.is_file() else 0, "valid": True})
                else:
                    if not reference_path.is_file() or not unity_path.is_file():
                        row.update({"reference_count": int(np.fromfile(reference_path, dtype="<f4").size) if reference_path.is_file() else 0, "unity_count": int(np.fromfile(unity_path, dtype="<f4").size) if unity_path.is_file() else 0, "valid": False})
                    else:
                        reference = np.fromfile(reference_path, dtype="<f4")
                        unity = np.fromfile(unity_path, dtype="<f4")
                        valid, metrics = compare_values(reference, unity, layer["comparison"])
                        row.update(metrics)
                        row["valid"] = valid
                rows.append(row)
    rows.sort(key=lambda row: (row["network"], row["layer_index"], row["blob"]))
    failures = [row for row in rows if not row["valid"]]
    report = {
        "schema": "qwen35.full-checkpoint-compare/v1",
        "manifest": str(args.manifest.resolve()),
        "checkpoint_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checkpoints": rows,
        "valid": not failures and len(rows) == int(manifest["total_checkpoints"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checkpoint_count", "pass_count", "failure_count", "first_failure", "valid")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
