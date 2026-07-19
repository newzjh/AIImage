"""Compare Unity decoder-prefix texture checkpoints with ncnn/Python dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--unity-prefix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-index", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    layer_by_top = {}
    for layer in manifest["networks"]["decoder"]["layers"]:
        for top in layer["tops"]:
            layer_by_top[top] = layer

    rows = []
    reference_files = sorted(
        args.reference_dir.glob(f"reference.decode{args.decode_index}.blob_*.f32"),
        key=lambda path: (
            int(layer_by_top[path.stem.split("blob_")[-1]]["index"]),
            path.stem.split("blob_")[-1],
        ),
    )
    for reference_path in reference_files:
        blob = reference_path.stem.split("blob_")[-1]
        unity_path = Path(str(args.unity_prefix) + f".decode{args.decode_index}.blob_{blob}.f32")
        layer = layer_by_top[blob]
        comparison = layer["comparison"]
        atol = float(comparison.get("atol", 0.0))
        rtol = float(comparison.get("rtol", 0.0))
        reference = np.fromfile(reference_path, dtype=np.float32)
        unity = np.fromfile(unity_path, dtype=np.float32) if unity_path.is_file() else np.empty(0, np.float32)
        input_contract = layer["type"] == "Input" and not unity_path.is_file()
        count_match = reference.size == unity.size
        if input_contract:
            exceed = 0
            nonfinite = 0
            cosine = None
            max_abs = None
            mean_abs = None
        elif count_match and reference.size:
            error = np.abs(unity - reference)
            threshold = atol + rtol * np.abs(reference)
            nonfinite = int(np.count_nonzero(~np.isfinite(unity)))
            exceed = int(np.count_nonzero(error > threshold)) + nonfinite
            finite_values = error[np.isfinite(error)]
            max_abs = float(finite_values.max()) if finite_values.size else None
            mean_abs = float(finite_values.mean()) if finite_values.size else None
            if nonfinite == 0:
                denominator = float(np.linalg.norm(reference) * np.linalg.norm(unity))
                cosine = float(np.dot(reference, unity) / denominator) if denominator else None
                if cosine is not None and not np.isfinite(cosine):
                    cosine = None
            else:
                cosine = None
        else:
            exceed = max(int(reference.size), int(unity.size))
            nonfinite = int(np.count_nonzero(~np.isfinite(unity)))
            cosine = None
            max_abs = None
            mean_abs = None
        max_mean_abs = comparison.get("max_mean_abs")
        min_cosine = comparison.get("min_cosine")
        numerical_valid = count_match and exceed == 0
        if numerical_valid and max_mean_abs is not None and mean_abs is not None:
            numerical_valid = mean_abs <= float(max_mean_abs)
        if numerical_valid and min_cosine is not None and cosine is not None:
            numerical_valid = cosine >= float(min_cosine)
        rows.append(
            {
                "layer_index": int(layer["index"]),
                "layer_type": layer["type"],
                "layer_name": layer["name"],
                "blob": blob,
                "reference_count": int(reference.size),
                "unity_count": int(unity.size),
                "atol": atol,
                "rtol": rtol,
                "max_mean_abs": float(max_mean_abs) if max_mean_abs is not None else None,
                "min_cosine": float(min_cosine) if min_cosine is not None else None,
                "validation_mode": "input_contract" if input_contract else "numerical",
                "exceed_count": exceed,
                "nonfinite_count": nonfinite,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "cosine": cosine,
                "valid": (input_contract and bool(np.all(np.isfinite(reference)))) or numerical_valid,
            }
        )

    rows.sort(key=lambda row: row["layer_index"])
    failures = [row for row in rows if not row["valid"]]
    report = {
        "schema": "qwen35.decoder-prefix-checkpoint-compare/v1",
        "decode_index": args.decode_index,
        "checkpoint_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checkpoints": rows,
        "valid": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checkpoint_count", "pass_count", "failure_count", "first_failure", "valid")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
