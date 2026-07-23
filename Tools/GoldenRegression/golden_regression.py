#!/usr/bin/env python3
"""Compare AIImage inference golden observations without importing Unity runtime code.

The tool intentionally consumes plain JSON observations. Unity Editor Debug/Oracle
tests export those observations after texture-aware readback; production runners do
not import or depend on this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "aiimage.inference.golden/v1"
TENSOR_SCHEMA = "aiimage.inference.tensor/v1"
REPORT_SCHEMA = "aiimage.inference.golden-report/v1"
DEFAULT_THRESHOLDS = {
    "FP32": {"abs": 1.0e-5, "rel": 1.0e-5},
    "FP16": {"abs": 2.0e-3, "rel": 2.0e-2},
    "INT8": {"abs": 8.0e-2, "rel": 1.5e-1},
}


class GoldenError(ValueError):
    pass


@dataclass(frozen=True)
class TensorKey:
    case_id: str
    node: str
    blob: str

    @property
    def display(self) -> str:
        return f"{self.case_id}:{self.node}/{self.blob}"


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as exc:
        raise GoldenError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GoldenError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def shape(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(not isinstance(item, int) or item <= 0 for item in value):
        raise GoldenError(f"{label} must be a non-empty list of positive integers")
    return value


def validate_tensor(tensor: dict[str, Any], source: str) -> None:
    if tensor.get("schema_version") != TENSOR_SCHEMA:
        raise GoldenError(f"{source}: tensor schema_version must be {TENSOR_SCHEMA}")
    if not isinstance(tensor.get("node"), str) or not tensor["node"]:
        raise GoldenError(f"{source}: node is required")
    if not isinstance(tensor.get("blob"), str) or not tensor["blob"]:
        raise GoldenError(f"{source}: blob is required")
    logical = shape(tensor.get("logical_shape"), f"{source}: logical_shape")
    shape(tensor.get("storage_shape"), f"{source}: storage_shape")
    values = tensor.get("values")
    if not isinstance(values, list) or any(not isinstance(item, (int, float)) for item in values):
        raise GoldenError(f"{source}: values must be a numeric list")
    expected_count = math.prod(logical[1:]) if len(logical) == 5 else math.prod(logical)
    if len(values) != expected_count:
        raise GoldenError(f"{source}: value count {len(values)} does not match logical shape {logical} ({expected_count})")
    if not isinstance(tensor.get("layout"), str) or not tensor["layout"]:
        raise GoldenError(f"{source}: layout is required")
    if not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]:
        raise GoldenError(f"{source}: dtype is required")


def read_tensor(path: Path) -> dict[str, Any]:
    tensor = load_json(path)
    validate_tensor(tensor, str(path))
    return tensor


def tensor_key(case_id: str, tensor: dict[str, Any]) -> TensorKey:
    return TensorKey(case_id, tensor["node"], tensor["blob"])


def observation_index(root: Path | None) -> dict[TensorKey, dict[str, Any]]:
    indexed: dict[TensorKey, dict[str, Any]] = {}
    if root is None or not root.exists():
        return indexed
    for path in sorted(root.rglob("*.json")):
        try:
            payload = load_json(path)
        except GoldenError:
            continue
        if payload.get("schema_version") == TENSOR_SCHEMA:
            validate_tensor(payload, str(path))
            case_id = payload.get("case_id")
            if isinstance(case_id, str) and case_id:
                indexed[tensor_key(case_id, payload)] = payload
        elif payload.get("schema_version") == "aiimage.inference.golden-observations/v1":
            case_id = payload.get("case_id")
            for tensor in payload.get("tensors", []):
                if isinstance(case_id, str) and isinstance(tensor, dict):
                    validate_tensor(tensor, str(path))
                    indexed[tensor_key(case_id, tensor)] = tensor
    return indexed


def thresholds(manifest: dict[str, Any]) -> dict[str, float]:
    precision = str(manifest.get("precision", "FP32")).upper()
    result = dict(DEFAULT_THRESHOLDS.get(precision, DEFAULT_THRESHOLDS["FP32"]))
    declared = manifest.get("thresholds", {})
    if isinstance(declared, dict):
        precision_values = declared.get(precision) or declared.get(precision.lower()) or declared
        if isinstance(precision_values, dict):
            for key in ("abs", "rel"):
                value = precision_values.get(key)
                if isinstance(value, (int, float)) and value >= 0:
                    result[key] = float(value)
    return result


def compare_values(expected: list[float], actual: list[float], limits: dict[str, float]) -> dict[str, Any]:
    count = min(len(expected), len(actual))
    max_abs = -1.0
    max_rel = -1.0
    sum_abs = 0.0
    sum_rel = 0.0
    worst_index = -1
    failures = 0
    non_finite = 0
    for index in range(count):
        reference = float(expected[index])
        observed = float(actual[index])
        if not math.isfinite(reference) or not math.isfinite(observed):
            non_finite += 1
            failures += 1
            if worst_index < 0:
                worst_index = index
            continue
        absolute = abs(observed - reference)
        relative = absolute / max(abs(reference), 1.0e-12)
        sum_abs += absolute
        sum_rel += relative
        if absolute > max_abs or (absolute == max_abs and relative > max_rel):
            max_abs = absolute
            max_rel = relative
            worst_index = index
        if absolute > limits["abs"] and relative > limits["rel"]:
            failures += 1
    if len(expected) != len(actual):
        failures += abs(len(expected) - len(actual))
    return {
        "pass": failures == 0,
        "value_count": count,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "failed_values": failures,
        "non_finite_values": non_finite,
        "max_abs_error": max(0.0, max_abs),
        "mean_abs_error": sum_abs / count if count else 0.0,
        "max_rel_error": max(0.0, max_rel),
        "mean_rel_error": sum_rel / count if count else 0.0,
        "worst_index": worst_index,
    }


def matches_injection(key: TensorKey, injection: str | None) -> tuple[bool, int, float]:
    if not injection:
        return False, -1, 0.0
    parts = injection.rsplit(":", 2)
    if len(parts) != 3:
        raise GoldenError("--inject-perturbation must be case_id:node/blob:index:delta")
    target, index_text, delta_text = parts
    if target != key.display:
        return False, -1, 0.0
    try:
        return True, int(index_text), float(delta_text)
    except ValueError as exc:
        raise GoldenError("--inject-perturbation index and delta must be numeric") from exc


def get_actual_tensor(
    manifest: dict[str, Any],
    manifest_path: Path,
    reference: dict[str, Any],
    expected: dict[str, Any],
    observations: dict[TensorKey, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    key = tensor_key(manifest["case_id"], expected)
    observed = observations.get(key)
    if observed is not None:
        return observed, "debug-oracle-observation"
    actual_path = resolve_path(manifest_path.parent, reference.get("actual_fixture"))
    if actual_path is not None and actual_path.exists():
        return read_tensor(actual_path), "fixture-observation"
    return None, "missing-observation"


def validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise GoldenError(f"{path}: schema_version must be {MANIFEST_SCHEMA}")
    for key in ("case_id", "scope", "precision"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise GoldenError(f"{path}: {key} is required")
    if manifest["scope"] not in {"single_layer", "subgraph", "model"}:
        raise GoldenError(f"{path}: scope must be single_layer, subgraph, or model")
    if not isinstance(manifest.get("input_fixtures"), list):
        raise GoldenError(f"{path}: input_fixtures must be a list")
    if not isinstance(manifest.get("expected_tensors"), list) or not manifest["expected_tensors"]:
        raise GoldenError(f"{path}: expected_tensors must be a non-empty list")
    if not isinstance(manifest.get("platform"), dict):
        raise GoldenError(f"{path}: platform is required")
    for fixture in manifest["input_fixtures"]:
        if not isinstance(fixture, dict) or not isinstance(fixture.get("name"), str) or not isinstance(fixture.get("kind"), str) or not isinstance(fixture.get("path"), str):
            raise GoldenError(f"{path}: each input fixture requires name, kind, and path")
        if fixture["kind"].startswith("external-private"):
            continue
        fixture_path = resolve_path(path.parent, fixture["path"])
        if fixture_path is None or not fixture_path.exists():
            raise GoldenError(f"{path}: input fixture does not exist: {fixture['path']}")
        if fixture["kind"] == "tensor":
            read_tensor(fixture_path)
    for tensor_ref in manifest["expected_tensors"]:
        if not isinstance(tensor_ref, dict) or not isinstance(tensor_ref.get("oracle_fixture"), str):
            raise GoldenError(f"{path}: expected_tensors require oracle_fixture")
        if not isinstance(tensor_ref.get("node"), str) or not isinstance(tensor_ref.get("blob"), str):
            raise GoldenError(f"{path}: expected tensor requires node and blob")


def compare_case(
    manifest_path: Path,
    observations: dict[TensorKey, dict[str, Any]],
    injection: str | None,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    validate_manifest(manifest, manifest_path)
    case = {
        "case_id": manifest["case_id"],
        "scope": manifest["scope"],
        "precision": manifest["precision"].upper(),
        "platform": manifest["platform"],
        "input_fixtures": manifest["input_fixtures"],
        "privacy": manifest.get("privacy", "synthetic-or-public"),
        "tensors": [],
        "metrics": [],
        "status": "passed",
    }
    limits = thresholds(manifest)
    for reference in manifest["expected_tensors"]:
        oracle_path = resolve_path(manifest_path.parent, reference["oracle_fixture"])
        if oracle_path is None:
            raise GoldenError(f"{manifest_path}: oracle_fixture missing")
        expected = read_tensor(oracle_path)
        key = tensor_key(manifest["case_id"], expected)
        if expected["node"] != reference["node"] or expected["blob"] != reference["blob"]:
            raise GoldenError(f"{manifest_path}: oracle tensor identity does not match manifest for {key.display}")
        actual, source = get_actual_tensor(manifest, manifest_path, reference, expected, observations)
        tensor_result: dict[str, Any] = {
            "node": expected["node"],
            "blob": expected["blob"],
            "logical_shape": expected["logical_shape"],
            "storage_shape": expected["storage_shape"],
            "layout": expected["layout"],
            "dtype": expected["dtype"],
            "oracle": str(oracle_path),
            "observation_source": source,
            "thresholds": limits,
        }
        if actual is None:
            tensor_result["status"] = "skipped"
            tensor_result["reason"] = "no fixture observation or Unity Debug/Oracle observation supplied"
            case["tensors"].append(tensor_result)
            if manifest.get("allow_missing_observation", False):
                continue
            case["status"] = "failed"
            continue
        for field in ("node", "blob", "logical_shape", "storage_shape", "layout", "dtype"):
            if actual.get(field) != expected.get(field):
                tensor_result["status"] = "failed"
                tensor_result["reason"] = f"contract mismatch for {field}: expected={expected.get(field)!r}, actual={actual.get(field)!r}"
                case["status"] = "failed"
                break
        else:
            actual_values = list(actual["values"])
            inject, index, delta = matches_injection(key, injection)
            if inject:
                if index < 0 or index >= len(actual_values):
                    raise GoldenError(f"injection index out of range for {key.display}: {index}")
                actual_values[index] = float(actual_values[index]) + delta
                tensor_result["injected_perturbation"] = {"index": index, "delta": delta}
            errors = compare_values(expected["values"], actual_values, limits)
            tensor_result.update(errors)
            tensor_result["status"] = "passed" if errors["pass"] else "failed"
            if not errors["pass"]:
                case["status"] = "failed"
        case["tensors"].append(tensor_result)
    for metric in manifest.get("task_metrics", []):
        result = compare_metric(metric)
        case["metrics"].append(result)
        if result["status"] == "failed":
            case["status"] = "failed"
    return case


def compare_metric(metric: Any) -> dict[str, Any]:
    if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
        raise GoldenError("task_metrics entries require a name")
    actual = metric.get("actual")
    expected = metric.get("expected")
    comparison = metric.get("comparison", "equals")
    status = "passed"
    if comparison == "equals":
        status = "passed" if actual == expected else "failed"
    elif comparison == "minimum":
        status = "passed" if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual >= expected else "failed"
    elif comparison == "maximum":
        status = "passed" if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual <= expected else "failed"
    else:
        raise GoldenError(f"unknown metric comparison: {comparison}")
    return {"name": metric["name"], "actual": actual, "expected": expected, "comparison": comparison, "status": status}


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# AIImage Inference Golden Regression", "", f"Status: **{report['status'].upper()}**", ""]
    lines.extend(["## Environment", "", f"- Host: `{report['environment']['host']}`", f"- Python: `{report['environment']['python']}`", f"- Platform: `{report['environment']['platform']}`", ""])
    lines.extend(["## Cases", "", "| Case | Scope | Precision | Status |", "| --- | --- | --- | --- |"])
    for case in report["cases"]:
        lines.append(f"| {case['case_id']} | {case['scope']} | {case['precision']} | {case['status']} |")
    for case in report["cases"]:
        lines.extend(["", f"## {case['case_id']}", "", f"Privacy: `{case['privacy']}`", "", "| Node | Blob | Logical shape | Storage shape | Dtype | Max abs | Max rel | Status |", "| --- | --- | --- | --- | --- | ---: | ---: | --- |"])
        for tensor in case["tensors"]:
            lines.append(
                "| {node} | {blob} | `{logical}` | `{storage}` | {dtype} | {abs:.6g} | {rel:.6g} | {status} |".format(
                    node=tensor["node"], blob=tensor["blob"], logical=tensor["logical_shape"], storage=tensor["storage_shape"], dtype=tensor["dtype"], abs=float(tensor.get("max_abs_error", 0.0)), rel=float(tensor.get("max_rel_error", 0.0)), status=tensor["status"]
                )
            )
            if tensor.get("reason"):
                lines.append(f"\nReason for `{tensor['node']}/{tensor['blob']}`: {tensor['reason']}")
        if case["metrics"]:
            lines.extend(["", "| Metric | Actual | Expected | Status |", "| --- | ---: | ---: | --- |"])
            for metric in case["metrics"]:
                lines.append(f"| {metric['name']} | {metric['actual']} | {metric['expected']} | {metric['status']} |")
    lines.append("")
    return "\n".join(lines)


def discover_manifests(values: Iterable[str]) -> list[Path]:
    manifests: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            manifests.extend(sorted(path.rglob("*.golden.json")))
        elif path.is_file():
            manifests.append(path)
        else:
            raise GoldenError(f"manifest path not found: {path}")
    if not manifests:
        raise GoldenError("no golden manifests found")
    return sorted(set(manifests))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AIImage inference golden tensors and task metrics.")
    parser.add_argument("--manifest", action="append", default=[], help="Manifest file or directory. May be repeated.")
    parser.add_argument("--observation-root", help="Directory containing Unity Debug/Oracle observation JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for golden-report.json and golden-report.md.")
    parser.add_argument("--inject-perturbation", help="Test hook: case_id:node/blob:index:delta")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        script_root = Path(__file__).resolve().parent
        manifests = discover_manifests(args.manifest or [str(script_root / "manifests")])
        observations = observation_index(Path(args.observation_root).resolve() if args.observation_root else None)
        cases = [compare_case(path, observations, args.inject_perturbation) for path in manifests]
        status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
        report = {
            "schema_version": REPORT_SCHEMA,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": status,
            "environment": {"host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0], "cwd": os.getcwd()},
            "cases": cases,
        }
        output_dir = Path(args.output_dir)
        write_json(output_dir / "golden-report.json", report)
        (output_dir / "golden-report.md").write_text(markdown_report(report), encoding="utf-8", newline="\n")
        print(f"Golden regression {status}: {output_dir / 'golden-report.json'}")
        return 0 if status == "passed" else 1
    except GoldenError as exc:
        print(f"golden regression configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
