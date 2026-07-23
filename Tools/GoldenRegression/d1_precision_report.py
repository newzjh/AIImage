#!/usr/bin/env python3
"""Summarize measured FP32/FP16 D1 runs without treating debug readback as production IO."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_vector(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        values.append(float(parts[1]))
    if not values:
        raise ValueError(f"no indexed values in {path}")
    return values


def read_scores(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            result[parts[0]] = float(parts[1])
    if not result:
        raise ValueError(f"no label scores in {path}")
    return result


def vector_error(reference: list[float], actual: list[float]) -> dict[str, float | int]:
    if len(reference) != len(actual):
        raise ValueError(f"vector size mismatch: {len(reference)} vs {len(actual)}")
    absolute = [abs(a - b) for a, b in zip(reference, actual)]
    relative = [delta / max(abs(a), 1e-8) for a, delta in zip(reference, absolute)]
    return {
        "count": len(reference),
        "max_abs": max(absolute),
        "mean_abs": sum(absolute) / len(absolute),
        "rms": math.sqrt(sum(value * value for value in absolute) / len(absolute)),
        "max_rel": max(relative),
    }


def parse_metric_detail(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in value.split("|"):
        key, separator, field_value = field.strip().partition("=")
        if separator:
            result[key] = field_value
    return result


def benchmark_index(directory: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime):
        report = load_json(path)
        if report.get("schemaVersion") != "aiimage.inference.runtime-benchmark/v1":
            continue
        key = (str(report.get("runner", "")), str(report.get("activationDtype", "")), str(report.get("weightDtype", "")))
        report["report_path"] = str(path)
        result[key] = report
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = ["# D1 FP16 Mixed Precision Measurements", "", "## Benchmarks", "", "| Runner | Dtype | Status | Elapsed ms | Peak temp RT bytes | Peak total bytes | Task metric |", "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for benchmark in report["benchmarks"]:
        lines.append(
            "| {runner} | {dtype} | {status} | {elapsed} | {temp} | {total} | {metric}={value} {detail} |".format(
                runner=benchmark.get("runner", ""),
                dtype=precision_label(benchmark),
                status=benchmark.get("status", ""),
                elapsed=benchmark.get("elapsedMs", 0),
                temp=benchmark.get("peakTemporaryTextureBytes", 0),
                total=benchmark.get("peakTotalBytes", 0),
                metric=benchmark.get("taskMetricName", ""),
                value=benchmark.get("taskMetricValue", ""),
                detail=benchmark.get("taskMetricDetail", ""),
            )
        )

    clip = report["clip"]
    lines.extend(["", "## CLIP", "", f"- Embedding count: `{clip['embedding_error']['count']}`", f"- Max / mean absolute error: `{clip['embedding_error']['max_abs']:.8g}` / `{clip['embedding_error']['mean_abs']:.8g}`", f"- Top-1 FP32 / FP16: `{clip['top1_fp32']}` / `{clip['top1_fp16']}`", f"- Top-1 stable: `{clip['top1_stable']}`", f"- Top-1 probability delta: `{clip['top1_probability_abs_delta']:.8g}`"])

    matting = report["matting"]
    lines.extend(["", "## Matting", "", f"- Matte size: `{matting.get('size', '')}`", f"- Max / mean alpha error (0-1): `{matting['max_abs_u8'] / 255.0:.8g}` / `{matting['mean_abs_u8'] / 255.0:.8g}`", f"- Foreground IoU at alpha >= 0.5: `{matting['foreground_iou_128']:.8g}`"])

    if report.get("medical"):
        medical = report["medical"]
        lines.extend(["", "## Medical Probe", "", f"- Status: `{medical.get('status', '')}`", f"- Error: `{medical.get('error', '')}`"])
    lines.append("")
    return "\n".join(lines)


def precision_label(benchmark: dict[str, Any]) -> str:
    activation = str(benchmark.get("activationDtype", ""))
    weight = str(benchmark.get("weightDtype", ""))
    return activation if activation == weight else activation + " activation / " + weight + " weights"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a D1 FP32 versus FP16 error and benchmark report.")
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--clip-fp32-dir", required=True, type=Path)
    parser.add_argument("--clip-fp16-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    benchmarks = benchmark_index(args.benchmark_dir)
    matting_fp16_key = ("matting", "Float32", "Float16")
    required = [("clip", "Float32", "Float32"), ("clip", "Float16", "Float16"), ("matting", "Float32", "Float32"), matting_fp16_key]
    missing = [f"{runner}/{activation}/{weight}" for runner, activation, weight in required if (runner, activation, weight) not in benchmarks]
    if missing:
        raise SystemExit("missing benchmark reports: " + ", ".join(missing))

    fp32_embedding = read_vector(args.clip_fp32_dir / "image_embedding.txt")
    fp16_embedding = read_vector(args.clip_fp16_dir / "image_embedding.txt")
    fp32_scores = read_scores(args.clip_fp32_dir / "scores.txt")
    fp16_scores = read_scores(args.clip_fp16_dir / "scores.txt")
    top1_fp32 = max(fp32_scores, key=fp32_scores.get)
    top1_fp16 = max(fp16_scores, key=fp16_scores.get)

    matting_detail = parse_metric_detail(str(benchmarks[matting_fp16_key].get("taskMetricDetail", "")))
    required_matte_metrics = ("fp32_mean_abs_u8", "fp32_max_abs_u8", "fp32_foreground_iou_128")
    missing_matte_metrics = [key for key in required_matte_metrics if key not in matting_detail]
    if missing_matte_metrics:
        raise SystemExit("FP16 Matting benchmark has no FP32 comparison metrics: " + ", ".join(missing_matte_metrics))

    report: dict[str, Any] = {
        "schema_version": "aiimage.inference.d1-precision-report/v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmarks": [benchmarks[key] for key in required],
        "clip": {
            "fp32_dump": str(args.clip_fp32_dir),
            "fp16_dump": str(args.clip_fp16_dir),
            "embedding_error": vector_error(fp32_embedding, fp16_embedding),
            "top1_fp32": top1_fp32,
            "top1_fp16": top1_fp16,
            "top1_stable": top1_fp32 == top1_fp16,
            "top1_probability_abs_delta": abs(fp32_scores[top1_fp32] - fp16_scores.get(top1_fp32, 0.0)),
        },
        "matting": {
            "size": str(benchmarks[matting_fp16_key].get("taskMetricDetail", "")).split("|")[0].strip(),
            "mean_abs_u8": float(matting_detail["fp32_mean_abs_u8"]),
            "max_abs_u8": int(matting_detail["fp32_max_abs_u8"]),
            "foreground_iou_128": float(matting_detail["fp32_foreground_iou_128"]),
        },
    }
    medical = benchmarks.get(("monai-probe", "Float16", "Float16"))
    if medical is not None:
        report["medical"] = medical

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "d1-precision-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "d1-precision-report.md").write_text(markdown(report), encoding="utf-8")
    print(args.output_dir / "d1-precision-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
