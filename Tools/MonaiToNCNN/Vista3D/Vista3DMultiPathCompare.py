from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUESTED_LABELS = [
    {"key": "ventricles", "aliases": ["ventricle", "ventricles"]},
    {"key": "skull", "aliases": ["skull"]},
    {"key": "white_matter", "aliases": ["white matter", "white_matter"]},
    {"key": "hippocampus", "aliases": ["hippocampus"]},
    {"key": "brain", "aliases": ["brain"]},
]

EXPORTED_ARTIFACTS = (
    "summary.txt",
    "timings.json",
    "resource_stats.json",
    "run_manifest.json",
    "baseline_manifest.json",
    "resource_snapshots.txt",
    "gpu_resource_stats.txt",
    "gpu_resource_stats_after_release.txt",
    "layer_runtime_profile.tsv",
    "labelmap_restored.nii.gz",
    "labelmap_restored.nii",
    "labelmap_restored.nrrd",
)

REQUIRED_BY_KIND = {
    "export": ("manifest.json",),
    "official": ("baseline_manifest.json", "timings.json", "resource_stats.json", "summary.txt"),
    "unity": ("run_manifest.json", "timings.json", "resource_stats.json", "summary.txt"),
}


class CompareError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run Vista3D official baseline and Unity buffer/pack4 comparisons for selected labels.")
    parser.add_argument(
        "--input",
        default=r"E:\Projects\CTData\sliceexampledata2\CT_Electrodes\CT_Electrodes.nii.gz",
        help="Input CT path used for all runs.",
    )
    parser.add_argument(
        "--compare-root",
        default=str(root / "manual_test" / "vista3d_label_compare"),
        help="Root folder for combined comparison outputs.",
    )
    parser.add_argument(
        "--batch-timeout-minutes",
        type=int,
        default=10,
        help="Timeout passed to official and Unity batch helpers.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated subset of requested label keys, for example brain,skull.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_print(text: str, stream=None) -> None:
    target = stream or sys.stdout
    try:
        target.write(text)
    except UnicodeEncodeError:
        encoded = text.encode(getattr(target, "encoding", "utf-8") or "utf-8", errors="replace")
        decoded = encoded.decode(getattr(target, "encoding", "utf-8") or "utf-8", errors="replace")
        target.write(decoded)
    try:
        target.flush()
    except Exception:
        pass


def parse_selected_label_keys(raw: str) -> set[str] | None:
    keys = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return keys or None


def case_stem_from_input(input_path: Path) -> str:
    name = input_path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return input_path.stem


def tail_lines(text: str, count: int = 40) -> list[str]:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[-count:]


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_requested_labels(metadata: dict[str, Any], selected_keys: set[str] | None) -> list[dict[str, Any]]:
    channel_def = (
        metadata.get("network_data_format", {})
        .get("outputs", {})
        .get("pred", {})
        .get("channel_def", {})
    )
    label_items = {int(key): str(value) for key, value in channel_def.items()}

    resolved: list[dict[str, Any]] = []
    for item in REQUESTED_LABELS:
        requested_key = str(item["key"])
        if selected_keys is not None and requested_key.lower() not in selected_keys:
            continue

        matched_label = None
        matched_name = None
        for label_value, label_name in label_items.items():
            lower_name = label_name.lower()
            if any(alias in lower_name for alias in item["aliases"]):
                matched_label = label_value
                matched_name = label_name
                break

        resolved.append(
            {
                "requested_key": requested_key,
                "label_value": matched_label,
                "label_name": matched_name,
                "available": matched_label is not None,
            }
        )
    return resolved


def run_command(command: list[str], env_overrides: dict[str, str], cwd: Path, timeout_seconds: int | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(env_overrides)
    try:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if process.stdout:
            safe_print(process.stdout)
        if process.stderr:
            safe_print(process.stderr, stream=sys.stderr)
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": int(process.returncode),
            "ok": process.returncode == 0,
            "stdout_tail": tail_lines(process.stdout),
            "stderr_tail": tail_lines(process.stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        stdout_text = error.stdout or ""
        stderr_text = error.stderr or ""
        if isinstance(stdout_text, bytes):
            stdout_text = stdout_text.decode("utf-8", errors="replace")
        if isinstance(stderr_text, bytes):
            stderr_text = stderr_text.decode("utf-8", errors="replace")
        if stdout_text:
            safe_print(stdout_text)
        if stderr_text:
            safe_print(stderr_text, stream=sys.stderr)
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": 124,
            "ok": False,
            "stdout_tail": tail_lines(stdout_text),
            "stderr_tail": tail_lines(stderr_text),
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
        }


def run_python_tool(script_path: Path, args: list[str], cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    return run_command([sys.executable, str(script_path), *args], {}, cwd, timeout_seconds=timeout_seconds)


def validate_required_outputs(path_dir: Path, kind: str) -> list[str]:
    required = REQUIRED_BY_KIND.get(kind, ())
    return [name for name in required if not (path_dir / name).exists()]


def find_restored_labelmap(path_dir: Path) -> Path | None:
    for name in ("labelmap_restored.nii.gz", "labelmap_restored.nii", "labelmap_restored.nrrd"):
        candidate = path_dir / name
        if candidate.exists():
            return candidate
    return None


def find_existing_subset(path_dir: Path, label_name: str) -> Path | None:
    subset_dir = path_dir / "label_subsets"
    for name in (
        f"{label_name}_mask_restored.nii.gz",
        f"{label_name}_mask_restored.nii",
        f"{label_name}_mask_restored.nrrd",
    ):
        candidate = subset_dir / name
        if candidate.exists():
            return candidate
    return None


def compute_subset_output_path(labelmap_path: Path, label_name: str) -> Path:
    subset_dir = labelmap_path.parent / "label_subsets"
    if labelmap_path.name.lower().endswith(".nii.gz"):
        return subset_dir / f"{label_name}_mask_restored.nii.gz"
    if labelmap_path.suffix.lower() == ".nii":
        return subset_dir / f"{label_name}_mask_restored.nii"
    if labelmap_path.suffix.lower() == ".nrrd":
        return subset_dir / f"{label_name}_mask_restored.nrrd"
    raise CompareError(f"Unsupported restored labelmap format: {labelmap_path}")


def extract_subset_if_possible(tools_root: Path, path_dir: Path, label_value: int, label_name: str) -> dict[str, Any]:
    existing_subset = find_existing_subset(path_dir, label_name)
    if existing_subset is not None:
        return {
            "status": "completed",
            "output_path": str(existing_subset),
            "manifest_json": str(existing_subset.with_suffix(existing_subset.suffix + ".json")) if existing_subset.with_suffix(existing_subset.suffix + ".json").exists() else None,
        }

    labelmap_path = find_restored_labelmap(path_dir)
    if labelmap_path is None:
        return {"status": "missing_labelmap", "output_path": None}

    output_path = compute_subset_output_path(labelmap_path, label_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str((tools_root / "Vista3D" / "ExtractLabelSubset.py").resolve()),
        "--labelmap",
        str(labelmap_path),
        "--label-value",
        str(label_value),
        "--output",
        str(output_path),
    ]
    command_result = run_command(command, {}, tools_root)
    if not command_result["ok"] or not output_path.exists():
        return {
            "status": "failed",
            "output_path": None,
            "command": command_result,
        }

    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    return {
        "status": "completed",
        "output_path": str(output_path),
        "manifest_json": str(sidecar) if sidecar.exists() else None,
    }


def summarize_timings(path_dir: Path) -> dict[str, Any] | None:
    payload = try_load_json(path_dir / "timings.json")
    if payload is None:
        return None
    return {
        key: payload.get(key)
        for key in (
            "load_model_ms",
            "prepare_input_ms",
            "inference_ms",
            "postprocess_ms",
            "baseline_compare_ms",
            "total_elapsed_ms",
            "path_mode",
            "layer_profile_path_kind",
            "layer_profile_total_ms",
        )
        if key in payload
    }


def summarize_resource_stats(path_dir: Path) -> dict[str, Any] | None:
    payload = try_load_json(path_dir / "resource_stats.json")
    if payload is None:
        return None

    summary: dict[str, Any] = {
        key: payload.get(key)
        for key in (
            "path_mode",
            "process_private_mb",
            "process_rss_mb",
            "process_vms_mb",
            "process_working_set_mb",
            "managed_heap_mb",
            "graphics_driver_mb",
            "unity_rendertexture_object_count",
            "torch_cuda_available",
            "torch_device",
            "torch_cuda_device_count",
            "note",
        )
        if key in payload
    }

    torch_cuda_memory = payload.get("torch_cuda_memory")
    if isinstance(torch_cuda_memory, dict):
        for key in ("allocated_mb", "reserved_mb", "max_allocated_mb", "max_reserved_mb", "device_name"):
            if key in torch_cuda_memory:
                summary[f"torch_cuda_{key}"] = torch_cuda_memory.get(key)

    gpu_tracker = payload.get("gpu_tracker")
    if isinstance(gpu_tracker, dict):
        for key in (
            "current_total_mb",
            "current_buffers_mb",
            "current_rendertextures_mb",
            "peak_total_mb",
            "peak_buffers_mb",
            "peak_rendertextures_mb",
            "live_buffer_count",
            "live_rendertexture_count",
            "peak_buffer_count",
            "peak_rendertexture_count",
            "low_memory_warning_count",
        ):
            if key in gpu_tracker:
                summary[f"gpu_{key}"] = gpu_tracker.get(key)
    return summary


def summarize_compare_metrics(path_dir: Path) -> dict[str, Any] | None:
    manifest = try_load_json(path_dir / "run_manifest.json")
    if manifest is None:
        return None

    compare = manifest.get("baseline_compare")
    if not isinstance(compare, dict):
        return None

    summary: dict[str, Any] = {}
    probs = compare.get("probs")
    if isinstance(probs, dict):
        summary["probs_mean_abs"] = probs.get("mean_abs")
        summary["probs_max_abs"] = probs.get("max_abs")

    for key in ("masks", "labelmap"):
        payload = compare.get(key)
        if isinstance(payload, dict):
            summary[f"{key}_equal_ratio"] = payload.get("equal_ratio")
            summary[f"{key}_mismatch_count"] = payload.get("mismatch_count")
            summary[f"{key}_max_abs"] = payload.get("max_abs")
    return summary or None


def build_path_result(path_dir: Path, kind: str, command_result: dict[str, Any] | None, subset_result: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "output_dir": str(path_dir),
        "kind": kind,
    }

    artifacts: dict[str, str] = {}
    for name in EXPORTED_ARTIFACTS:
        candidate = path_dir / name
        if candidate.exists():
            artifacts[name] = str(candidate)
    result["artifacts"] = artifacts

    labelmap_path = find_restored_labelmap(path_dir)
    if labelmap_path is not None:
        result["labelmap_path"] = str(labelmap_path)

    missing_files = validate_required_outputs(path_dir, kind)
    if missing_files:
        result["missing_files"] = missing_files

    result["timings_summary"] = summarize_timings(path_dir)
    result["resource_summary"] = summarize_resource_stats(path_dir)

    compare_summary = summarize_compare_metrics(path_dir)
    if compare_summary is not None:
        result["baseline_compare_summary"] = compare_summary

    if subset_result is not None:
        result["subset_result"] = subset_result

    if command_result is not None:
        result["command"] = command_result
        if not command_result.get("ok", False):
            result["status"] = "failed"
            result["error"] = f"Command failed with exit code {command_result.get('returncode')}"
        else:
            result["status"] = "completed"
    else:
        result["status"] = "completed"

    if missing_files:
        result["status"] = "failed"
        result.setdefault("error", "Required artifacts are missing.")

    return result


def build_compare_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        f"input_path={summary['input_path']}",
        f"compare_root={summary['compare_root']}",
        f"batch_timeout_minutes={summary['batch_timeout_minutes']}",
    ]

    requested_labels = summary.get("requested_labels") or []
    if requested_labels:
        lines.append("requested_labels:")
        for item in requested_labels:
            if item.get("available"):
                lines.append(
                    "  "
                    + f"{item['requested_key']}={item['label_value']} ({item['label_name']})"
                )
            else:
                lines.append("  " + f"{item['requested_key']}=unavailable")

    for run in summary.get("runs") or []:
        lines.append("")
        lines.append(
            f"label={run.get('requested_key')} | status={run.get('status')} | "
            f"label_value={run.get('label_value')} | label_name={run.get('label_name')}"
        )
        if run.get("message"):
            lines.append(f"message={run['message']}")
        if run.get("export_manifest"):
            lines.append(f"export_manifest={run['export_manifest']}")

        for path_key, title in (
            ("official", "official_python"),
            ("unity_compute_buffer", "unity_compute_buffer"),
            ("unity_pack4_rendertexture", "unity_pack4_rendertexture"),
        ):
            payload = run.get(path_key)
            if not isinstance(payload, dict):
                continue

            lines.append(f"{title}.status={payload.get('status')}")
            lines.append(f"{title}.output_dir={payload.get('output_dir')}")
            if payload.get("error"):
                lines.append(f"{title}.error={payload['error']}")
            if payload.get("labelmap_path"):
                lines.append(f"{title}.labelmap_path={payload['labelmap_path']}")
            if payload.get("missing_files"):
                lines.append(f"{title}.missing_files={','.join(payload['missing_files'])}")

            subset_result = payload.get("subset_result")
            if isinstance(subset_result, dict):
                lines.append(f"{title}.subset_status={subset_result.get('status')}")
                if subset_result.get("output_path"):
                    lines.append(f"{title}.subset_output={subset_result['output_path']}")

            timings = payload.get("timings_summary")
            if isinstance(timings, dict):
                for key, value in timings.items():
                    lines.append(f"{title}.timings.{key}={value}")

            resources = payload.get("resource_summary")
            if isinstance(resources, dict):
                for key, value in resources.items():
                    lines.append(f"{title}.resources.{key}={value}")

            compare = payload.get("baseline_compare_summary")
            if isinstance(compare, dict):
                for key, value in compare.items():
                    lines.append(f"{title}.compare.{key}={value}")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    tools_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).resolve()
    compare_root = Path(args.compare_root).resolve()
    compare_root.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise CompareError(f"Input CT not found: {input_path}")

    metadata = load_json(tools_root / "bundle_cache" / "model-zoo-dev" / "models" / "vista3d" / "configs" / "metadata.json")
    labels = resolve_requested_labels(metadata, parse_selected_label_keys(args.labels))
    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "compare_root": str(compare_root),
        "batch_timeout_minutes": args.batch_timeout_minutes,
        "requested_labels": labels,
        "runs": [],
    }

    export_script = (tools_root / "Vista3D" / "Vista3DFixedPromptExport.py").resolve()
    baseline_script = (tools_root / "Vista3D" / "Vista3DBaseline.py").resolve()
    run_unity_bat = str((tools_root / "RunVista3DUnityDebug.bat").resolve())
    python_timeout_seconds = max(60, int(args.batch_timeout_minutes) * 60)

    for label in labels:
        if not label["available"]:
            summary["runs"].append(
                {
                    "requested_key": label["requested_key"],
                    "label_name": None,
                    "label_value": None,
                    "status": "unavailable",
                    "message": "Label not present in official Vista3D metadata.",
                }
            )
            continue

        label_value = int(label["label_value"])
        label_name = str(label["label_name"])
        case_slug = f"{case_stem_from_input(input_path)}_{label['requested_key']}"

        export_dir = compare_root / "exports" / case_slug
        baseline_root = compare_root / "official"
        baseline_case_name = case_slug
        baseline_case_dir = baseline_root / baseline_case_name
        unity_buffer_dir = compare_root / "unity_buffer" / case_slug
        unity_pack4_dir = compare_root / "unity_pack4" / case_slug
        manifest_path = export_dir / "manifest.json"

        reset_directory(export_dir)
        reset_directory(baseline_case_dir)
        reset_directory(unity_buffer_dir)
        reset_directory(unity_pack4_dir)

        common_env = {
            "AIIMAGE_BATCH_TIMEOUT_MINUTES": str(args.batch_timeout_minutes),
            "AIIMAGE_VISTA_LABEL_PROMPT": str(label_value),
            "AIIMAGE_VISTA_LABEL_NAME": label["requested_key"],
            "AIIMAGE_VISTA_INPUT_PATH": str(input_path),
            "AIIMAGE_VISTA_CASE_TAG": case_slug,
            "AIIMAGE_VISTA_EXPORT_OUTPUT_DIR": str(export_dir),
            "AIIMAGE_VISTA_EXPORT_MANIFEST": str(manifest_path),
            "AIIMAGE_VISTA_CASE_NAME": baseline_case_name,
            "AIIMAGE_VISTA_BASELINE_OUTPUT_DIR": str(baseline_root),
            "AIIMAGE_MONAI_CLEAR_TEMP_POOL_EACH_PATCH": "1",
            "AIIMAGE_MONAI_TEMP_POOL_CLEAR_INTERVAL": "1",
            "AIIMAGE_MONAI_YIELD_INTERVAL": "1",
            "AIIMAGE_MONAI_MANAGED_CLEANUP_INTERVAL": "1",
            "AIIMAGE_MONAI_RESOURCE_SNAPSHOT_INTERVAL": "1",
            "AIIMAGE_MONAI_ABORT_PRIVATE_MEMORY_MB": "8192",
            "AIIMAGE_REPRO_TEMP_POOL": "0",
        }

        export_command = run_python_tool(
            export_script,
            [
                "--label-prompt",
                str(label_value),
                "--label-name",
                str(label["requested_key"]),
                "--case-tag",
                case_slug,
                "--output-dir",
                str(export_dir),
            ],
            tools_root,
            timeout_seconds=python_timeout_seconds,
        )
        export_missing = validate_required_outputs(export_dir, "export")
        if not export_command["ok"] or export_missing:
            summary["runs"].append(
                {
                    "requested_key": label["requested_key"],
                    "label_value": label_value,
                    "label_name": label_name,
                    "status": "failed",
                    "message": "Vista3D export step failed.",
                    "export_dir": str(export_dir),
                    "export_command": export_command,
                    "export_missing_files": export_missing,
                }
            )
            continue

        baseline_command = run_python_tool(
            baseline_script,
            [
                "--input",
                str(input_path),
                "--label-prompt",
                str(label_value),
                "--label-name",
                str(label["requested_key"]),
                "--case-name",
                baseline_case_name,
                "--output-dir",
                str(baseline_root),
                "--ncnn-manifest",
                str(manifest_path),
            ],
            tools_root,
            timeout_seconds=python_timeout_seconds,
        )
        official_subset = extract_subset_if_possible(tools_root, baseline_case_dir, label_value, label["requested_key"]) if baseline_command["ok"] else None
        official_result = build_path_result(baseline_case_dir, "official", baseline_command, official_subset)

        run_record: dict[str, Any] = {
            "requested_key": label["requested_key"],
            "label_value": label_value,
            "label_name": label_name,
            "export_manifest": str(manifest_path),
            "official": official_result,
        }

        if official_result["status"] != "completed":
            run_record["status"] = "failed"
            run_record["message"] = "Official baseline failed, so Unity comparison was skipped."
            summary["runs"].append(run_record)
            continue

        buffer_env = {
            **common_env,
            "AIIMAGE_MONAI_BASELINE_MANIFEST": str((baseline_case_dir / "baseline_manifest.json").resolve()),
            "AIIMAGE_MONAI_INPUT_PATHS": str(input_path),
            "AIIMAGE_MONAI_USE_BASELINE_TENSOR": "1",
            "AIIMAGE_MONAI_COMPARE_BASELINE": "1",
            "AIIMAGE_MONAI_ENABLE_DUMP": "1",
            "AIIMAGE_MONAI_DUMP_LARGE_TENSORS": "0",
            "AIIMAGE_MONAI_PATCH_INPUT_MODE": "compute_buffer",
            "AIIMAGE_MONAI_FORCE_BUFFER_ALL": "0",
            "AIIMAGE_MONAI_FORCE_CPU_GEMM": "0",
            "AIIMAGE_MONAI_FORCE_BUFFER_OUTPUTS_DIMS4": "0",
            "AIIMAGE_MONAI_PACK4_ONLY_GUARD": "0",
            "AIIMAGE_MONAI_PROBE_ONLY": "0",
            "AIIMAGE_MONAI_MAX_PATCHES": "0",
            "AIIMAGE_MONAI_OUTPUT_DIR": str(unity_buffer_dir),
            "AIIMAGE_MONAI_CASE_NAME": case_slug,
        }

        pack4_env = {
            **common_env,
            "AIIMAGE_MONAI_BASELINE_MANIFEST": str((baseline_case_dir / "baseline_manifest.json").resolve()),
            "AIIMAGE_MONAI_INPUT_PATHS": str(input_path),
            "AIIMAGE_MONAI_USE_BASELINE_TENSOR": "1",
            "AIIMAGE_MONAI_COMPARE_BASELINE": "1",
            "AIIMAGE_MONAI_ENABLE_DUMP": "1",
            "AIIMAGE_MONAI_DUMP_LARGE_TENSORS": "0",
            "AIIMAGE_MONAI_PATCH_INPUT_MODE": "pack4_rt",
            "AIIMAGE_MONAI_FORCE_BUFFER_ALL": "0",
            "AIIMAGE_MONAI_FORCE_CPU_GEMM": "0",
            "AIIMAGE_MONAI_FORCE_BUFFER_OUTPUTS_DIMS4": "0",
            "AIIMAGE_MONAI_PACK4_ONLY_GUARD": "1",
            "AIIMAGE_MONAI_PROBE_ONLY": "0",
            "AIIMAGE_MONAI_MAX_PATCHES": "0",
            "AIIMAGE_MONAI_OUTPUT_DIR": str(unity_pack4_dir),
            "AIIMAGE_MONAI_CASE_NAME": case_slug,
        }

        buffer_command = run_command(["cmd", "/c", run_unity_bat], buffer_env, tools_root)
        buffer_subset = extract_subset_if_possible(tools_root, unity_buffer_dir, label_value, label["requested_key"]) if buffer_command["ok"] else None
        buffer_result = build_path_result(unity_buffer_dir, "unity", buffer_command, buffer_subset)

        pack4_command = run_command(["cmd", "/c", run_unity_bat], pack4_env, tools_root)
        pack4_subset = extract_subset_if_possible(tools_root, unity_pack4_dir, label_value, label["requested_key"]) if pack4_command["ok"] else None
        pack4_result = build_path_result(unity_pack4_dir, "unity", pack4_command, pack4_subset)

        run_record["unity_compute_buffer"] = buffer_result
        run_record["unity_pack4_rendertexture"] = pack4_result

        statuses = [
            official_result.get("status"),
            buffer_result.get("status"),
            pack4_result.get("status"),
        ]
        run_record["status"] = "completed" if all(status == "completed" for status in statuses) else "partial_failed"
        summary["runs"].append(run_record)

    save_json(compare_root / "compare_manifest.json", summary)
    save_text(compare_root / "compare_summary.txt", build_compare_summary_text(summary))
    print(str(compare_root / "compare_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
