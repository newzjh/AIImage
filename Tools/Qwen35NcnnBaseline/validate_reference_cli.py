"""Run the exact Qwen3.5 image CLI workflow and persist validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = "请识别图片中的竖排手写繁体中文，按从右到左的列顺序抄写，只输出识别结果。"
DEFAULT_MARKER_GROUPS = (
    ("仍未忘跟你約定", "仍未忘跟你约定"),
    ("決心忘記我便記不起", "决心忘记我便记不起"),
    ("剪影的你輪廓太好看", "剪影的你轮廓太好看"),
    ("還記得當天旅館的門牌", "还记得当天旅馆的门牌"),
    ("漫天黃葉滿飛", "漫天黄叶满飞"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assistant_text(stdout: str) -> str:
    marker = "Assistant: "
    if marker not in stdout:
        return ""
    tail = stdout.rsplit(marker, 1)[1]
    return tail.split("\nUser:", 1)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=ROOT / "ref" / "ncnn_llm-main")
    parser.add_argument("--model", default="./assets/qwen3.5_0.8b")
    parser.add_argument("--image", default="test.jpg")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--min-marker-groups", type=int, default=2)
    parser.add_argument(
        "--no-builtin-tools",
        action="store_true",
        help="append the reference CLI flag that disables random/add demo tools",
    )
    parser.add_argument("--output", type=Path, default=BASELINE_DIR / "reports" / "reference_cli_validation.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    executable = args.executable.resolve()
    image_path = (project_dir / args.image).resolve()
    model_path = (project_dir / args.model).resolve()
    for path in (executable, image_path, model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    command = [str(executable), "--model", args.model, "--image", args.image]
    if args.no_builtin_tools:
        command.append("--no-builtin-tools")
    print("+", subprocess.list2cmdline(command), flush=True)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=project_dir,
        input=(args.prompt + "\nexit\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_seconds,
        check=False,
    )
    elapsed = time.time() - started
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assistant_text = parse_assistant_text(stdout)
    marker_hits = [variants for variants in DEFAULT_MARKER_GROUPS if any(marker in assistant_text for marker in variants)]
    checks = {
        "exit_code_zero": completed.returncode == 0,
        "image_loaded": "Image loaded:" in stderr,
        "assistant_text_nonempty": bool(assistant_text),
        "semantic_marker_groups": len(marker_hits) >= args.min_marker_groups,
        "no_nan_or_inf_text": "nan" not in assistant_text.lower() and "inf" not in assistant_text.lower(),
    }
    report = {
        "schema_version": 1,
        "command_equivalent": (
            "./llm_ncnn_run --model ./assets/qwen3.5_0.8b --image test.jpg"
            + (" --no-builtin-tools" if args.no_builtin_tools else "")
        ),
        "actual_command": command,
        "working_directory": str(project_dir),
        "prompt": args.prompt,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "model_directory": str(model_path),
        "image": {"path": str(image_path), "sha256": sha256(image_path)},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "assistant_text": assistant_text,
        "marker_group_hits": [list(group) for group in marker_hits],
        "checks": checks,
        "valid": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".stdout.txt").write_text(stdout, encoding="utf-8")
    args.output.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
    print(assistant_text)
    print(f"Wrote validation evidence to {args.output.resolve()}")
    return 0 if report["valid"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        print(f"reference CLI timed out after {exc.timeout} seconds", file=sys.stderr)
        raise SystemExit(4) from exc
    except Exception as exc:
        print(f"reference CLI validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
