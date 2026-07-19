"""Configure and build the unmodified ncnn_llm reference CLI with CMake."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent


def run(command: list[str], cwd: Path) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}")


def find_nlohmann_include(global_dir: Path) -> Path:
    matches = sorted(global_dir.glob(".xmake/packages/n/nlohmann_json/*/*/include/nlohmann/json.hpp"))
    if not matches:
        raise FileNotFoundError(
            "nlohmann/json.hpp was not found; pass --nlohmann-include or let xmake install nlohmann_json first"
        )
    return matches[-1].parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncnn-source", type=Path, required=True, help="Tencent/ncnn master source directory")
    parser.add_argument("--nlohmann-include", type=Path)
    parser.add_argument("--xmake-global", type=Path, default=ROOT / "tmp" / "xmake-global")
    parser.add_argument("--build-dir", type=Path, default=BASELINE_DIR / "_build" / "reference_cli")
    parser.add_argument("--config", choices=("Release", "RelWithDebInfo", "Debug"), default="Release")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ncnn_source = args.ncnn_source.resolve()
    nlohmann_include = (
        args.nlohmann_include.resolve()
        if args.nlohmann_include
        else find_nlohmann_include(args.xmake_global.resolve())
    )
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(BASELINE_DIR / "reference_cli"),
        "-B",
        str(build_dir),
        "-A",
        "x64",
        f"-DNCNN_SOURCE_DIR={ncnn_source}",
        f"-DNCNN_LLM_SOURCE_DIR={ROOT / 'ref' / 'ncnn_llm-main'}",
        f"-DNLOHMANN_JSON_INCLUDE_DIR={nlohmann_include}",
    ]
    run(configure, ROOT)
    run(["cmake", "--build", str(build_dir), "--config", args.config, "--target", "llm_ncnn_run", "-j", "8"], ROOT)
    executable = build_dir / args.config / "llm_ncnn_run.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"build completed but executable is missing: {executable}")
    print(f"Reference CLI: {executable}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"reference CLI build failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

