from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    diffusers: Path
    onnx: Path
    ncnn: Path


def force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sd15inpainting2ncnnExporter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_output_dirs(root: Path) -> OutputDirs:
    diffusers = root / "diffusers"
    onnx = root / "onnx"
    ncnn = root / "ncnn"
    for path in (root, diffusers, onnx, ncnn):
        path.mkdir(parents=True, exist_ok=True)
    return OutputDirs(root=root, diffusers=diffusers, onnx=onnx, ncnn=ncnn)


def log_versions(logger: logging.Logger) -> None:
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version.replace("\n", " "))

    packages = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "onnx",
        "onnxruntime",
        "onnxsim",
        "optimum",
        "pnnx",
    ]

    for package_name in packages:
        try:
            module = __import__(package_name)
            version = getattr(module, "__version__", "unknown")
            logger.info("%s=%s", package_name, version)
        except Exception:
            logger.info("%s=not-installed", package_name)


def run_command(command: list[str], logger: logging.Logger, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    logger.info("RUN: %s", " ".join(command))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if process.stdout:
        logger.info(process.stdout.rstrip())
    if process.stderr:
        logger.info(process.stderr.rstrip())
    if process.returncode != 0:
        raise ConversionError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    return process


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_file_and_sidecars(path: Path) -> None:
    candidates = [
        path,
        path.parent / f"{path.name}.data",
        path.parent / f"{path.name}_data",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_file():
                candidate.unlink()
            else:
                shutil.rmtree(candidate)


def save_onnx_model(model: Any, destination: Path, external_data: bool) -> None:
    import onnx

    cleanup_file_and_sidecars(destination)
    if external_data:
        onnx.save_model(
            model,
            str(destination),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{destination.name}.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        return
    onnx.save_model(model, str(destination))


def copy_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def looks_like_windows_executable(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def resolve_explicit_or_path(candidate: str) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser().resolve()
    if not path.exists():
        raise ConversionError(f"Explicit tool path does not exist: {path}")
    return path


def shutil_which(names: list[str]) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return None
