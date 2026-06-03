from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .onnx_export import OnnxExportArtifact
from .utils import (
    ConversionError,
    copy_or_replace,
    looks_like_windows_executable,
    resolve_explicit_or_path,
    run_command,
    shutil_which,
)


@dataclass(frozen=True)
class NcnnArtifact:
    name: str
    converter: str
    input_onnx: str
    raw_param: str
    raw_bin: str
    final_param: str
    final_bin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "converter": self.converter,
            "input_onnx": self.input_onnx,
            "raw_param": self.raw_param,
            "raw_bin": self.raw_bin,
            "final_param": self.final_param,
            "final_bin": self.final_bin,
        }


def _candidate_paths(tools_root: Path, names: list[str]) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [
        tools_root,
        tools_root.parent / "MonaiToNCNN",
        tools_root.parent.parent / "RealESRGAN",
        tools_root.parent.parent / "ref",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for name in names:
            candidates.extend(root.rglob(name))
    def score(path: Path) -> tuple[int, int, str]:
        path_text = str(path).lower()
        arch_score = 0
        if "\\x64\\" in path_text or "/x64/" in path_text:
            arch_score = 3
        elif "\\arm64\\" in path_text or "/arm64/" in path_text:
            arch_score = 1
        elif "\\x86\\" in path_text or "/x86/" in path_text:
            arch_score = 0
        name_score = 1 if path.suffix.lower() == ".exe" else 0
        return (arch_score, name_score, path_text)
    candidates.sort(key=score, reverse=True)
    return candidates


def _resolve_windows_tool(explicit: str, tools_root: Path, names: list[str], logger: logging.Logger, label: str) -> Path | None:
    explicit_path = resolve_explicit_or_path(explicit)
    if explicit_path is not None:
        return explicit_path

    which_path = shutil_which(names)
    if which_path is not None:
        return which_path

    for candidate in _candidate_paths(tools_root, names):
        if candidate.suffix.lower() == ".exe":
            logger.info("Found %s candidate: %s", label, candidate)
            return candidate.resolve()
        if looks_like_windows_executable(candidate):
            logger.info("Found %s candidate without .exe suffix: %s", label, candidate)
            return candidate.resolve()
        logger.info("Ignoring non-Windows %s candidate: %s", label, candidate)
    return None


def _resolve_pnnx(explicit: str, logger: logging.Logger) -> Path | None:
    explicit_path = resolve_explicit_or_path(explicit)
    if explicit_path is not None:
        return explicit_path

    which_path = shutil_which(["pnnx.exe", "pnnx"])
    if which_path is not None:
        return which_path

    try:
        import pnnx  # type: ignore

        candidate = Path(pnnx.__file__).resolve().parent / "pnnx.exe"
        if candidate.exists():
            logger.info("Found pnnx package binary: %s", candidate)
            return candidate
    except Exception:
        return None
    return None


def _build_inputshape_arg(input_shapes: dict[str, list[int]]) -> str:
    parts = []
    for shape in input_shapes.values():
        parts.append("[" + ",".join(str(value) for value in shape) + "]")
    return ",".join(parts)


def _build_inputshape_with_dtype_arg(input_shapes: dict[str, list[int]], input_dtypes: list[str]) -> str:
    parts = []
    for index, shape in enumerate(input_shapes.values()):
        dtype = input_dtypes[index] if index < len(input_dtypes) else "f32"
        parts.append("[" + ",".join(str(value) for value in shape) + "]" + dtype)
    return ",".join(parts)


def _convert_with_onnx2ncnn(
    onnx2ncnn_path: Path,
    ncnnoptimize_path: Path | None,
    export: OnnxExportArtifact,
    output_dir: Path,
    optimize_to_fp16: bool,
    logger: logging.Logger,
) -> NcnnArtifact:
    raw_param = output_dir / f"{export.ncnn_basename}.raw.param"
    raw_bin = output_dir / f"{export.ncnn_basename}.raw.bin"
    final_param = output_dir / f"{export.ncnn_basename}.param"
    final_bin = output_dir / f"{export.ncnn_basename}.bin"

    run_command(
        [str(onnx2ncnn_path), str(export.simplified_onnx), str(raw_param), str(raw_bin)],
        logger=logger,
    )
    if not raw_param.exists() or not raw_bin.exists():
        raise ConversionError(f"onnx2ncnn did not produce expected outputs for {export.name}.")

    if ncnnoptimize_path is not None:
        optimize_flag = "1" if optimize_to_fp16 else "0"
        run_command(
            [
                str(ncnnoptimize_path),
                str(raw_param),
                str(raw_bin),
                str(final_param),
                str(final_bin),
                optimize_flag,
            ],
            logger=logger,
        )
    else:
        logger.warning("ncnnoptimize was not found. Keeping raw NCNN outputs for %s.", export.name)
        copy_or_replace(raw_param, final_param)
        copy_or_replace(raw_bin, final_bin)

    return NcnnArtifact(
        name=export.ncnn_basename,
        converter="onnx2ncnn",
        input_onnx=str(export.simplified_onnx),
        raw_param=str(raw_param),
        raw_bin=str(raw_bin),
        final_param=str(final_param),
        final_bin=str(final_bin),
    )


def _convert_with_pnnx(
    pnnx_path: Path,
    ncnnoptimize_path: Path | None,
    export: OnnxExportArtifact,
    output_dir: Path,
    optimize_to_fp16: bool,
    logger: logging.Logger,
) -> NcnnArtifact:
    raw_param = output_dir / f"{export.ncnn_basename}.raw.param"
    raw_bin = output_dir / f"{export.ncnn_basename}.raw.bin"
    final_param = output_dir / f"{export.ncnn_basename}.param"
    final_bin = output_dir / f"{export.ncnn_basename}.bin"
    pnnx_param = output_dir / f"{export.ncnn_basename}.pnnx.param"
    pnnx_bin = output_dir / f"{export.ncnn_basename}.pnnx.bin"
    pnnx_py = output_dir / f"{export.ncnn_basename}_pnnx.py"
    pnnx_onnx = output_dir / f"{export.ncnn_basename}.pnnx.onnx"
    ncnn_py = output_dir / f"{export.ncnn_basename}_ncnn.py"

    source_path = export.torchscript_path if export.torchscript_path.exists() else export.simplified_onnx
    inputshape = (
        _build_inputshape_with_dtype_arg(export.input_shapes, export.input_dtypes)
        if source_path.suffix.lower() in (".pt", ".pth", ".ts")
        else _build_inputshape_arg(export.input_shapes)
    )
    run_command(
        [
            str(pnnx_path),
            str(source_path),
            f"inputshape={inputshape}",
            f"pnnxparam={pnnx_param}",
            f"pnnxbin={pnnx_bin}",
            f"pnnxpy={pnnx_py}",
            f"pnnxonnx={pnnx_onnx}",
            f"ncnnparam={raw_param}",
            f"ncnnbin={raw_bin}",
            f"ncnnpy={ncnn_py}",
            f"fp16={1 if optimize_to_fp16 else 0}",
        ],
        logger=logger,
        cwd=output_dir,
    )
    if not raw_param.exists() or not raw_bin.exists():
        raise ConversionError(f"pnnx did not produce expected outputs for {export.name}.")

    if ncnnoptimize_path is not None:
        optimize_flag = "1" if optimize_to_fp16 else "0"
        run_command(
            [
                str(ncnnoptimize_path),
                str(raw_param),
                str(raw_bin),
                str(final_param),
                str(final_bin),
                optimize_flag,
            ],
            logger=logger,
        )
    else:
        logger.warning("ncnnoptimize was not found. Keeping pnnx raw NCNN outputs for %s.", export.name)
        copy_or_replace(raw_param, final_param)
        copy_or_replace(raw_bin, final_bin)

    return NcnnArtifact(
        name=export.ncnn_basename,
        converter="pnnx",
        input_onnx=str(source_path),
        raw_param=str(raw_param),
        raw_bin=str(raw_bin),
        final_param=str(final_param),
        final_bin=str(final_bin),
    )


def convert_onnx_to_ncnn(
    exports: dict[str, OnnxExportArtifact],
    output_dir: Path,
    tools_root: Path,
    explicit_onnx2ncnn: str,
    explicit_ncnnoptimize: str,
    explicit_pnnx: str,
    optimize_to_fp16: bool,
    allow_pnnx_fallback: bool,
    logger: logging.Logger,
) -> dict[str, NcnnArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx2ncnn_path = _resolve_windows_tool(
        explicit=explicit_onnx2ncnn,
        tools_root=tools_root,
        names=["onnx2ncnn.exe", "onnx2ncnn"],
        logger=logger,
        label="onnx2ncnn",
    )
    ncnnoptimize_path = _resolve_windows_tool(
        explicit=explicit_ncnnoptimize,
        tools_root=tools_root,
        names=["ncnnoptimize.exe", "ncnnoptimize"],
        logger=logger,
        label="ncnnoptimize",
    )
    pnnx_path = _resolve_pnnx(explicit_pnnx, logger=logger)

    if onnx2ncnn_path is None:
        logger.warning("Windows onnx2ncnn was not found.")
    else:
        logger.info("Using onnx2ncnn: %s", onnx2ncnn_path)

    if ncnnoptimize_path is None:
        logger.warning("ncnnoptimize was not found.")
    else:
        logger.info("Using ncnnoptimize: %s", ncnnoptimize_path)

    if pnnx_path is not None:
        logger.info("Using pnnx fallback candidate: %s", pnnx_path)

    results: dict[str, NcnnArtifact] = {}
    for export in exports.values():
        logger.info("Converting %s -> NCNN", export.name)
        if onnx2ncnn_path is not None:
            try:
                results[export.ncnn_basename] = _convert_with_onnx2ncnn(
                    onnx2ncnn_path=onnx2ncnn_path,
                    ncnnoptimize_path=ncnnoptimize_path,
                    export=export,
                    output_dir=output_dir,
                    optimize_to_fp16=optimize_to_fp16,
                    logger=logger,
                )
                continue
            except Exception as error:
                logger.warning("onnx2ncnn conversion failed for %s: %s", export.name, error)
                if not allow_pnnx_fallback:
                    raise

        if not allow_pnnx_fallback:
            raise ConversionError(
                f"onnx2ncnn is unavailable for {export.name} and pnnx fallback is disabled."
            )
        if pnnx_path is None:
            raise ConversionError(
                f"Failed to convert {export.name}: onnx2ncnn is unavailable and pnnx fallback could not be found."
            )

        results[export.ncnn_basename] = _convert_with_pnnx(
            pnnx_path=pnnx_path,
            ncnnoptimize_path=ncnnoptimize_path,
            export=export,
            output_dir=output_dir,
            optimize_to_fp16=optimize_to_fp16,
            logger=logger,
        )

    return results
