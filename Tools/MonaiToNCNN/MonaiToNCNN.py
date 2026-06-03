from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


FAILED_ONNX_OPS = ("Gather", "Expand", "Shape", "Resize")
MODEL_CANDIDATES = ("model.ts", "model.pt", "model.pth")


class ConversionError(RuntimeError):
    pass


def force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download a MONAI bundle and convert it to NCNN.")
    parser.add_argument("bundle_name", help="MONAI bundle name, for example: brats_mri_segmentation")
    parser.add_argument(
        "--bundle-dir",
        default=str(base_dir / "bundle_cache"),
        help="Directory that stores downloaded MONAI bundles.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir / "outputs"),
        help="Directory for ONNX, NCNN, logs and manifest outputs.",
    )
    parser.add_argument(
        "--source",
        default="monaihosting",
        help="Source passed to `python -m monai.bundle download`.",
    )
    parser.add_argument(
        "--input-shape",
        default="",
        help="Explicit input shape like `1,4,224,224,144`. If omitted, the script tries to infer it from configs.",
    )
    parser.add_argument(
        "--prefer",
        choices=("auto", "torchscript", "weights"),
        default="auto",
        help="Prefer TorchScript or PyTorch weights when both are present.",
    )
    parser.add_argument(
        "--onnx2ncnn",
        default="",
        help="Optional explicit path to onnx2ncnn(.exe).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the bundle again even if it already exists locally.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version used during export.",
    )
    parser.add_argument(
        "--no-pnnx-fallback",
        action="store_true",
        help="Fail instead of falling back to pnnx when onnx2ncnn is unavailable or fails.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("MonaiToNCNN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_versions(logger: logging.Logger) -> None:
    import monai
    import onnx
    import onnxsim
    import pnnx
    import torch

    logger.info("Python executable: %s", sys.executable)
    logger.info("monai=%s torch=%s onnx=%s onnxsim=%s pnnx=%s", monai.__version__, torch.__version__, onnx.__version__, onnxsim.__version__, pnnx.__version__)


def run_command(command: list[str], logger: logging.Logger, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    logger.info("RUN: %s", " ".join(command))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.stdout:
        logger.info(proc.stdout.rstrip())
    if proc.stderr:
        logger.info(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise ConversionError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")
    return proc


def download_bundle(bundle_name: str, bundle_dir: Path, source: str, force_download: bool, logger: logging.Logger) -> Path:
    bundle_root = bundle_dir / bundle_name
    if bundle_root.exists() and not force_download:
        logger.info("Bundle already exists, skip download: %s", bundle_root)
        return bundle_root
    bundle_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "monai.bundle",
            "download",
            "--name",
            bundle_name,
            "--bundle_dir",
            str(bundle_dir),
            "--source",
            source,
        ],
        logger=logger,
    )
    if not bundle_root.exists():
        raise ConversionError(f"Bundle download finished but folder was not found: {bundle_root}")
    return bundle_root


def find_model_artifacts(bundle_root: Path) -> dict[str, list[Path]]:
    models: dict[str, list[Path]] = {name: [] for name in MODEL_CANDIDATES}
    for path in bundle_root.rglob("*"):
        if path.is_file() and path.name in models:
            models[path.name].append(path)
    return models


def classify_artifact(path: Path) -> str:
    import torch

    if path.suffix.lower() == ".ts":
        return "torchscript"
    try:
        torch.jit.load(str(path), map_location="cpu")
        return "torchscript"
    except Exception:
        return "weights"


def choose_artifacts(artifacts: dict[str, list[Path]], prefer: str, logger: logging.Logger) -> tuple[Path | None, Path | None]:
    torchscript: Path | None = None
    weights: Path | None = None

    ordered: list[Path] = []
    for name in MODEL_CANDIDATES:
        ordered.extend(sorted(artifacts.get(name, [])))

    for path in ordered:
        kind = classify_artifact(path)
        logger.info("Detected model artifact: %s (%s)", path, kind)
        if kind == "torchscript" and torchscript is None:
            torchscript = path
        if kind == "weights" and weights is None:
            weights = path

    if prefer == "torchscript" and torchscript is None:
        raise ConversionError("Requested TorchScript preference, but no TorchScript artifact was found.")
    if prefer == "weights" and weights is None:
        raise ConversionError("Requested weights preference, but no PyTorch weights artifact was found.")
    if torchscript is None and weights is None:
        raise ConversionError("No model.ts / model.pt / model.pth artifact was found in the bundle.")
    return torchscript, weights


def load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_network_config(bundle_root: Path, logger: logging.Logger) -> tuple[Path, dict[str, Any]]:
    config_dir = bundle_root / "configs"
    if not config_dir.exists():
        raise ConversionError(f"Config directory not found: {config_dir}")

    preferred = [
        config_dir / "train.json",
        config_dir / "inference.json",
        config_dir / "evaluate.json",
        config_dir / "inference_trt.json",
    ]

    candidates = [path for path in preferred if path.exists()]
    candidates.extend(path for path in sorted(config_dir.glob("*.json")) if path not in candidates)

    for path in candidates:
        config = load_json(path)
        if "network_def" in config:
            logger.info("Using config for network reconstruction: %s", path)
            return path, config
    raise ConversionError(f"No config containing `network_def` was found under: {config_dir}")


def find_first_roi_size(node: Any) -> list[int] | None:
    if isinstance(node, dict):
        if "roi_size" in node and isinstance(node["roi_size"], list):
            return [int(v) for v in node["roi_size"]]
        for value in node.values():
            found = find_first_roi_size(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_first_roi_size(value)
            if found is not None:
                return found
    return None


def parse_input_shape(shape_text: str) -> tuple[int, ...]:
    dims = [int(part.strip()) for part in shape_text.split(",") if part.strip()]
    if len(dims) < 3:
        raise ConversionError(f"Invalid input shape: {shape_text}")
    return tuple(dims)


def infer_input_shape(config: dict[str, Any], logger: logging.Logger) -> tuple[int, ...]:
    network_def = config.get("network_def", {})
    in_channels = int(network_def.get("in_channels", 1))
    spatial_dims = int(network_def.get("spatial_dims", 3))
    roi_size = find_first_roi_size(config)

    if roi_size is None:
        roi_size = [224, 224] if spatial_dims == 2 else [64, 64, 64]
        logger.warning("Could not infer roi_size from configs, fallback to %s", roi_size)

    input_shape = tuple([1, in_channels, *roi_size])
    logger.info("Inferred input shape from configs: %s", input_shape)
    return input_shape


def build_network_from_config(config: dict[str, Any], logger: logging.Logger):
    from monai.bundle import ConfigParser

    parser = ConfigParser(config)
    net = parser.get_parsed_content("network_def")
    logger.info("Instantiated network from config: %s", type(net).__name__)
    return net


def extract_state_dict(payload: Any) -> dict[str, Any]:
    import torch

    if isinstance(payload, dict):
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
        for key in ("model", "network", "state_dict", "net", "module"):
            if key in payload:
                nested = extract_state_dict(payload[key])
                if nested:
                    return nested
    raise ConversionError("Could not extract a PyTorch state_dict from the checkpoint.")


def export_onnx_from_torchscript(model_path: Path, output_path: Path, input_shape: tuple[int, ...], opset: int, logger: logging.Logger) -> None:
    import torch

    logger.info("Trying TorchScript -> ONNX export: %s", model_path)
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )


def export_onnx_from_weights(
    weight_path: Path,
    config_path: Path,
    config: dict[str, Any],
    output_path: Path,
    input_shape: tuple[int, ...],
    opset: int,
    logger: logging.Logger,
) -> None:
    import torch
    from monai.networks import copy_model_state

    logger.info("Reconstruct network from config: %s", config_path)
    net = build_network_from_config(config, logger)
    payload = torch.load(str(weight_path), map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(payload)
    copy_model_state(dst=net, src=state_dict)
    net.eval()
    dummy = torch.randn(*input_shape)

    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )


def collect_onnx_op_stats(onnx_path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(onnx_path))
    counts = Counter(node.op_type for node in model.graph.node)
    return dict(sorted(counts.items()))


def log_failed_op_scan(op_counts: dict[str, int], logger: logging.Logger, stage: str) -> None:
    logger.info("%s ONNX op counts for watched layers: %s", stage, {name: op_counts.get(name, 0) for name in FAILED_ONNX_OPS})
    watched = {name: op_counts.get(name, 0) for name in FAILED_ONNX_OPS if op_counts.get(name, 0)}
    if watched:
        logger.warning("Potentially fragile ONNX ops detected at %s: %s", stage, watched)


def simplify_onnx(input_path: Path, output_path: Path, logger: logging.Logger) -> None:
    run_command([sys.executable, "-m", "onnxsim", str(input_path), str(output_path)], logger=logger)
    if not output_path.exists():
        raise ConversionError(f"onnxsim finished without producing: {output_path}")


def find_onnx2ncnn(explicit: str, tools_dir: Path, logger: logging.Logger) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise ConversionError(f"Explicit onnx2ncnn path does not exist: {path}")
        return path

    for name in ("onnx2ncnn.exe", "onnx2ncnn"):
        found = shutil.which(name)
        if found:
            return Path(found)

    for path in tools_dir.rglob("onnx2ncnn.exe"):
        logger.info("Found local onnx2ncnn candidate: %s", path)
        return path

    return None


def find_pnnx(logger: logging.Logger) -> Path:
    candidates = [
        Path(sys.executable).with_name("pnnx.exe"),
        Path(sys.executable).with_name("pnnx"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = shutil.which("pnnx.exe") or shutil.which("pnnx")
    if found:
        return Path(found)
    raise ConversionError("pnnx was not found in the active environment.")


def convert_with_onnx2ncnn(onnx2ncnn_path: Path, input_path: Path, param_path: Path, bin_path: Path, logger: logging.Logger) -> str:
    run_command([str(onnx2ncnn_path), str(input_path), str(param_path), str(bin_path)], logger=logger)
    if not param_path.exists() or not bin_path.exists():
        raise ConversionError("onnx2ncnn did not create the expected .param/.bin outputs.")
    return "onnx2ncnn"


def convert_with_pnnx(
    pnnx_path: Path,
    input_path: Path,
    input_shape: tuple[int, ...],
    output_dir: Path,
    final_param: Path,
    final_bin: Path,
    logger: logging.Logger,
) -> str:
    stem = input_path.stem
    pnnx_param = output_dir / f"{stem}.pnnx.param"
    pnnx_bin = output_dir / f"{stem}.pnnx.bin"
    pnnx_py = output_dir / f"{stem}_pnnx.py"
    pnnx_onnx = output_dir / f"{stem}.pnnx.onnx"
    ncnn_py = output_dir / f"{stem}_ncnn.py"
    shape_text = ",".join(str(v) for v in input_shape)

    run_command(
        [
            str(pnnx_path),
            str(input_path),
            f"inputshape=[{shape_text}]",
            f"pnnxparam={pnnx_param}",
            f"pnnxbin={pnnx_bin}",
            f"pnnxpy={pnnx_py}",
            f"pnnxonnx={pnnx_onnx}",
            f"ncnnparam={final_param}",
            f"ncnnbin={final_bin}",
            f"ncnnpy={ncnn_py}",
        ],
        logger=logger,
        cwd=output_dir,
    )
    if not final_param.exists() or not final_bin.exists():
        raise ConversionError("pnnx fallback did not create the expected .param/.bin outputs.")
    return "pnnx"


def write_manifest(
    manifest_path: Path,
    bundle_name: str,
    bundle_root: Path,
    input_shape: tuple[int, ...],
    export_branch: str,
    converter: str,
    raw_onnx: Path,
    sim_onnx: Path,
    param_path: Path,
    bin_path: Path,
    raw_ops: dict[str, int],
    sim_ops: dict[str, int],
) -> None:
    payload = {
        "bundle_name": bundle_name,
        "bundle_root": str(bundle_root),
        "input_shape": list(input_shape),
        "export_branch": export_branch,
        "converter": converter,
        "raw_onnx": str(raw_onnx),
        "sim_onnx": str(sim_onnx),
        "param_path": str(param_path),
        "bin_path": str(bin_path),
        "raw_onnx_ops": raw_ops,
        "simplified_onnx_ops": sim_ops,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    force_utf8_stdio()
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    bundle_dir = Path(args.bundle_dir).resolve()
    run_output_dir = Path(args.output_dir).resolve() / args.bundle_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_output_dir / "conversion.log"
    logger = setup_logging(log_path)

    try:
        log_versions(logger)

        bundle_root = download_bundle(
            bundle_name=args.bundle_name,
            bundle_dir=bundle_dir,
            source=args.source,
            force_download=args.force_download,
            logger=logger,
        )

        artifacts = find_model_artifacts(bundle_root)
        torchscript_path, weight_path = choose_artifacts(artifacts, args.prefer, logger)

        config_path, config = find_network_config(bundle_root, logger)
        input_shape = parse_input_shape(args.input_shape) if args.input_shape else infer_input_shape(config, logger)

        raw_onnx = run_output_dir / f"{args.bundle_name}.raw.onnx"
        sim_onnx = run_output_dir / f"{args.bundle_name}.sim.onnx"
        final_param = run_output_dir / f"{args.bundle_name}.param"
        final_bin = run_output_dir / f"{args.bundle_name}.bin"
        manifest_path = run_output_dir / "manifest.json"

        export_branch = ""
        if args.prefer != "weights" and torchscript_path is not None:
            try:
                export_onnx_from_torchscript(
                    model_path=torchscript_path,
                    output_path=raw_onnx,
                    input_shape=input_shape,
                    opset=args.opset,
                    logger=logger,
                )
                export_branch = "torchscript"
            except Exception as torchscript_error:
                logger.exception("TorchScript -> ONNX export failed: %s", torchscript_error)
                if weight_path is None:
                    raise
                logger.warning("Falling back to PyTorch weights + configs export path.")

        if not raw_onnx.exists():
            if weight_path is None:
                raise ConversionError("No PyTorch weights artifact is available for fallback export.")
            export_onnx_from_weights(
                weight_path=weight_path,
                config_path=config_path,
                config=config,
                output_path=raw_onnx,
                input_shape=input_shape,
                opset=args.opset,
                logger=logger,
            )
            export_branch = "weights"

        raw_ops = collect_onnx_op_stats(raw_onnx)
        log_failed_op_scan(raw_ops, logger, "Raw")

        simplify_onnx(raw_onnx, sim_onnx, logger)
        sim_ops = collect_onnx_op_stats(sim_onnx)
        log_failed_op_scan(sim_ops, logger, "Simplified")

        converter = ""
        onnx2ncnn_path = find_onnx2ncnn(args.onnx2ncnn, tools_dir, logger)
        if onnx2ncnn_path is not None:
            try:
                converter = convert_with_onnx2ncnn(onnx2ncnn_path, sim_onnx, final_param, final_bin, logger)
            except Exception as onnx2ncnn_error:
                logger.exception("onnx2ncnn conversion failed: %s", onnx2ncnn_error)
                if args.no_pnnx_fallback:
                    raise
                logger.warning("Falling back to pnnx because onnx2ncnn failed.")
        else:
            logger.warning("onnx2ncnn was not found on this machine.")

        if not final_param.exists() or not final_bin.exists():
            if args.no_pnnx_fallback:
                raise ConversionError("onnx2ncnn output is missing and pnnx fallback is disabled.")
            pnnx_path = find_pnnx(logger)
            converter = convert_with_pnnx(
                pnnx_path=pnnx_path,
                input_path=sim_onnx,
                input_shape=input_shape,
                output_dir=run_output_dir,
                final_param=final_param,
                final_bin=final_bin,
                logger=logger,
            )

        write_manifest(
            manifest_path=manifest_path,
            bundle_name=args.bundle_name,
            bundle_root=bundle_root,
            input_shape=input_shape,
            export_branch=export_branch,
            converter=converter,
            raw_onnx=raw_onnx,
            sim_onnx=sim_onnx,
            param_path=final_param,
            bin_path=final_bin,
            raw_ops=raw_ops,
            sim_ops=sim_ops,
        )

        logger.info("Conversion finished successfully.")
        logger.info("Bundle root: %s", bundle_root)
        logger.info("Export branch: %s", export_branch)
        logger.info("Converter: %s", converter)
        logger.info("Output param: %s", final_param)
        logger.info("Output bin: %s", final_bin)
        logger.info("Manifest: %s", manifest_path)
        logger.info("Log file: %s", log_path)
        return 0
    except Exception as error:
        logger.exception("Conversion failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
