from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class ExportError(RuntimeError):
    pass


def force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export a fixed-label-prompt Vista3D wrapper to NCNN via pnnx.")
    parser.add_argument(
        "--bundle-root",
        default=str(root / "bundle_cache" / "model-zoo-dev" / "models" / "vista3d"),
        help="Path to the Vista3D MONAI bundle root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "vista3d_ct_philips_heart"),
        help="Directory for ONNX, pnnx and NCNN outputs.",
    )
    parser.add_argument(
        "--label-prompt",
        type=int,
        default=115,
        help="Requested Vista3D label prompt.",
    )
    parser.add_argument(
        "--label-name",
        default="heart",
        help="Human-readable label name for manifests.",
    )
    parser.add_argument(
        "--case-tag",
        default="ct_philips_heart",
        help="Short output stem used for generated files.",
    )
    parser.add_argument(
        "--input-shape",
        default="1,1,128,128,128",
        help="Static NCDHW input shape for ONNX export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Vista3DFixedPromptExport")
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


def run_command(command: list[str], logger: logging.Logger, cwd: Path | None = None) -> None:
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
        raise ExportError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def run_command_allow_partial_outputs(
    command: list[str],
    logger: logging.Logger,
    expected_outputs: list[Path],
    cwd: Path | None = None,
) -> None:
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
    if process.returncode != 0 and not any(path.exists() for path in expected_outputs):
        raise ExportError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")
    if process.returncode != 0:
        logger.warning("Command returned non-zero exit code but expected outputs exist; continuing best-effort.")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_input_shape(shape_text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())
    if len(values) != 5:
        raise ExportError(f"Expected 5 input dimensions in NCDHW order, got: {shape_text}")
    return values


def expand_label_prompt(requested_label: int, config: dict[str, Any]) -> list[int]:
    subclass = config.get("subclass") or {}
    values = subclass.get(str(int(requested_label)))
    if not values:
        return [int(requested_label)]
    return [int(value) for value in values]


def ensure_bundle_pythonpath(bundle_root: Path) -> None:
    bundle_path = str(bundle_root.resolve())
    if bundle_path not in sys.path:
        sys.path.insert(0, bundle_path)


def find_pnnx() -> Path:
    candidates = [
        Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "pnnx" / "pnnx.exe",
        Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "pnnx" / "pnnx",
        Path(sys.executable).with_name("pnnx.exe"),
        Path(sys.executable).with_name("pnnx"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    found = shutil.which("pnnx.exe") or shutil.which("pnnx")
    if found:
        return Path(found)

    raise ExportError("pnnx executable was not found in the active environment.")


def build_network(bundle_root: Path, config: dict[str, Any], logger: logging.Logger):
    import torch
    from monai.bundle import ConfigParser
    from monai.networks import copy_model_state

    ensure_bundle_pythonpath(bundle_root)
    parser = ConfigParser(config)
    network = parser.get_parsed_content("network_def")
    checkpoint_path = bundle_root / "models" / "model.pt"
    if not checkpoint_path.exists():
        raise ExportError(f"Vista3D checkpoint not found: {checkpoint_path}")

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ExportError("Vista3D checkpoint format is unexpected; expected a state_dict-compatible mapping.")

    state_dict = payload["model"] if "model" in payload and isinstance(payload["model"], dict) else payload

    copy_model_state(dst=network, src=state_dict)
    network.eval()
    logger.info("Loaded Vista3D network with checkpoint: %s", checkpoint_path)
    return network


def export_fixed_prompt_model(
    bundle_root: Path,
    config: dict[str, Any],
    requested_label: int,
    label_name: str,
    input_shape: tuple[int, ...],
    output_dir: Path,
    case_tag: str,
    opset: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    import torch

    transformed_labels = expand_label_prompt(requested_label, config)
    network = build_network(bundle_root, config, logger)

    class FixedPromptVista3D(torch.nn.Module):
        def __init__(self, wrapped_network: torch.nn.Module, labels: list[int]) -> None:
            super().__init__()
            self.wrapped_network = wrapped_network
            self.register_buffer(
                "class_vector",
                torch.tensor([[int(label)] for label in labels], dtype=torch.long),
                persistent=True,
            )

        def forward(self, input_image: torch.Tensor) -> torch.Tensor:
            return self.wrapped_network(
                input_image,
                class_vector=self.class_vector.to(device=input_image.device),
                point_coords=None,
                point_labels=None,
                transpose=True,
            )

    wrapper = FixedPromptVista3D(network, transformed_labels).cpu().eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    stem = f"vista3d_fixed_{case_tag}"
    raw_onnx = output_dir / f"{stem}.raw.onnx"
    sim_onnx = output_dir / f"{stem}.sim.onnx"
    final_param = output_dir / f"{stem}.param"
    final_bin = output_dir / f"{stem}.bin"
    pnnx_param = output_dir / f"{stem}.pnnx.param"
    pnnx_bin = output_dir / f"{stem}.pnnx.bin"
    pnnx_py = output_dir / f"{stem}_pnnx.py"
    pnnx_onnx = output_dir / f"{stem}.pnnx.onnx"
    ncnn_py = output_dir / f"{stem}_ncnn.py"

    logger.info("Export ONNX | input_shape=%s | requested_label=%s | transformed_labels=%s", input_shape, requested_label, transformed_labels)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(raw_onnx),
            opset_version=opset,
            input_names=["in0"],
            output_names=["out0"],
            do_constant_folding=True,
        )

    try:
        run_command([sys.executable, "-m", "onnxsim", str(raw_onnx), str(sim_onnx)], logger=logger)
        export_input = sim_onnx
    except Exception as error:
        logger.warning("onnxsim failed, falling back to raw ONNX: %s", error)
        export_input = raw_onnx

    pnnx_path = find_pnnx()
    shape_text = ",".join(str(value) for value in input_shape)
    run_command_allow_partial_outputs(
        [
            str(pnnx_path),
            str(export_input),
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
        expected_outputs=[final_param, final_bin, pnnx_param, pnnx_bin, pnnx_onnx, ncnn_py],
        cwd=output_dir,
    )

    if not final_param.exists() or not final_bin.exists():
        raise ExportError("pnnx did not create usable NCNN .param/.bin outputs.")

    payload = {
        "bundle_name": "vista3d_fixed_prompt",
        "source_bundle_root": str(bundle_root),
        "requested_label_prompt": [int(requested_label)],
        "transformed_label_prompt": transformed_labels,
        "label_name": label_name,
        "case_tag": case_tag,
        "task_mode": "binary_label_prompt" if len(transformed_labels) == 1 else "multiclass",
        "input_shape": list(input_shape),
        "input_blob_name": "in0",
        "output_blob_name": "out0",
        "raw_onnx": str(raw_onnx),
        "sim_onnx": str(sim_onnx if sim_onnx.exists() else raw_onnx),
        "param_path": str(final_param),
        "bin_path": str(final_bin),
        "pnnx_param_path": str(pnnx_param),
        "pnnx_bin_path": str(pnnx_bin),
        "pnnx_onnx": str(pnnx_onnx),
        "ncnn_python": str(ncnn_py),
    }
    return payload


def main() -> int:
    force_utf8_stdio()
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "conversion.log")

    try:
        config = load_json(bundle_root / "configs" / "inference.json")
        metadata = load_json(bundle_root / "configs" / "metadata.json")
        input_shape = parse_input_shape(args.input_shape)

        payload = export_fixed_prompt_model(
            bundle_root=bundle_root,
            config=config,
            requested_label=args.label_prompt,
            label_name=args.label_name,
            input_shape=input_shape,
            output_dir=output_dir,
            case_tag=args.case_tag,
            opset=args.opset,
            logger=logger,
        )
        payload["metadata_name"] = metadata.get("name")
        payload["metadata_task"] = metadata.get("task")

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Export complete | manifest=%s", manifest_path)
        print(str(manifest_path))
        return 0
    except Exception as error:
        logger.exception("Vista3D export failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
