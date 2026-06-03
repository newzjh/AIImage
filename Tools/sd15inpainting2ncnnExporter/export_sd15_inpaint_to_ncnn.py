from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ckpt_loader import inspect_checkpoint
from src.diffusers_converter import convert_ckpt_to_diffusers
from src.ncnn_convert import convert_onnx_to_ncnn
from src.onnx_export import export_onnx_models
from src.utils import (
    ConversionError,
    build_output_dirs,
    force_utf8_stdio,
    log_versions,
    save_manifest,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export Stable Diffusion v1.5 inpainting checkpoint to NCNN param/bin files."
    )
    parser.add_argument("--ckpt", required=True, help="Path to sd-v1-5-inpainting.ckpt or .safetensors.")
    parser.add_argument(
        "--output",
        default=str(base_dir / "output"),
        help="Root output directory. Subfolders diffusers/, onnx/, ncnn/ will be created here.",
    )
    parser.add_argument("--width", type=int, default=512, help="Export image width. Must be divisible by 8.")
    parser.add_argument("--height", type=int, default=512, help="Export image height. Must be divisible by 8.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--fp16", action="store_true", help="Optimize final NCNN weights to fp16 storage.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used while loading/exporting diffusers modules.",
    )
    parser.add_argument(
        "--onnx2ncnn",
        default="",
        help="Optional explicit path to onnx2ncnn(.exe). If omitted the script tries to auto-detect it.",
    )
    parser.add_argument(
        "--ncnnoptimize",
        default="",
        help="Optional explicit path to ncnnoptimize(.exe). If omitted the script tries to auto-detect it.",
    )
    parser.add_argument(
        "--pnnx",
        default="",
        help="Optional explicit path to pnnx(.exe). Used only as a fallback when onnx2ncnn is unavailable or fails.",
    )
    parser.add_argument(
        "--no-pnnx-fallback",
        action="store_true",
        help="Fail instead of falling back to pnnx when onnx2ncnn is unavailable or fails.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not allow Hugging Face downloads while converting ckpt to diffusers.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ConversionError("Width and height must be positive integers.")
    if (args.width % 8) != 0 or (args.height % 8) != 0:
        raise ConversionError("Width and height must both be divisible by 8.")
    if args.opset < 17:
        raise ConversionError("Use opset 17 or newer for this exporter.")
    if args.device == "cuda":
        try:
            import torch
        except Exception as error:
            raise ConversionError("CUDA export was requested but torch is not installed yet.") from error
        if not torch.cuda.is_available():
            raise ConversionError("CUDA export was requested but torch.cuda.is_available() is false.")


def main() -> int:
    force_utf8_stdio()
    args = parse_args()
    logger = None

    try:
        validate_args(args)
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "configs" / "sd15_inpaint.yaml"
        if not config_path.exists():
            raise ConversionError(f"Inpainting config file is missing: {config_path}")

        output_dirs = build_output_dirs(Path(args.output).resolve())
        logger = setup_logging(output_dirs.root / "export.log")
        log_versions(logger)
        logger.info("CLI args: %s", vars(args))

        logger.info("[1/4] Inspect checkpoint")
        inspection = inspect_checkpoint(Path(args.ckpt).resolve(), logger)
        logger.info("Checkpoint inspection summary: %s", inspection.to_dict())

        logger.info("[2/4] Convert checkpoint to Diffusers")
        diffusers_dir = convert_ckpt_to_diffusers(
            ckpt_path=Path(args.ckpt).resolve(),
            config_path=config_path,
            output_dir=output_dirs.diffusers,
            requested_fp16=args.fp16,
            device=args.device,
            local_files_only=args.local_files_only,
            logger=logger,
        )

        logger.info("[3/4] Export ONNX models")
        onnx_exports = export_onnx_models(
            diffusers_dir=diffusers_dir,
            output_dir=output_dirs.onnx,
            width=args.width,
            height=args.height,
            opset=args.opset,
            requested_fp16=args.fp16,
            device=args.device,
            logger=logger,
        )

        logger.info("[4/4] Convert ONNX to NCNN")
        ncnn_outputs = convert_onnx_to_ncnn(
            exports=onnx_exports,
            output_dir=output_dirs.ncnn,
            tools_root=script_dir,
            explicit_onnx2ncnn=args.onnx2ncnn,
            explicit_ncnnoptimize=args.ncnnoptimize,
            explicit_pnnx=args.pnnx,
            optimize_to_fp16=args.fp16,
            allow_pnnx_fallback=not args.no_pnnx_fallback,
            logger=logger,
        )

        manifest = {
            "ckpt": str(Path(args.ckpt).resolve()),
            "config": str(config_path),
            "output_root": str(output_dirs.root),
            "width": args.width,
            "height": args.height,
            "fp16": bool(args.fp16),
            "device": args.device,
            "inspection": inspection.to_dict(),
            "diffusers_dir": str(diffusers_dir),
            "onnx_exports": {name: export.to_dict() for name, export in onnx_exports.items()},
            "ncnn_outputs": {name: artifact.to_dict() for name, artifact in ncnn_outputs.items()},
        }
        save_manifest(output_dirs.root / "manifest.json", manifest)

        logger.info("Export finished successfully.")
        logger.info("NCNN output folder: %s", output_dirs.ncnn)
        logger.info("Required outputs: %s", ", ".join(sorted(["unet", "vae", "text_encoder"])))
        logger.info("Additional outputs: %s", ", ".join(sorted(set(ncnn_outputs) - {"unet", "vae", "text_encoder"})))
        return 0
    except Exception as error:
        if logger is not None:
            logger.exception("Export failed: %s", error)
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
