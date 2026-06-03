from __future__ import annotations

import logging
from pathlib import Path

from .utils import ConversionError


DEFAULT_DIFFUSERS_CONFIG_REPO = "runwayml/stable-diffusion-inpainting"


def convert_ckpt_to_diffusers(
    ckpt_path: Path,
    config_path: Path,
    output_dir: Path,
    requested_fp16: bool,
    device: str,
    local_files_only: bool,
    logger: logging.Logger,
) -> Path:
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if requested_fp16 and device == "cpu":
        logger.info("Requested --fp16 on CPU. Diffusers/ONNX export will stay in fp32; NCNN optimize will handle fp16 storage.")

    logger.info("Loading StableDiffusionInpaintPipeline.from_single_file from %s", ckpt_path)
    logger.info("Using original config: %s", config_path)

    pipeline = StableDiffusionInpaintPipeline.from_single_file(
        str(ckpt_path),
        config=DEFAULT_DIFFUSERS_CONFIG_REPO,
        original_config=str(config_path),
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)

    actual_in_channels = int(getattr(pipeline.unet.config, "in_channels", -1))
    logger.info("Diffusers pipeline loaded. pipe.unet.config.in_channels=%s", actual_in_channels)
    if actual_in_channels != 9:
        raise ConversionError(
            "Converted Diffusers UNet is not an inpainting UNet. "
            f"Expected in_channels=9 but got {actual_in_channels}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving diffusers pipeline to %s", output_dir)
    pipeline.save_pretrained(str(output_dir))
    return output_dir
