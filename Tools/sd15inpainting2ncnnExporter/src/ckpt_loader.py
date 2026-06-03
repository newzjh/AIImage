from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ConversionError


EXPECTED_CORE_KEYS = (
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.out.2.weight",
    "first_stage_model.encoder.conv_in.weight",
    "first_stage_model.decoder.conv_out.weight",
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
)


@dataclass(frozen=True)
class CheckpointInspection:
    path: str
    format: str
    key_count: int
    unet_in_channels: int | None
    first_conv_shape: list[int] | None
    missing_core_keys: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "key_count": self.key_count,
            "unet_in_channels": self.unet_in_channels,
            "first_conv_shape": self.first_conv_shape,
            "missing_core_keys": self.missing_core_keys,
        }


def _load_raw_checkpoint(ckpt_path: Path) -> tuple[dict[str, Any], str]:
    suffix = ckpt_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(ckpt_path), device="cpu"), "safetensors"

    import torch

    return torch.load(str(ckpt_path), map_location="cpu"), "torch"


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    import torch

    if isinstance(payload, dict):
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
        for key in ("state_dict", "model", "module", "network", "net"):
            if key in payload:
                nested = _extract_state_dict(payload[key])
                if nested:
                    return nested
    raise ConversionError("Could not extract a Tensor state_dict from the checkpoint payload.")


def _resolve_unet_in_channels(state_dict: dict[str, Any]) -> tuple[int | None, list[int] | None]:
    tensor = state_dict.get("model.diffusion_model.input_blocks.0.0.weight")
    if tensor is None:
        return None, None
    shape = list(tensor.shape)
    if len(shape) < 2:
        return None, shape
    return int(shape[1]), shape


def inspect_checkpoint(ckpt_path: Path, logger: logging.Logger) -> CheckpointInspection:
    if not ckpt_path.exists():
        raise ConversionError(f"Checkpoint file does not exist: {ckpt_path}")

    logger.info("Loading checkpoint header/state_dict: %s", ckpt_path)
    raw_payload, checkpoint_format = _load_raw_checkpoint(ckpt_path)
    state_dict = _extract_state_dict(raw_payload)
    key_count = len(state_dict)
    logger.info("Loaded checkpoint format=%s keys=%d", checkpoint_format, key_count)

    missing_core_keys = [key for key in EXPECTED_CORE_KEYS if key not in state_dict]
    if missing_core_keys:
        logger.warning("Checkpoint is missing %d required keys.", len(missing_core_keys))
        for missing_key in missing_core_keys:
            logger.warning("Missing key: %s", missing_key)

    unet_in_channels, first_conv_shape = _resolve_unet_in_channels(state_dict)
    logger.info("Detected UNet input conv shape: %s", first_conv_shape)
    logger.info("Detected UNet input channels: %s", unet_in_channels)

    if missing_core_keys:
        raise ConversionError(
            "Checkpoint is missing core Stable Diffusion v1.5 inpainting keys. "
            f"Missing keys: {', '.join(missing_core_keys)}"
        )
    if unet_in_channels != 9:
        raise ConversionError(
            "UNet input channel mismatch. Expected 9 channels for SD1.5 inpainting "
            f"but found {unet_in_channels} from {ckpt_path.name}."
        )

    return CheckpointInspection(
        path=str(ckpt_path),
        format=checkpoint_format,
        key_count=key_count,
        unet_in_channels=unet_in_channels,
        first_conv_shape=first_conv_shape,
        missing_core_keys=missing_core_keys,
    )
