from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ConversionError, cleanup_file_and_sidecars, save_onnx_model


@dataclass(frozen=True)
class OnnxExportArtifact:
    name: str
    ncnn_basename: str
    raw_onnx: Path
    simplified_onnx: Path
    torchscript_path: Path
    input_shapes: dict[str, list[int]]
    input_dtypes: list[str]
    output_names: list[str]
    preferred_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ncnn_basename": self.ncnn_basename,
            "raw_onnx": str(self.raw_onnx),
            "simplified_onnx": str(self.simplified_onnx),
            "torchscript_path": str(self.torchscript_path),
            "input_shapes": self.input_shapes,
            "input_dtypes": self.input_dtypes,
            "output_names": self.output_names,
            "preferred_source": self.preferred_source,
        }


def _load_pipeline(diffusers_dir: Path, requested_fp16: bool, device: str, logger: logging.Logger):
    from diffusers import StableDiffusionInpaintPipeline

    if requested_fp16 and device == "cpu":
        logger.info("Keeping diffusers export pipeline in fp32 on CPU; fp16 is applied later during NCNN optimization.")
    logger.info("Loading diffusers pipeline from %s", diffusers_dir)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(str(diffusers_dir))
    pipeline = pipeline.to(device)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    return pipeline


def _make_wrappers(pipeline):
    import torch
    import torch.nn as nn

    class UNetWrapper(nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

    class VaeDecoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            return self.vae.decode(latent_sample, return_dict=False)[0]

    class VaeEncoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, sample):
            hidden = self.vae.encoder(sample)
            moments = self.vae.quant_conv(hidden)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            return mean, std

    class TextEncoderWrapper(nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, input_ids):
            return self.text_encoder(input_ids=input_ids.to(dtype=torch.int64), return_dict=False)[0]

    return {
        "unet": UNetWrapper(pipeline.unet),
        "vae_decoder": VaeDecoderWrapper(pipeline.vae),
        "vae_encoder": VaeEncoderWrapper(pipeline.vae),
        "text_encoder": TextEncoderWrapper(pipeline.text_encoder),
    }


def _torch_export(
    model,
    args: tuple[Any, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    logger: logging.Logger,
) -> None:
    import torch

    cleanup_file_and_sidecars(output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            external_data=True,
            dynamo=False,
        )
    logger.info("Exported raw ONNX: %s", output_path)


def _torchscript_export(
    model,
    args: tuple[Any, ...],
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with torch.no_grad():
        traced = torch.jit.trace(model, args, strict=False)
        traced.save(str(output_path))
    logger.info("Exported TorchScript: %s", output_path)
    return output_path


def _torch_dtype_to_pnnx_name(dtype) -> str:
    import torch

    mapping = {
        torch.float32: "f32",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.uint8: "u8",
        torch.int8: "i8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
    }
    return mapping.get(dtype, "f32")


def _infer_input_dtypes(args: tuple[Any, ...]) -> list[str]:
    dtypes: list[str] = []
    for arg in args:
        dtype = getattr(arg, "dtype", None)
        dtypes.append(_torch_dtype_to_pnnx_name(dtype))
    return dtypes


def _simplify_onnx(
    raw_onnx: Path,
    simplified_onnx: Path,
    input_shapes: dict[str, list[int]],
    logger: logging.Logger,
) -> Path:
    from onnxsim import simplify

    try:
        logger.info("Simplifying ONNX: %s", raw_onnx)
        model_simplified, check_ok = simplify(
            str(raw_onnx),
            check_n=0,
            overwrite_input_shapes=input_shapes,
            test_input_shapes=input_shapes,
            dynamic_input_shape=False,
        )
        if not check_ok:
            logger.warning("onnxsim returned check_ok=False for %s. Continuing with simplified graph anyway.", raw_onnx.name)
        save_onnx_model(model_simplified, simplified_onnx, external_data=True)
        logger.info("Saved simplified ONNX: %s", simplified_onnx)
        return simplified_onnx
    except Exception as error:
        logger.warning("onnxsim failed for %s: %s", raw_onnx.name, error)
        logger.warning("Falling back to raw ONNX for subsequent NCNN conversion.")
        cleanup_file_and_sidecars(simplified_onnx)
        return raw_onnx


def export_onnx_models(
    diffusers_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    opset: int,
    requested_fp16: bool,
    device: str,
    logger: logging.Logger,
) -> dict[str, OnnxExportArtifact]:
    import torch

    pipeline = _load_pipeline(diffusers_dir, requested_fp16=requested_fp16, device=device, logger=logger)
    wrappers = _make_wrappers(pipeline)

    latent_width = width // 8
    latent_height = height // 8
    if latent_width <= 0 or latent_height <= 0:
        raise ConversionError("Computed latent size is invalid. Check width/height arguments.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torchscript_dir = output_dir / "_torchscript"
    torchscript_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, OnnxExportArtifact] = {}

    export_specs = [
        {
            "name": "unet",
            "ncnn_basename": "unet",
            "model": wrappers["unet"],
            "args": (
                torch.randn(1, 9, latent_height, latent_width, device=device, dtype=torch.float32),
                torch.tensor([1.0], device=device, dtype=torch.float32),
                torch.randn(1, 77, 768, device=device, dtype=torch.float32),
            ),
            "input_names": ["sample", "timestep", "encoder_hidden_states"],
            "output_names": ["latent"],
            "dynamic_axes": {
                "sample": {0: "batch"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "latent": {0: "batch"},
            },
            "input_shapes": {
                "sample": [1, 9, latent_height, latent_width],
                "timestep": [1],
                "encoder_hidden_states": [1, 77, 768],
            },
        },
        {
            "name": "vae_decoder",
            "ncnn_basename": "vae",
            "model": wrappers["vae_decoder"],
            "args": (torch.randn(1, 4, latent_height, latent_width, device=device, dtype=torch.float32),),
            "input_names": ["latent_sample"],
            "output_names": ["sample"],
            "dynamic_axes": {
                "latent_sample": {0: "batch"},
                "sample": {0: "batch"},
            },
            "input_shapes": {
                "latent_sample": [1, 4, latent_height, latent_width],
            },
        },
        {
            "name": "vae_encoder",
            "ncnn_basename": "vae_encoder",
            "model": wrappers["vae_encoder"],
            "args": (torch.randn(1, 3, height, width, device=device, dtype=torch.float32),),
            "input_names": ["sample"],
            "output_names": ["out0", "out1"],
            "dynamic_axes": {
                "sample": {0: "batch"},
                "out0": {0: "batch"},
                "out1": {0: "batch"},
            },
            "input_shapes": {
                "sample": [1, 3, height, width],
            },
        },
        {
            "name": "text_encoder",
            "ncnn_basename": "text_encoder",
            "model": wrappers["text_encoder"],
            "args": (torch.randint(0, 49408, (1, 77), device=device, dtype=torch.int64),),
            "input_names": ["input_ids"],
            "output_names": ["last_hidden_state"],
            "dynamic_axes": {
                "input_ids": {0: "batch"},
                "last_hidden_state": {0: "batch"},
            },
            "input_shapes": {
                "input_ids": [1, 77],
            },
        },
    ]

    for spec in export_specs:
        name = spec["name"]
        raw_onnx = output_dir / f"{name}.onnx"
        simplified_onnx = output_dir / f"{name}.sim.onnx"
        torchscript_path = torchscript_dir / f"{name}.pt"

        logger.info("Exporting %s TorchScript", name)
        _torchscript_export(
            model=spec["model"],
            args=spec["args"],
            output_path=torchscript_path,
            logger=logger,
        )

        logger.info("Exporting %s raw ONNX", name)
        _torch_export(
            model=spec["model"],
            args=spec["args"],
            output_path=raw_onnx,
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            dynamic_axes=spec["dynamic_axes"],
            opset=opset,
            logger=logger,
        )

        selected_onnx = _simplify_onnx(
            raw_onnx=raw_onnx,
            simplified_onnx=simplified_onnx,
            input_shapes=spec["input_shapes"],
            logger=logger,
        )
        input_dtypes = _infer_input_dtypes(spec["args"])

        artifacts[name] = OnnxExportArtifact(
            name=name,
            ncnn_basename=spec["ncnn_basename"],
            raw_onnx=raw_onnx,
            simplified_onnx=selected_onnx,
            torchscript_path=torchscript_path,
            input_shapes=spec["input_shapes"],
            input_dtypes=input_dtypes,
            output_names=spec["output_names"],
            preferred_source="torchscript",
        )

    return artifacts
