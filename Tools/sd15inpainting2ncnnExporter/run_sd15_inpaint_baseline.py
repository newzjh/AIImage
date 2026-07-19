from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import torch
from PIL import Image

from src.diffusers_converter import DEFAULT_DIFFUSERS_CONFIG_REPO
from src.utils import ConversionError, force_utf8_stdio, log_versions, setup_logging


DEFAULT_POSITIVE_PROMPT = (
    "best quality, realistic photo, empty indoor background, clean wall, shelf, table, furniture, "
    "coherent texture, seamless fill, background only, no people, no person, no human"
)
DEFAULT_NEGATIVE_PROMPT = (
    "person, people, human, man, woman, child, face, portrait, head, body, skin, hands, arms, legs, "
    "crowd, group photo, selfie, mannequin, statue, reflection, silhouette, duplicate, blurry, deformed, "
    "extra limbs, artifacts, text, watermark"
)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run an external Stable Diffusion 1.5 inpainting baseline and dump first-step UNet tensors."
    )
    parser.add_argument("--input-dir", default="", help="Optional directory containing source/mask files.")
    parser.add_argument("--source", default="", help="Optional explicit source image path.")
    parser.add_argument("--mask", default="", help="Optional explicit mask image path.")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    parser.add_argument(
        "--diffusers-dir",
        default=str(base_dir / "output" / "diffusers"),
        help="Diffusers pipeline directory. Preferred over checkpoint when it exists.",
    )
    parser.add_argument(
        "--ckpt",
        default=str(base_dir.parent.parent / "ref" / "sd1.5inpainting" / "sd-v1-5-inpainting.ckpt"),
        help="Fallback checkpoint path when diffusers-dir is missing.",
    )
    parser.add_argument("--use-ckpt", action="store_true", help="Force loading the original checkpoint instead of output/diffusers.")
    parser.add_argument("--width", type=int, default=512, help="Working width.")
    parser.add_argument("--height", type=int, default=512, help="Working height.")
    parser.add_argument("--steps", type=int, default=12, help="Inference steps.")
    parser.add_argument("--seed", type=int, default=123456, help="Random seed.")
    parser.add_argument("--strength", type=float, default=1.0, help="Img2img/inpaint strength.")
    parser.add_argument("--guidance-scale", type=float, default=10.0, help="CFG guidance scale.")
    parser.add_argument("--positive-prompt", default=DEFAULT_POSITIVE_PROMPT, help="Positive prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--black-mask-means-inpaint", action="store_true", help="Invert mask semantics before thresholding.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Execution device.")
    parser.add_argument("--local-files-only", action="store_true", help="Disallow Hugging Face downloads when loading from checkpoint.")
    parser.add_argument("--save-step-latents", action="store_true", help="Also dump epsilon and next-latent tensors for every step.")
    return parser.parse_args()


def resolve_existing_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    return path if path.exists() else None


def resolve_input_file(input_dir: Path | None, explicit: str, candidates: list[str], label: str) -> Path:
    explicit_path = resolve_existing_path(explicit)
    if explicit_path is not None:
        return explicit_path

    if input_dir is not None:
        for name in candidates:
            candidate = (input_dir / name).resolve()
            if candidate.exists():
                return candidate

    raise ConversionError(f"Could not resolve {label} file. Checked explicit path and candidates: {candidates}")


def create_default_output_dir() -> Path:
    root = Path(gettempdir()) / "YanQi" / "AIImage"
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("AIImage_SD15InpaintPythonBaseline_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def pil_resample(mode: str):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, mode)
    return getattr(Image, mode)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_float_bin(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=np.float32).tofile(path)


def write_float_stats(path: Path, array: np.ndarray) -> None:
    data = np.asarray(array, dtype=np.float32).reshape(-1)
    finite = data[np.isfinite(data)]
    lines = [f"count={data.size}"]
    if finite.size == 0:
        lines.extend(["finite=0", "nan=0", "inf=0", "min=n/a", "max=n/a", "mean=0", "mean_abs=0"])
    else:
        lines.extend(
            [
                f"finite={finite.size}",
                f"nan={np.isnan(data).sum()}",
                f"inf={np.isinf(data).sum()}",
                f"min={float(finite.min()):.9g}",
                f"max={float(finite.max()):.9g}",
                f"mean={float(finite.mean()):.9g}",
                f"mean_abs={float(np.abs(finite).mean()):.9g}",
            ]
        )
    write_text(path, "\n".join(lines) + "\n")


def tensor_to_numpy_chw(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)


def save_tensor_chw(path_stem: Path, tensor) -> None:
    array = tensor_to_numpy_chw(tensor)
    write_float_bin(path_stem.with_suffix(".bin"), array.reshape(-1))
    write_float_stats(path_stem.with_name(path_stem.name + "_stats.txt"), array)


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_and_prepare_source_mask(source_path: Path, mask_path: Path, width: int, height: int, black_mask_means_inpaint: bool):
    source = Image.open(source_path).convert("RGBA")
    mask = Image.open(mask_path).convert("RGBA")
    source_original_size = source.size

    source_512 = source.resize((width, height), pil_resample("BILINEAR"))
    mask_input_512 = mask.resize((width, height), pil_resample("BILINEAR"))

    source_np = np.asarray(source_512, dtype=np.uint8)
    mask_np = np.asarray(mask_input_512, dtype=np.uint8)
    luminance = mask_np[..., :3].mean(axis=2) / 255.0
    alpha = mask_np[..., 3] / 255.0
    weight = np.clip(luminance * alpha, 0.0, 1.0)
    if black_mask_means_inpaint:
        weight = 1.0 - weight
    weight = np.where(weight >= 0.5, 1.0, 0.0).astype(np.float32)

    mask_binary = np.clip(np.round(weight * 255.0), 0, 255).astype(np.uint8)
    mask_512 = Image.fromarray(mask_binary, mode="L")

    inv = 1.0 - weight[..., None]
    masked_base = 127.5 * weight[..., None]
    masked_rgb = np.clip(np.round(source_np[..., :3].astype(np.float32) * inv + masked_base), 0, 255).astype(np.uint8)
    masked_rgba = np.concatenate([masked_rgb, np.full((height, width, 1), 255, dtype=np.uint8)], axis=2)
    masked_source = Image.fromarray(masked_rgba, mode="RGBA")

    mask_64 = mask_512.resize((width // 8, height // 8), pil_resample("NEAREST"))
    return {
        "source_original_size": source_original_size,
        "source_512": source_512.convert("RGB"),
        "mask_input_512": mask_input_512.convert("RGBA"),
        "mask_512": mask_512,
        "masked_source_512": masked_source.convert("RGB"),
        "mask_64": mask_64,
    }


def tensor_image_to_pil(tensor) -> Image.Image:
    array = tensor.detach().float().cpu()
    if array.ndim == 4:
        array = array[0]
    array = array.clamp(-1, 1)
    array = ((array + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def tensor_mask_to_pil(mask_tensor) -> Image.Image:
    array = mask_tensor.detach().float().cpu()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        array = array[0]
    array = (array.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L")


def load_pipeline(args: argparse.Namespace, logger):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ConversionError("CUDA was requested but this Python environment reports torch.cuda.is_available() == False.")

    diffusers_dir = Path(args.diffusers_dir).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    config_path = Path(__file__).resolve().parent / "configs" / "sd15_inpaint.yaml"

    if not args.use_ckpt and diffusers_dir.exists():
        logger.info("Loading diffusers pipeline from %s", diffusers_dir)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(str(diffusers_dir))
    else:
        if not ckpt_path.exists():
            raise ConversionError(f"Checkpoint not found: {ckpt_path}")
        if not config_path.exists():
            raise ConversionError(f"Config not found: {config_path}")
        logger.info("Loading checkpoint pipeline from %s", ckpt_path)
        pipe = StableDiffusionInpaintPipeline.from_single_file(
            str(ckpt_path),
            config=DEFAULT_DIFFUSERS_CONFIG_REPO,
            original_config=str(config_path),
            torch_dtype=torch.float32,
            local_files_only=args.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing("auto")
    logger.info("Loaded pipeline scheduler=%s", type(pipe.scheduler).__name__)
    logger.info("Scheduler config=%s", dict(pipe.scheduler.config))
    return pipe


def run_baseline(args: argparse.Namespace, output_dir: Path, logger) -> dict[str, object]:
    import torch

    pipe = load_pipeline(args, logger)
    source_mask = load_and_prepare_source_mask(
        source_path=Path(args.source).resolve(),
        mask_path=Path(args.mask).resolve(),
        width=args.width,
        height=args.height,
        black_mask_means_inpaint=args.black_mask_means_inpaint,
    )

    source_512 = source_mask["source_512"]
    mask_input_512 = source_mask["mask_input_512"]
    mask_512 = source_mask["mask_512"]
    masked_source_512 = source_mask["masked_source_512"]
    mask_64 = source_mask["mask_64"]
    source_original_width, source_original_height = source_mask["source_original_size"]

    save_image(output_dir / "01_source_512.png", source_512)
    save_image(output_dir / "02_mask_input_512.png", mask_input_512)
    save_image(output_dir / "02_mask_512.png", mask_512)
    save_image(output_dir / "03_masked_source_512.png", masked_source_512)
    save_image(output_dir / "04_mask_64.png", mask_64)
    write_text(output_dir / "positive_prompt.txt", args.positive_prompt)
    write_text(output_dir / "negative_prompt.txt", args.negative_prompt)
    write_text(
        output_dir / "run_config.txt",
        "\n".join(
            [
                f"steps={args.steps}",
                f"seed={args.seed}",
                f"strength={args.strength}",
                f"guidance_scale={args.guidance_scale}",
                f"black_mask_means_inpaint={str(args.black_mask_means_inpaint).lower()}",
            ]
        )
        + "\n",
    )

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    do_cfg = args.guidance_scale > 1.0
    batch_size = 1
    num_images_per_prompt = 1
    eta = 0.0

    begin = time.perf_counter()
    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=args.positive_prompt,
            device=pipe.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=args.negative_prompt,
        )
        save_tensor_chw(output_dir / "prompt_cond_f32", prompt_embeds[0])
        if negative_prompt_embeds is not None:
            save_tensor_chw(output_dir / "prompt_uncond_f32", negative_prompt_embeds[0])
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        init_image = pipe.image_processor.preprocess(source_512, height=args.height, width=args.width)
        init_image = init_image.to(dtype=torch.float32)
        mask_condition = pipe.mask_processor.preprocess(mask_512, height=args.height, width=args.width)
        masked_image = init_image * (mask_condition < 0.5)

        save_image(output_dir / "03_masked_source_diffusers_512.png", tensor_image_to_pil(masked_image))

        pipe.scheduler.set_timesteps(args.steps, device=pipe.device)
        timesteps, num_inference_steps = pipe.get_timesteps(
            num_inference_steps=args.steps,
            strength=args.strength,
            device=pipe.device,
        )
        if num_inference_steps < 1:
            raise ConversionError("Strength/steps combination produced fewer than 1 inference step.")
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = math.isclose(args.strength, 1.0, rel_tol=0.0, abs_tol=1e-6)

        latents_outputs = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            pipe.vae.config.latent_channels,
            args.height,
            args.width,
            prompt_embeds.dtype,
            pipe.device,
            generator,
            latents=None,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=pipe.unet.config.in_channels == 4,
        )
        if pipe.unet.config.in_channels == 4:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs
            image_latents = None

        mask_latents, masked_image_latents = pipe.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            args.height,
            args.width,
            prompt_embeds.dtype,
            pipe.device,
            generator,
            do_cfg,
        )

        save_image(output_dir / "04_mask_diffusers_64.png", tensor_mask_to_pil(mask_latents[0]))
        save_tensor_chw(output_dir / "latent_noise_f32", noise[0])
        save_tensor_chw(output_dir / "latent_init_f32", latents[0])
        save_tensor_chw(output_dir / "latent_masked_f32", masked_image_latents[0])
        save_tensor_chw(output_dir / "latent_mask_64_f32", mask_latents[0])
        if image_latents is not None:
            save_tensor_chw(output_dir / "latent_image_f32", image_latents[0])

        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
        timesteps_list = [int(t.item()) for t in timesteps]
        write_text(output_dir / "timesteps.txt", "\n".join(str(v) for v in timesteps_list) + "\n")

        first_step = {}
        for step_index, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            if pipe.unet.config.in_channels == 9:
                latent_model_input = torch.cat([latent_model_input, mask_latents, masked_image_latents], dim=1)

            if step_index == 0:
                save_tensor_chw(output_dir / "unet_in0_f32", latent_model_input[0])
                write_float_bin(output_dir / "unity_unet_timestep_f32.bin", np.asarray([float(t.item())], dtype=np.float32))
                first_step["timestep"] = int(t.item())

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                if step_index == 0:
                    save_tensor_chw(output_dir / "unet_uncond_out_f32", noise_pred_uncond[0])
                    save_tensor_chw(output_dir / "unet_cond_out_f32", noise_pred_text[0])
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                if step_index == 0:
                    save_tensor_chw(output_dir / "unet_cond_out_f32", noise_pred[0])

            if step_index == 0 or args.save_step_latents:
                save_tensor_chw(output_dir / f"epsilon_step_{step_index}_f32", noise_pred[0])

            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if step_index == 0 or args.save_step_latents:
                save_tensor_chw(output_dir / f"latent_step_{step_index}_f32", latents[0])

        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        generated_512 = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    elapsed_ms = int(round((time.perf_counter() - begin) * 1000.0))
    save_image(output_dir / "05_generated_512.png", generated_512)
    final_output = generated_512.resize((source_original_width, source_original_height), pil_resample("BILINEAR"))
    save_image(output_dir / "07_final_output.png", final_output)

    summary = {
        "source": str(Path(args.source).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "output_dir": str(output_dir),
        "diffusers_dir": str(Path(args.diffusers_dir).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()),
        "used_checkpoint": bool(args.use_ckpt or not Path(args.diffusers_dir).expanduser().resolve().exists()),
        "steps": args.steps,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "device": args.device,
        "elapsed_ms": elapsed_ms,
        "timesteps": timesteps_list,
        "first_step": first_step,
    }
    write_text(output_dir / "summary.txt", "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n")
    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    logger.info("Baseline completed in %d ms", elapsed_ms)
    logger.info("Output dir: %s", output_dir)
    return summary


def main() -> int:
    force_utf8_stdio()
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_dir = resolve_existing_path(args.input_dir)

    try:
        if args.width <= 0 or args.height <= 0 or (args.width % 8) != 0 or (args.height % 8) != 0:
            raise ConversionError("Width and height must be positive and divisible by 8.")
        if args.steps <= 0:
            raise ConversionError("Steps must be positive.")
        if args.guidance_scale < 1.0:
            raise ConversionError("Guidance scale must be >= 1.")
        if not 0.0 <= args.strength <= 1.0:
            raise ConversionError("Strength must be in [0, 1].")

        source_path = resolve_input_file(input_dir, args.source, ["01_source_512.png", "00_source.png"], "source")
        mask_path = resolve_input_file(input_dir, args.mask, ["02_mask_512.png", "01_person_mask.png", "02_mask_input_512.png"], "mask")
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else create_default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        args.source = str(source_path)
        args.mask = str(mask_path)

        logger = setup_logging(output_dir / "baseline.log")
        log_versions(logger)
        logger.info("CLI args: %s", vars(args))
        run_baseline(args, output_dir, logger)
        return 0
    except Exception as error:
        print(f"[ERROR] {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
