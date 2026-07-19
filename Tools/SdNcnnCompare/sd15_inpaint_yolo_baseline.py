#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from dump_sd15_unet_pnnx_blobs import (
    DEFAULT_BLOBS,
    HEADER_SHAPES,
    SUPPORTED_BLOBS,
    attach_capture_hooks,
    clone_tensor,
    finalize_captured_blobs,
    flatten_tensor,
    write_float_lines,
    write_stats,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTER_ROOT = REPO_ROOT / "Tools" / "sd15inpainting2ncnnExporter"
DEFAULT_MODEL_DIR = EXPORTER_ROOT / "output" / "diffusers"
DEFAULT_NCNN_DIR = EXPORTER_ROOT / "output" / "ncnn"
DEFAULT_YANQI_TEMP = Path.home() / "AppData" / "Local" / "Temp" / "YanQi" / "AIImage"

MODEL_SIZE = 512
LATENT_SIZE = 64
LATENT_CHANNELS = 4
UNET_IN_CHANNELS = 9

DEFAULT_PROMPT = (
    "best quality, realistic photo, empty indoor background, clean wall, shelf, table, furniture, "
    "coherent texture, seamless fill, background only, no people, no person, no human"
)
DEFAULT_NEGATIVE_PROMPT = (
    "person, people, human, man, woman, child, face, portrait, head, body, skin, hands, arms, legs, "
    "crowd, group photo, selfie, mannequin, statue, reflection, silhouette, duplicate, blurry, "
    "deformed, extra limbs, artifacts, text, watermark"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run an external SD1.5 inpainting PyTorch baseline for AIImage YOLO/SD inpainting dumps "
            "and export Unity/NCNN-comparable UNet step0 tensors."
        )
    )
    ap.add_argument("--dump-dir", type=Path, help="Existing SDInpaintingNcnnRepro dump directory.")
    ap.add_argument("--yolo-dir", type=Path, help="Existing YoloInpaintingRepro dump directory.")
    ap.add_argument("--image", type=Path, help="Explicit source or already-masked RGB image.")
    ap.add_argument("--mask", type=Path, help="Explicit white-means-inpaint mask image.")
    ap.add_argument(
        "--masked-image",
        type=Path,
        help="Optional already-masked RGB image to save/use as baseline input instead of rebuilding it.",
    )
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ncnn-dir", type=Path, default=DEFAULT_NCNN_DIR)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="Limit denoise steps for quick step0 checks.")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--blob", action="append", default=[], help="UNet blob to export; may be repeated.")
    return ap.parse_args()


def read_key_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def choose_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir.resolve()
    if args.dump_dir is not None:
        return (args.dump_dir / "python_sd15_inpaint_baseline").resolve()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return (DEFAULT_YANQI_TEMP / f"AIImage_SD15Inpaint_PythonBaseline_{stamp}").resolve()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path | None]:
    dump_dir = args.dump_dir.resolve() if args.dump_dir is not None else None
    yolo_dir = args.yolo_dir.resolve() if args.yolo_dir is not None else None

    if args.image is not None and args.mask is not None:
        return args.image.resolve(), args.mask.resolve(), args.masked_image.resolve() if args.masked_image else None, dump_dir

    if dump_dir is not None:
        image = dump_dir / "01_source_512.png"
        mask = dump_dir / "02_mask_512.png"
        masked = dump_dir / "03_masked_source_512.png"
        if image.exists() and mask.exists():
            return image, mask, masked if masked.exists() else None, dump_dir

    if yolo_dir is not None:
        image = yolo_dir / "00_source.png"
        mask = yolo_dir / "01_person_mask.png"
        if image.exists() and mask.exists():
            return image, mask, None, yolo_dir

    raise FileNotFoundError(
        "Could not resolve inpainting input. Provide --dump-dir with 01_source_512/02_mask_512, "
        "--yolo-dir with 00_source/01_person_mask, or explicit --image and --mask."
    )


def read_text_or_default(path: Path, default: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


def resolve_run_config(args: argparse.Namespace, input_root: Path | None) -> dict[str, object]:
    config = read_key_values(input_root / "run_config.txt") if input_root is not None else {}
    summary = read_key_values(input_root / "summary.txt") if input_root is not None else {}

    prompt = args.prompt
    negative_prompt = args.negative_prompt
    if prompt is None and input_root is not None:
        prompt = read_text_or_default(input_root / "positive_prompt.txt", DEFAULT_PROMPT)
    if negative_prompt is None and input_root is not None:
        negative_prompt = read_text_or_default(input_root / "negative_prompt.txt", DEFAULT_NEGATIVE_PROMPT)

    return {
        "prompt": prompt if prompt is not None else DEFAULT_PROMPT,
        "negative_prompt": negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT,
        "seed": int(args.seed if args.seed is not None else summary.get("seed", config.get("seed", 123456))),
        "steps": int(args.steps if args.steps is not None else summary.get("steps", config.get("steps", 12))),
        "strength": float(args.strength if args.strength is not None else summary.get("strength", config.get("strength", 1.0))),
        "guidance_scale": float(
            args.guidance_scale
            if args.guidance_scale is not None
            else summary.get("guidance_scale", config.get("guidance_scale", 10.0))
        ),
    }


def load_rgb(path: Path, size: int = MODEL_SIZE) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image


def normalize_mask(path: Path, black_mask_means_inpaint: bool = False, size: int = MODEL_SIZE) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    rgba = Image.open(path).convert("RGBA")
    if rgba.size != (size, size):
        rgba = rgba.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgba, dtype=np.float32)
    luminance = arr[..., :3].mean(axis=2) / 255.0
    alpha = arr[..., 3] / 255.0
    mask = luminance * alpha
    if black_mask_means_inpaint:
        mask = 1.0 - mask
    mask = (mask >= 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def build_masked_image(source: Image.Image, mask: Image.Image) -> Image.Image:
    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    out = src * (1.0 - alpha) + 127.5 * alpha
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), mode="RGB")


def save_bin(path: Path, tensor: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor
    np.asarray(arr, dtype=np.float32).tofile(path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_stats(path: Path, values: torch.Tensor | np.ndarray) -> None:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = values
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    write_stats(path, arr)


def validate_ncnn_unet(ncnn_dir: Path) -> dict[str, object]:
    param = ncnn_dir / "unet.param"
    bin_path = ncnn_dir / "unet.bin"
    if not param.exists():
        raise FileNotFoundError(param)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    conv_line = ""
    for line in param.read_text(encoding="utf-8").splitlines():
        if " conv_160 " in f" {line} ":
            conv_line = line
            break
    if not conv_line:
        raise RuntimeError(f"Could not find conv_160 in {param}")

    weight_size = None
    for token in conv_line.split():
        if token.startswith("6="):
            weight_size = int(token[2:])
            break
    expected = 320 * 9 * 3 * 3
    if weight_size != expected:
        raise RuntimeError(
            f"{param} conv_160 weight size is {weight_size}, expected {expected} for SD1.5 inpainting 9-channel UNet."
        )

    return {
        "ncnn_dir": str(ncnn_dir.resolve()),
        "unet_param": str(param.resolve()),
        "unet_bin": str(bin_path.resolve()),
        "conv_160_line": conv_line,
        "conv_160_weight_size": weight_size,
        "conv_160_in_channels": 9,
    }


def export_official_blobs(out_dir: Path, captured: dict[str, torch.Tensor], requested: list[str], final_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for blob in requested:
        if blob not in captured:
            raise KeyError(f"captured tensor missing for blob {blob}")
        flat = flatten_tensor(captured[blob])
        write_float_lines(out_dir / f"official_unet_blob_{blob}.txt", flat, HEADER_SHAPES.get(blob))
        write_stats(out_dir / f"official_unet_blob_{blob}_stats.txt", flat)

    final_flat = flatten_tensor(captured["out0"])
    write_float_lines(out_dir / final_name, final_flat, HEADER_SHAPES.get("out0"))
    write_stats(out_dir / f"{Path(final_name).stem}_stats.txt", final_flat)


@torch.no_grad()
def run_unet_once(
    pipe: StableDiffusionInpaintPipeline,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    prompt: torch.Tensor,
    requested: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    attach_capture_hooks(pipe, captured)
    try:
        output = pipe.unet(sample, timestep, encoder_hidden_states=prompt, return_dict=False)[0]
    finally:
        restore = getattr(pipe, "_sd_capture_restore", None)
        if restore is not None:
            restore()

    captured["out0"] = clone_tensor(output)
    finalize_captured_blobs(captured)
    missing = sorted(set(requested) - set(captured))
    if missing:
        raise KeyError(f"missing requested blobs: {', '.join(missing)}")
    return output, captured


def write_step_compare(out_dir: Path, reference: np.ndarray, unity_path: Path, name: str) -> str | None:
    if not unity_path.exists():
        return None
    unity = np.fromfile(unity_path, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    if unity.size != ref.size:
        return f"{name}: shape_mismatch ref={ref.size} unity={unity.size}"
    diff = np.abs(ref - unity)
    return f"{name}: mae={float(diff.mean()):.9g} max={float(diff.max()):.9g}"


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    source_path, mask_path, masked_path, input_root = resolve_inputs(args)
    out_dir = choose_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = resolve_run_config(args, input_root)
    requested = args.blob or DEFAULT_BLOBS
    unsupported = sorted(set(requested) - set(SUPPORTED_BLOBS))
    if unsupported:
        raise KeyError(f"unsupported blob requests: {', '.join(unsupported)}")

    ncnn_info = validate_ncnn_unet(args.ncnn_dir.resolve())

    source = load_rgb(source_path)
    mask = normalize_mask(mask_path)
    if masked_path is not None and masked_path.exists():
        masked = load_rgb(masked_path)
    else:
        masked = build_masked_image(source, mask)
    latent_mask = mask.resize((LATENT_SIZE, LATENT_SIZE), Image.Resampling.NEAREST)

    source.save(out_dir / "01_source_512.png")
    mask.save(out_dir / "02_mask_512.png")
    masked.save(out_dir / "03_masked_source_512.png")
    latent_mask.save(out_dir / "04_mask_64.png")
    save_text(out_dir / "positive_prompt.txt", str(run_config["prompt"]))
    save_text(out_dir / "negative_prompt.txt", str(run_config["negative_prompt"]))

    device = torch.device(args.device)
    dtype = torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(args.model_dir.resolve()),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    generator = torch.Generator(device=device).manual_seed(int(run_config["seed"]))
    do_cfg = float(run_config["guidance_scale"]) > 1.0 and pipe.unet.config.time_cond_proj_dim is None

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=str(run_config["prompt"]),
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=str(run_config["negative_prompt"]),
    )

    steps = max(1, int(run_config["steps"]))
    strength = min(1.0, max(0.0, float(run_config["strength"])))
    guidance = float(run_config["guidance_scale"])
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps, _ = pipe.get_timesteps(steps, strength, device)
    if args.max_steps > 0:
        timesteps = timesteps[: args.max_steps]
    if timesteps.numel() < 1:
        raise RuntimeError("No active timesteps.")

    init_image = pipe.image_processor.preprocess(source, height=MODEL_SIZE, width=MODEL_SIZE).to(device=device, dtype=dtype)
    mask_condition = pipe.mask_processor.preprocess(mask, height=MODEL_SIZE, width=MODEL_SIZE).to(device=device, dtype=dtype)
    masked_image = init_image * (mask_condition < 0.5)

    latent_timestep = timesteps[:1].repeat(1)
    latents, noise = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.vae.config.latent_channels,
        height=MODEL_SIZE,
        width=MODEL_SIZE,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=math.isclose(strength, 1.0, rel_tol=0.0, abs_tol=1e-6),
        return_noise=True,
        return_image_latents=False,
    )
    mask_latents, masked_image_latents = pipe.prepare_mask_latents(
        mask_condition,
        masked_image,
        batch_size=1,
        height=MODEL_SIZE,
        width=MODEL_SIZE,
        dtype=dtype,
        device=device,
        generator=generator,
        do_classifier_free_guidance=False,
    )

    save_bin(out_dir / "prompt_cond_f32.bin", prompt_embeds.squeeze(0))
    save_bin(out_dir / "prompt_uncond_f32.bin", negative_prompt_embeds.squeeze(0))
    save_bin(out_dir / "latent_mask_f32.bin", mask_latents)
    save_bin(out_dir / "masked_latents_f32.bin", masked_image_latents)
    save_bin(out_dir / "latent_init_f32.bin", latents)
    save_bin(out_dir / "latent_noise_f32.bin", noise)
    save_stats(out_dir / "latent_init_stats.txt", latents)
    save_stats(out_dir / "latent_mask_stats.txt", mask_latents)
    save_stats(out_dir / "latent_masked_stats.txt", masked_image_latents)

    first_step_saved = False
    step_compare_lines: list[str] = []
    for step_index, timestep in enumerate(timesteps):
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep)
        unet_in0 = torch.cat([latent_model_input, mask_latents, masked_image_latents], dim=1)
        if unet_in0.shape[1] != UNET_IN_CHANNELS:
            raise RuntimeError(f"UNet input has {unet_in0.shape[1]} channels, expected {UNET_IN_CHANNELS}.")

        cond_out, cond_captured = run_unet_once(pipe, unet_in0, timestep, prompt_embeds, requested)
        uncond_out, uncond_captured = run_unet_once(pipe, unet_in0, timestep, negative_prompt_embeds, requested)
        eps = uncond_out + guidance * (cond_out - uncond_out)

        if not first_step_saved:
            save_bin(out_dir / "unet_in0_f32.bin", unet_in0)
            save_bin(out_dir / "unet_cond_out_f32.bin", cond_out)
            save_bin(out_dir / "unet_uncond_out_f32.bin", uncond_out)
            save_bin(out_dir / "unet_eps_f32.bin", eps)
            save_bin(out_dir / "unity_unet_timestep_f32.bin", np.asarray([float(timestep.item())], dtype=np.float32))
            save_text(out_dir / "timestep.txt", f"{int(timestep.item())}\n")
            save_stats(out_dir / "unet_in0_stats.txt", unet_in0)
            save_stats(out_dir / "unet_cond_out_stats.txt", cond_out)
            save_stats(out_dir / "unet_uncond_out_stats.txt", uncond_out)
            save_stats(out_dir / "unet_eps_stats.txt", eps)

            write_float_lines(
                out_dir / "official_unet_in0.txt",
                flatten_tensor(unet_in0),
                (3, LATENT_SIZE, LATENT_SIZE, 1, UNET_IN_CHANNELS, 1),
            )
            write_float_lines(out_dir / "official_unet_in1_t.txt", np.asarray([float(timestep.item())], dtype=np.float32), (1, 1, 1, 1, 1, 1))
            write_float_lines(out_dir / "official_cond.txt", flatten_tensor(prompt_embeds), (2, 768, 77, 1, 1, 1))
            write_float_lines(out_dir / "official_uncond.txt", flatten_tensor(negative_prompt_embeds), (2, 768, 77, 1, 1, 1))
            export_official_blobs(out_dir / "unet_step0_cond", cond_captured, requested, "official_unet_outout_cond.txt")
            export_official_blobs(out_dir / "unet_step0_uncond", uncond_captured, requested, "official_unet_outout_uncond.txt")

            # Keep the root-level official names compatible with existing C# replay defaults for cond.
            export_official_blobs(out_dir, cond_captured, requested, "official_unet_outout_cond.txt")
            write_float_lines(
                out_dir / "official_unet_outout_uncond.txt",
                flatten_tensor(uncond_captured["out0"]),
                HEADER_SHAPES.get("out0"),
            )

            if input_root is not None:
                comparisons = [
                    ("unet_in0", unet_in0.detach().cpu().numpy(), input_root / "unity_unet_in0_f32.bin"),
                    ("cond", cond_out.detach().cpu().numpy(), input_root / "unity_unet_cond_out_f32.bin"),
                    ("uncond", uncond_out.detach().cpu().numpy(), input_root / "unity_unet_uncond_out_f32.bin"),
                    ("eps", eps.detach().cpu().numpy(), input_root / "unity_unet_eps_f32.bin"),
                ]
                for name, ref, unity_path in comparisons:
                    line = write_step_compare(out_dir, ref, unity_path, name)
                    if line is not None:
                        step_compare_lines.append(line)
                if step_compare_lines:
                    save_text(out_dir / "compare_unity_step0.txt", "\n".join(step_compare_lines) + "\n")
            first_step_saved = True

        latents = pipe.scheduler.step(eps, timestep, latents, return_dict=False)[0]
        save_bin(out_dir / f"latent_step_{step_index}_f32.bin", latents)
        save_stats(out_dir / f"latent_step_{step_index}_stats.txt", latents)

    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0].detach()
    image = pipe.image_processor.postprocess(decoded, output_type="pil", do_denormalize=[True])[0]
    image.save(out_dir / "05_generated_512.png")
    image.save(out_dir / "06_generated_fullres.png")
    image.save(out_dir / "07_final_output.png")

    run_config_text = (
        f"seed={int(run_config['seed'])}\n"
        f"steps={steps}\n"
        f"strength={strength:.6f}\n"
        f"guidance_scale={guidance:.6f}\n"
        "black_mask_means_inpaint=false\n"
        f"active_timesteps={int(timesteps.numel())}\n"
        f"debug_max_denoise_steps={int(args.max_steps)}\n"
        f"first_timestep={int(timesteps[0].item())}\n"
    )
    save_text(out_dir / "run_config.txt", run_config_text)

    manifest = {
        "source": str(source_path.resolve()),
        "mask": str(mask_path.resolve()),
        "masked_image": str(masked_path.resolve()) if masked_path is not None and masked_path.exists() else None,
        "input_root": str(input_root.resolve()) if input_root is not None else None,
        "out_dir": str(out_dir),
        "model_dir": str(args.model_dir.resolve()),
        "ncnn": ncnn_info,
        "run_config": run_config,
        "active_timesteps": [int(t.item()) for t in timesteps],
        "requested_blobs": requested,
        "outputs": {
            "generated_512": str((out_dir / "05_generated_512.png").resolve()),
            "step0_cond_blobs": str((out_dir / "unet_step0_cond").resolve()),
            "step0_uncond_blobs": str((out_dir / "unet_step0_uncond").resolve()),
            "compare_unity_step0": str((out_dir / "compare_unity_step0.txt").resolve())
            if step_compare_lines
            else None,
        },
    }
    save_text(out_dir / "baseline_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"out_dir={out_dir}")
    print(f"generated={out_dir / '05_generated_512.png'}")
    print(f"first_timestep={int(timesteps[0].item())}")
    if step_compare_lines:
        print((out_dir / "compare_unity_step0.txt").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
