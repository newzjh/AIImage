#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def mae_max(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a - b)
    return float(d.mean()), float(d.max())


def save_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.float32).tofile(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    dump_dir: Path = args.dump_dir
    out_dir: Path = args.out_dir or (dump_dir / "python_ref")

    prompt = (dump_dir / "positive_prompt.txt").read_text(encoding="utf-8").strip()
    negative_prompt = (dump_dir / "negative_prompt.txt").read_text(encoding="utf-8").strip()
    seed = 123456
    steps = 8
    strength = 1.0
    guidance = 7.5

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cpu")
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = Image.open(dump_dir / "01_source_512.png").convert("RGB")
    mask_image = Image.open(dump_dir / "02_mask_512.png").convert("L")

    batch_size = 1
    num_images_per_prompt = 1
    device = torch.device("cpu")

    do_cfg = guidance > 1.0 and pipe.unet.config.time_cond_proj_dim is None

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps, _ = pipe.get_timesteps(steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    is_strength_max = strength == 1.0

    init_image = pipe.image_processor.preprocess(image, height=512, width=512)
    init_image = init_image.to(dtype=torch.float32)

    latents_outputs = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        pipe.vae.config.latent_channels,
        512,
        512,
        prompt_embeds.dtype,
        device,
        generator,
        None,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=False,
    )
    latents, noise = latents_outputs

    mask_condition = pipe.mask_processor.preprocess(mask_image, height=512, width=512)
    masked_image = init_image * (mask_condition < 0.5)
    mask, masked_image_latents = pipe.prepare_mask_latents(
        mask_condition,
        masked_image,
        batch_size * num_images_per_prompt,
        512,
        512,
        prompt_embeds.dtype,
        device,
        generator,
        do_cfg,
    )

    t = timesteps[0]
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) if do_cfg else prompt_embeds,
            return_dict=False,
        )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_text = noise_pred
            noise_pred_uncond = noise_pred
            noise_pred_cfg = noise_pred

    out_dir.mkdir(parents=True, exist_ok=True)
    save_bin(out_dir / "prompt_cond_f32.bin", prompt_embeds.detach().squeeze(0).cpu().numpy())
    save_bin(out_dir / "prompt_uncond_f32.bin", negative_prompt_embeds.detach().squeeze(0).cpu().numpy())
    save_bin(out_dir / "latent_mask_f32.bin", mask[:1].detach().cpu().numpy())
    save_bin(out_dir / "masked_latents_f32.bin", masked_image_latents[:1].detach().cpu().numpy())
    save_bin(out_dir / "latent_init_f32.bin", latents.detach().cpu().numpy())
    save_bin(out_dir / "unet_in0_f32.bin", latent_model_input[:1].detach().cpu().numpy())
    save_bin(out_dir / "unet_cond_out_f32.bin", noise_pred_text.detach().cpu().numpy())
    save_bin(out_dir / "unet_uncond_out_f32.bin", noise_pred_uncond.detach().cpu().numpy())
    save_bin(out_dir / "unet_eps_f32.bin", noise_pred_cfg.detach().cpu().numpy())

    unity_in0 = np.fromfile(dump_dir / "unity_unet_in0_f32.bin", dtype=np.float32).reshape(1, 9, 64, 64)
    unity_cond = np.fromfile(dump_dir / "unity_unet_cond_out_f32.bin", dtype=np.float32).reshape(1, 4, 64, 64)
    unity_uncond = np.fromfile(dump_dir / "unity_unet_uncond_out_f32.bin", dtype=np.float32).reshape(1, 4, 64, 64)
    unity_eps = np.fromfile(dump_dir / "unity_unet_eps_f32.bin", dtype=np.float32).reshape(1, 4, 64, 64)

    ref_in0 = latent_model_input[:1].detach().cpu().numpy()
    ref_cond = noise_pred_text.detach().cpu().numpy()
    ref_uncond = noise_pred_uncond.detach().cpu().numpy()
    ref_eps = noise_pred_cfg.detach().cpu().numpy()

    with (out_dir / "compare.txt").open("w", encoding="utf-8") as f:
        for name, ref, unity in [
            ("unet_in0", ref_in0, unity_in0),
            ("cond", ref_cond, unity_cond),
            ("uncond", ref_uncond, unity_uncond),
            ("eps", ref_eps, unity_eps),
        ]:
            mae, mx = mae_max(ref, unity)
            f.write(f"{name}: mae={mae:.9g} max={mx:.9g}\n")

    print((out_dir / "compare.txt").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
