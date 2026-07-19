#!/usr/bin/env python
"""Run the reference DeepFillV2 PyTorch project without torchvision.

The official `deepfillv2-pytorch-master/test.py` imports torchvision only for
`ToTensor()`.  Some local Python environments have a torch/torchvision ABI
mismatch, so this script mirrors the same preprocessing/postprocessing with
PIL + numpy while still importing the official model implementation directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT / "ref" / "deepfillv2" / "deepfillv2-pytorch-master"
DEFAULT_CASE_DIR = DEFAULT_REPO / "examples" / "inpaint"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ref" / "deepfillv2" / "states_tf_celebahq.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Tools" / "DeepFillV2" / "output" / "pytorch_case1_states_tf_celebahq"


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _to_tensor_01(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value > 1.0:
            arr /= max_value
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()


def _save_minus1_1_rgb(tensor: torch.Tensor, path: Path) -> None:
    image = ((tensor[0].detach().cpu().permute(1, 2, 0) + 1.0) * 127.5)
    image = image.to(dtype=torch.uint8).numpy()
    Image.fromarray(image).save(path)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > 127.5


def _compare(
    reference: Path,
    actual: Path,
    mask_path: Path | None,
    mask_array: np.ndarray | None = None,
) -> dict[str, float | int | list[int]]:
    ref = _load_rgb(reference)
    got = _load_rgb(actual)
    h = min(ref.shape[0], got.shape[0])
    w = min(ref.shape[1], got.shape[1])
    ref = ref[:h, :w]
    got = got[:h, :w]
    diff = np.abs(ref - got)
    metrics: dict[str, float | int | list[int]] = {
        "height": int(h),
        "width": int(w),
        "full_mae_rgb": float(diff.mean()),
        "full_max_abs_rgb": int(diff.max()),
    }
    if mask_path is not None or mask_array is not None:
        mask = (_load_mask(mask_path) if mask_array is None else mask_array)[:h, :w]
        metrics["mask_pixels"] = int(mask.sum())
        metrics["mask_coverage"] = float(mask.mean())
        if mask.any():
            metrics["masked_mae_rgb"] = float(diff[mask].mean())
            metrics["masked_max_abs_rgb"] = int(diff[mask].max())
        if (~mask).any():
            metrics["unmasked_mae_rgb"] = float(diff[~mask].mean())
            metrics["unmasked_max_abs_rgb"] = int(diff[~mask].max())
    return metrics


def _restore_original_size(cropped_output: Path, original_image: Path, restored_output: Path) -> Path:
    """Paste the 8-aligned model output back into the original image extent.

    The official PyTorch code crops inputs to multiples of 8 before inference.
    The checked-in `examples/inpaint/case1_out.png`, however, keeps the original
    406-pixel width and preserves the untouched rightmost 6 columns from the
    source image.  This helper keeps the raw cropped output reproducible while
    allowing like-for-like comparison with that checked-in artifact.
    """

    cropped = np.asarray(Image.open(cropped_output).convert("RGB"), dtype=np.uint8)
    restored = np.asarray(Image.open(original_image).convert("RGB"), dtype=np.uint8).copy()
    h = min(restored.shape[0], cropped.shape[0])
    w = min(restored.shape[1], cropped.shape[1])
    restored[:h, :w] = cropped[:h, :w]
    Image.fromarray(restored).save(restored_output)
    return restored_output


def _install_case1_2021_attention(generator: torch.nn.Module, networks_tf: Any) -> None:
    """Restore the contextual-attention math used when case1_out.png was added.

    The checked-in example predates commit b56ad856, which changed nearest
    downsampling and patch normalization. The generator weights and topology
    are unchanged, so replacing this one forward method reproduces the original
    example path without modifying the reference checkout.
    """

    def forward(attention: torch.nn.Module, f: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None):
        device = f.device
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())

        kernel = 2 * attention.rate
        raw_w = networks_tf.extract_image_patches(
            b,
            ksizes=[kernel, kernel],
            strides=[attention.rate * attention.stride, attention.rate * attention.stride],
            rates=[1, 1],
            padding="same",
        )
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        f = F.interpolate(f, scale_factor=1.0 / attention.rate, mode="nearest", recompute_scale_factor=False)
        b = F.interpolate(b, scale_factor=1.0 / attention.rate, mode="nearest", recompute_scale_factor=False)
        int_fs = list(f.size())
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)

        w = networks_tf.extract_image_patches(
            b,
            ksizes=[attention.ksize, attention.ksize],
            strides=[attention.stride, attention.stride],
            rates=[1, 1],
            padding="same",
        )
        w = w.view(int_bs[0], int_bs[1], attention.ksize, attention.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)
        w_groups = torch.split(w, 1, dim=0)

        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            mask = F.interpolate(
                mask,
                scale_factor=1.0 / ((2**attention.n_down) * attention.rate),
                mode="nearest",
                recompute_scale_factor=False,
            )
        int_ms = list(mask.size())
        m = networks_tf.extract_image_patches(
            mask,
            ksizes=[attention.ksize, attention.ksize],
            strides=[attention.stride, attention.stride],
            rates=[1, 1],
            padding="same",
        )
        m = m.view(int_ms[0], int_ms[1], attention.ksize, attention.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)[0]
        mm = (torch.mean(m, dim=[1, 2, 3], keepdim=True) == 0.0).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)

        outputs = []
        fuse_weight = torch.eye(attention.fuse_k, device=device).view(
            1, 1, attention.fuse_k, attention.fuse_k
        )
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            wi = wi[0]
            max_wi = torch.sqrt(
                torch.sum(torch.pow(wi, 2) + 1.0e-4, dim=[1, 2, 3], keepdim=True)
            )
            wi_normed = wi / max_wi
            xi = networks_tf.same_padding(
                xi,
                [attention.ksize, attention.ksize],
                [1, 1],
                [1, 1],
            )
            yi = F.conv2d(xi, wi_normed, stride=1)

            if attention.fuse:
                k = attention.fuse_k
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = networks_tf.same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = networks_tf.same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            yi = yi * mm
            yi = F.softmax(yi * attention.softmax_scale, dim=1)
            yi = yi * mm
            yi = F.conv_transpose2d(yi, raw_wi[0], stride=attention.rate, padding=1) / 4.0
            outputs.append(yi)

        y = torch.cat(outputs, dim=0).contiguous().view(raw_int_fs)
        if attention.return_flow:
            raise RuntimeError("case1-2021 compatibility mode does not support return_flow=True")
        return y, None

    generator.contextual_attention.forward = forward.__get__(
        generator.contextual_attention,
        type(generator.contextual_attention),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepFillV2 PyTorch baseline without torchvision.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--image", type=Path, default=DEFAULT_CASE_DIR / "case1.png")
    parser.add_argument("--mask", type=Path, default=DEFAULT_CASE_DIR / "case1_mask.png")
    parser.add_argument(
        "--mask-from-masked-example",
        type=Path,
        default=None,
        help=(
            "Use the pure-white pixels from case1_masked.png as the inference mask. "
            "The checked-in case1_out.png uses this larger mask rather than case1_mask.png."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR / "case1_out_states_tf_celebahq.png")
    parser.add_argument("--compare", type=Path, default=DEFAULT_CASE_DIR / "case1_out.png")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_OUTPUT_DIR / "case1_metrics_states_tf_celebahq.json")
    parser.add_argument(
        "--entry",
        choices=["forward", "infer"],
        default="forward",
        help=(
            "forward mirrors test.py explicitly; infer mirrors the official "
            "Generator.infer() path used by the notebook/app helpers."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--attention-semantics",
        choices=["current", "case1-2021"],
        default="current",
        help="Use current reference math or the math present when case1_out.png was committed.",
    )
    parser.add_argument(
        "--restore-original-size",
        action="store_true",
        help=(
            "Also keep a *_cropped.png copy and paste the cropped inference "
            "result back into the original image extent before writing --out. "
            "This matches the checked-in case1_out.png dimensions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    repo = args.repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)
    sys.path.insert(0, str(repo))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = _torch_load(args.checkpoint.resolve(), device)
    generator_state_dict = checkpoint["G"]

    if "stage1.conv1.conv.weight" in generator_state_dict.keys():
        from model.networks import Generator  # type: ignore

        model_family = "networks.py"
    else:
        from model import networks_tf  # type: ignore

        Generator = networks_tf.Generator

        model_family = "networks_tf.py"

    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)
    generator.eval()
    if args.attention_semantics == "case1-2021":
        if model_family != "networks_tf.py":
            raise ValueError("case1-2021 attention semantics only apply to networks_tf.py checkpoints")
        _install_case1_2021_attention(generator, networks_tf)

    image_pil = Image.open(args.image)
    effective_mask_array: np.ndarray | None = None
    if args.mask_from_masked_example is not None:
        masked_example = np.asarray(Image.open(args.mask_from_masked_example).convert("RGB"), dtype=np.uint8)
        effective_mask_array = np.all(masked_example == 255, axis=2)
        mask_pil = Image.fromarray((effective_mask_array.astype(np.uint8) * 255), mode="L")
    else:
        mask_pil = Image.open(args.mask)
    image_tensor = _to_tensor_01(image_pil)
    mask_tensor = _to_tensor_01(mask_pil)

    _, h, w = image_tensor.shape
    grid = 8
    h8 = h // grid * grid
    w8 = w // grid * grid

    start = time.perf_counter()
    stage1_path = args.out.with_name(args.out.stem + "_stage1.png")
    stage2_path = args.out.with_name(args.out.stem + "_stage2.png")
    if args.entry == "infer":
        with torch.inference_mode():
            outputs = generator.infer(
                image_tensor[:3].to(device),
                mask_tensor.to(device),
                return_vals=["inpainted", "stage1", "stage2"],
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        inpainted_np, stage1_np, stage2_np = outputs
        Image.fromarray(inpainted_np).save(args.out)
        Image.fromarray(stage1_np).save(stage1_path)
        Image.fromarray(stage2_np).save(stage2_path)
    else:
        image = image_tensor[:3, :h8, :w8].unsqueeze(0)
        mask = mask_tensor[0:1, :h8, :w8].unsqueeze(0)
        image = (image * 2.0 - 1.0).to(device)
        mask = (mask > 0.5).to(dtype=torch.float32, device=device)
        image_masked = image * (1.0 - mask)
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)

        with torch.inference_mode():
            x_stage1, x_stage2 = generator(x, mask)
            image_inpainted = image * (1.0 - mask) + x_stage2 * mask
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        _save_minus1_1_rgb(image_inpainted, args.out)
        _save_minus1_1_rgb(x_stage1, stage1_path)
        _save_minus1_1_rgb(x_stage2, stage2_path)

    cropped_output_path: Path | None = None
    if args.restore_original_size:
        cropped_output_path = args.out.with_name(args.out.stem + "_cropped.png")
        Image.open(args.out).convert("RGB").save(cropped_output_path)
        _restore_original_size(cropped_output_path, args.image.resolve(), args.out)

    metrics: dict[str, Any] = {
        "repo": str(repo),
        "model_family": model_family,
        "checkpoint": str(args.checkpoint.resolve()),
        "image": str(args.image.resolve()),
        "mask": str(args.mask.resolve()),
        "mask_from_masked_example": (
            str(args.mask_from_masked_example.resolve()) if args.mask_from_masked_example else None
        ),
        "output": str(args.out.resolve()),
        "stage1": str(stage1_path.resolve()),
        "stage2": str(stage2_path.resolve()),
        "cropped_output": str(cropped_output_path.resolve()) if cropped_output_path else None,
        "restore_original_size": bool(args.restore_original_size),
        "device": str(device),
        "entry": args.entry,
        "attention_semantics": args.attention_semantics,
        "input_shape_chw": [int(v) for v in image_tensor[:3, :h8, :w8].shape],
        "cropped_height": int(h8),
        "cropped_width": int(w8),
        "elapsed_ms": elapsed_ms,
    }
    if args.compare:
        metrics["comparison"] = _compare(
            args.compare.resolve(),
            args.out.resolve(),
            args.mask.resolve(),
            effective_mask_array,
        )

    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
