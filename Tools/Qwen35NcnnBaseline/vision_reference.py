"""NumPy gold helpers for the Qwen3.5 vision input and RoPE path."""

from __future__ import annotations

import math

import numpy as np


def target_image_size(
    image_height: int,
    image_width: int,
    patch_size: int = 16,
    max_num_patches: int = 49_152,
) -> tuple[int, int]:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    effective = patch_size * 2
    scale = 1.0
    while True:
        target_h = max(effective, math.ceil(image_height * scale / effective) * effective)
        target_w = max(effective, math.ceil(image_width * scale / effective) * effective)
        if (target_h // patch_size) * (target_w // patch_size) <= max_num_patches:
            return target_h, target_w
        scale -= 0.02
        if scale <= 0:
            raise ValueError("could not fit image into the configured patch limit")


def reorder_patches_for_merge(values: np.ndarray, h_patches: int, w_patches: int, merge: int = 2) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != h_patches * w_patches:
        raise ValueError("patch count does not match the patch grid")
    if h_patches % merge or w_patches % merge:
        raise ValueError("patch grid must be divisible by spatial_merge_size")
    tail = values.shape[1:]
    grid = values.reshape(h_patches, w_patches, *tail)
    axes = (0, 2, 1, 3) + tuple(range(4, grid.ndim + 2))
    grouped = grid.reshape(h_patches // merge, merge, w_patches // merge, merge, *tail)
    return np.ascontiguousarray(grouped.transpose(axes).reshape(values.shape))


def rgb_to_duplicated_patches(rgb: np.ndarray, patch_size: int = 16, merge: int = 2) -> np.ndarray:
    """Return [patch, RGB, temporal=2, H, W] exactly as the reference path."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 [height, width, 3] array")
    height, width, _ = rgb.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image must already be resized to a patch-aligned shape")
    h_patches = height // patch_size
    w_patches = width // patch_size
    normalized = (rgb.astype(np.float32) / np.float32(255.5) - np.float32(0.5)) / np.float32(0.5)
    patches = (
        normalized.reshape(h_patches, patch_size, w_patches, patch_size, 3)
        .transpose(0, 2, 4, 1, 3)
        .reshape(h_patches * w_patches, 3, patch_size, patch_size)
    )
    patches = reorder_patches_for_merge(patches, h_patches, w_patches, merge)
    return np.ascontiguousarray(np.repeat(patches[:, :, None, :, :], 2, axis=2))


def vision_rope_2d(
    h_patches: int,
    w_patches: int,
    merge: int = 2,
    theta: float = 10_000.0,
    section: tuple[int, int] = (16, 16),
) -> tuple[np.ndarray, np.ndarray]:
    h_dim, w_dim = section
    rope_dim = h_dim + w_dim
    def inverse_frequency(index: int) -> np.float32:
        exponent = np.float32(index * 2 / rope_dim)
        powered = np.float32(math.pow(float(np.float32(theta)), float(exponent)))
        return np.float32(np.float32(1.0) / powered)

    inv_h = np.asarray([inverse_frequency(i) for i in range(h_dim)], dtype=np.float32)
    inv_w = np.asarray([inverse_frequency(i) for i in range(w_dim)], dtype=np.float32)
    cos_rows: list[np.ndarray] = []
    sin_rows: list[np.ndarray] = []
    for gh in range(h_patches // merge):
        for gw in range(w_patches // merge):
            for mh in range(merge):
                for mw in range(merge):
                    y = np.float32(gh * merge + mh)
                    x = np.float32(gw * merge + mw)
                    angles = np.concatenate((y * inv_h, x * inv_w)).astype(np.float32, copy=False)
                    cosine = np.asarray([np.float32(math.cos(float(angle))) for angle in angles], dtype=np.float32)
                    sine = np.asarray([np.float32(math.sin(float(angle))) for angle in angles], dtype=np.float32)
                    cos_rows.append(np.concatenate((cosine, cosine)))
                    sin_rows.append(np.concatenate((sine, sine)))
    return np.ascontiguousarray(cos_rows, dtype=np.float32), np.ascontiguousarray(sin_rows, dtype=np.float32)
