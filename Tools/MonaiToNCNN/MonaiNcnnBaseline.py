from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import psutil
import torch
import torch.nn.functional as F


@dataclass
class VolumeData:
    path: Path
    data: np.ndarray
    spacing: tuple[float, float, float] | None
    affine: np.ndarray | None
    source_format: str
    nrrd_header: dict[str, str] | None = None


def env_flag_enabled(*names: str) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
    return False


def env_positive_int(*names: str) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        text = raw.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value > 0:
            return value
    return 0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="External MONAI baseline for comparing Unity NCNN reproduction outputs."
    )
    parser.add_argument(
        "--bundle-root",
        default=str(root / "bundle_cache" / "brats_mri_segmentation"),
        help="Path to the MONAI bundle root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "manual_test" / "brats_mri_segmentation_baseline"),
        help="Directory where baseline dumps are written.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input medical volume path. Repeat up to 4 times for multi-modality input.",
    )
    parser.add_argument(
        "--target-shape",
        default="224,224,144",
        help="Target spatial shape in MONAI/PyTorch order D,H,W.",
    )
    parser.add_argument(
        "--channel-fill",
        choices=("duplicate-first", "duplicate-last", "zero"),
        default="duplicate-first",
        help="How to fill missing channels when fewer than the required model channels are supplied.",
    )
    parser.add_argument(
        "--normalize-nonzero",
        action="store_true",
        default=True,
        help="Apply MONAI-style per-channel non-zero normalization.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for per-channel masks.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--case-name",
        default="",
        help="Optional folder name override. Defaults to the first input stem.",
    )
    parser.add_argument(
        "--infer-mode",
        choices=("ncnn-fixed", "monai-sliding-window"),
        default="ncnn-fixed",
        help="ncnn-fixed matches the current NCNN export comparison path; monai-sliding-window follows the MONAI bundle inferer more closely.",
    )
    parser.add_argument(
        "--roi-size",
        default="240,240,160",
        help="Sliding window ROI size in D,H,W order for monai-sliding-window mode.",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap for monai-sliding-window mode.",
    )
    parser.add_argument(
        "--save-nifti",
        action="store_true",
        help="Also save the merged label map as NIfTI when affine metadata is available.",
    )
    parser.add_argument(
        "--save-restored",
        action="store_true",
        default=True,
        help="Save output volumes restored back to the first input volume shape and format.",
    )
    parser.add_argument(
        "--task-mode",
        choices=("auto", "brats-multilabel", "multiclass"),
        default="auto",
        help="Postprocess mode. auto keeps BraTS multilabel behavior for 3-channel outputs, otherwise uses multiclass argmax.",
    )
    parser.add_argument(
        "--save-channel-masks",
        action="store_true",
        help="Also save per-channel/restored masks when the output channel count is manageable.",
    )
    parser.add_argument(
        "--max-saved-mask-channels",
        type=int,
        default=16,
        help="Maximum number of channels for saving per-channel mask volumes when --save-channel-masks is enabled.",
    )
    parser.add_argument(
        "--label-subset-values",
        default="",
        help="Optional comma-separated label values to merge into one binary subset mask from the restored labelmap.",
    )
    parser.add_argument(
        "--label-subset-name",
        default="",
        help="Optional output name for the merged binary label subset.",
    )
    parser.add_argument(
        "--label-subsets",
        default="",
        help="Optional semicolon-separated merged binary subsets, for example ventricles=1,2,18,19,20,21;skull=7.",
    )
    parser.add_argument(
        "--save-multiclass-probs",
        action="store_true",
        help="Also materialize and save full multiclass softmax probabilities. Disabled by default to avoid very large allocations.",
    )
    parser.add_argument(
        "--save-multiclass-logits",
        action="store_true",
        help="Also materialize and save full multiclass logits. Disabled by default to avoid very large allocations.",
    )
    parser.add_argument(
        "--low-power-mode",
        action="store_true",
        default=env_flag_enabled("AIIMAGE_MONAI_LOW_POWER_MODE", "MONAI_LOW_POWER_MODE"),
        help="Reduce instantaneous CPU load while keeping the output labelmap unchanged.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=env_positive_int("AIIMAGE_MONAI_TORCH_THREADS", "TORCH_NUM_THREADS"),
        help="Optional torch intra-op thread count override.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=env_positive_int("AIIMAGE_MONAI_TORCH_INTEROP_THREADS", "TORCH_NUM_INTEROP_THREADS"),
        help="Optional torch inter-op thread count override.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_bundle_pythonpath(bundle_root: Path) -> None:
    bundle_path = str(bundle_root.resolve())
    if bundle_path not in sys.path:
        sys.path.insert(0, bundle_path)


def resolve_model(bundle_root: Path):
    ensure_bundle_pythonpath(bundle_root)
    config = read_json(bundle_root / "configs" / "inference.json")
    metadata = read_json(bundle_root / "configs" / "metadata.json")

    from monai.bundle import ConfigParser
    from monai.networks import copy_model_state

    parser = ConfigParser(config)
    network = parser.get_parsed_content("network_def")
    checkpoint_path = bundle_root / "models" / "model.pt"
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(payload)
    copy_model_state(dst=network, src=state_dict)
    network.eval()
    return network, config, metadata


def extract_state_dict(payload):
    if isinstance(payload, dict):
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
        for key in ("model", "network", "state_dict", "net", "module"):
            if key in payload:
                nested = extract_state_dict(payload[key])
                if nested:
                    return nested
    raise RuntimeError("Could not extract PyTorch state_dict from checkpoint.")


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 integers for target shape, got: {text}")
    return parts[0], parts[1], parts[2]


def load_volume(path: Path) -> VolumeData:
    lower = path.name.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_nifti(path)
    if lower.endswith(".nrrd") or lower.endswith(".nhdr"):
        return load_nrrd(path)
    raise ValueError(f"Unsupported volume format: {path}")


def load_nifti(path: Path) -> VolumeData:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"NIfTI volume must be 3D after squeeze, got {data.shape} from {path}")
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return VolumeData(
        path=path,
        data=np.ascontiguousarray(data, dtype=np.float32),
        spacing=spacing,
        affine=np.asarray(image.affine, dtype=np.float32),
        source_format="nifti",
        nrrd_header=None,
    )


def load_nrrd(path: Path) -> VolumeData:
    header_bytes, data_offset = split_nrrd_header(path)
    header_text = header_bytes.decode("ascii", errors="replace")
    header = parse_nrrd_header(header_text)

    if header.get("dimension") not in (None, "3"):
        raise ValueError(f"Only 3D scalar NRRD is supported for now: {path}")

    sizes = [int(v) for v in header["sizes"].split()]
    if len(sizes) != 3:
        raise ValueError(f"Only 3D scalar NRRD is supported for now: {path}")

    dtype = nrrd_dtype_to_numpy(header.get("type", "float"))
    endian = header.get("endian", "little").strip().lower()
    dtype = dtype.newbyteorder("<" if endian != "big" else ">")
    encoding = header.get("encoding", "raw").strip().lower()

    if path.suffix.lower() == ".nhdr":
        data_file = header.get("data file")
        if not data_file:
            raise ValueError(f"Detached NRRD header missing data file field: {path}")
        raw_path = (path.parent / data_file).resolve()
        raw_bytes = raw_path.read_bytes()
    else:
        raw_bytes = path.read_bytes()[data_offset:]

    if encoding in ("gzip", "gz"):
        raw_bytes = gzip.decompress(raw_bytes)
    elif encoding != "raw":
        raise ValueError(f"Unsupported NRRD encoding: {encoding} in {path}")

    data = np.frombuffer(raw_bytes, dtype=dtype)
    expected = sizes[0] * sizes[1] * sizes[2]
    if data.size != expected:
        raise ValueError(f"NRRD voxel count mismatch: expected {expected}, got {data.size} for {path}")

    # Keep the file axis order and match the raw loader's C-order layout.
    data = np.asarray(data.reshape((sizes[0], sizes[1], sizes[2]), order="F"), dtype=np.float32)

    spacing = None
    if "spacings" in header:
        values = [float(v) for v in header["spacings"].split()]
        if len(values) >= 3:
            spacing = (values[0], values[1], values[2])
    elif "space directions" in header:
        spacing = parse_nrrd_space_directions(header["space directions"])

    return VolumeData(
        path=path,
        data=np.ascontiguousarray(data, dtype=np.float32),
        spacing=spacing,
        affine=None,
        source_format="nrrd",
        nrrd_header=header,
    )


def split_nrrd_header(path: Path) -> tuple[bytes, int]:
    content = path.read_bytes()
    marker = b"\r\n\r\n"
    offset = content.find(marker)
    if offset >= 0:
        return content[:offset], offset + len(marker)

    marker = b"\n\n"
    offset = content.find(marker)
    if offset >= 0:
        return content[:offset], offset + len(marker)

    raise ValueError(f"NRRD header terminator not found: {path}")


def parse_nrrd_header(header_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("NRRD"):
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        result[key.strip().lower()] = value.strip()
    return result


def nrrd_dtype_to_numpy(type_name: str) -> np.dtype:
    normalized = type_name.strip().lower()
    mapping = {
        "char": np.dtype(np.int8),
        "signed char": np.dtype(np.int8),
        "int8": np.dtype(np.int8),
        "uchar": np.dtype(np.uint8),
        "unsigned char": np.dtype(np.uint8),
        "uint8": np.dtype(np.uint8),
        "short": np.dtype(np.int16),
        "short int": np.dtype(np.int16),
        "signed short": np.dtype(np.int16),
        "signed short int": np.dtype(np.int16),
        "int16": np.dtype(np.int16),
        "ushort": np.dtype(np.uint16),
        "unsigned short": np.dtype(np.uint16),
        "unsigned short int": np.dtype(np.uint16),
        "uint16": np.dtype(np.uint16),
        "int": np.dtype(np.int32),
        "signed int": np.dtype(np.int32),
        "int32": np.dtype(np.int32),
        "uint": np.dtype(np.uint32),
        "unsigned int": np.dtype(np.uint32),
        "uint32": np.dtype(np.uint32),
        "float": np.dtype(np.float32),
        "double": np.dtype(np.float64),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported NRRD scalar type: {type_name}")
    return mapping[normalized]


def parse_nrrd_space_directions(text: str) -> tuple[float, float, float] | None:
    values: list[float] = []
    for token in text.split():
        token = token.strip()
        if token.lower() == "none":
            values.append(1.0)
            continue
        if token.startswith("(") and token.endswith(")"):
            parts = [p.strip() for p in token[1:-1].split(",") if p.strip()]
            if not parts:
                continue
            length = math.sqrt(sum(float(part) * float(part) for part in parts))
            values.append(length)
    if len(values) >= 3:
        return values[0], values[1], values[2]
    return None


def build_affine_from_spacing_and_origin(
    spacing: tuple[float, float, float] | None,
    origin: tuple[float, float, float] | None = None,
) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)
    if spacing is not None:
        affine[0, 0] = float(spacing[0])
        affine[1, 1] = float(spacing[1])
        affine[2, 2] = float(spacing[2])
    if origin is not None:
        affine[0, 3] = float(origin[0])
        affine[1, 3] = float(origin[1])
        affine[2, 3] = float(origin[2])
    return affine


def parse_nrrd_space_origin(header: dict[str, str] | None) -> tuple[float, float, float] | None:
    if not header:
        return None
    raw = header.get("space origin")
    if not raw:
        return None
    text = raw.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    parts = [part.strip() for part in text[1:-1].split(",") if part.strip()]
    if len(parts) < 3:
        return None
    return float(parts[0]), float(parts[1]), float(parts[2])


def get_volume_affine(volume: VolumeData) -> np.ndarray:
    if volume.affine is not None:
        return np.asarray(volume.affine, dtype=np.float32)
    return build_affine_from_spacing_and_origin(volume.spacing, parse_nrrd_space_origin(volume.nrrd_header))


def resample_volume_to_reference(volume: VolumeData, reference: VolumeData) -> VolumeData:
    if volume.data.shape == reference.data.shape and volume.spacing == reference.spacing:
        return volume

    src = torch.from_numpy(np.ascontiguousarray(volume.data, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    ref_shape = tuple(int(v) for v in reference.data.shape)
    ref_affine = get_volume_affine(reference)
    src_affine = get_volume_affine(volume)
    transform = np.linalg.inv(src_affine) @ ref_affine

    d_out, h_out, w_out = ref_shape
    z = np.arange(d_out, dtype=np.float32)
    y = np.arange(h_out, dtype=np.float32)
    x = np.arange(w_out, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    ones = np.ones_like(xx, dtype=np.float32)
    dst_index = np.stack([xx, yy, zz, ones], axis=-1)
    src_index = dst_index @ transform.T

    w_in = max(1, int(volume.data.shape[2]))
    h_in = max(1, int(volume.data.shape[1]))
    d_in = max(1, int(volume.data.shape[0]))

    if w_in == 1:
        x_norm = np.zeros_like(src_index[..., 0], dtype=np.float32)
    else:
        x_norm = (src_index[..., 0] / (w_in - 1)) * 2.0 - 1.0
    if h_in == 1:
        y_norm = np.zeros_like(src_index[..., 1], dtype=np.float32)
    else:
        y_norm = (src_index[..., 1] / (h_in - 1)) * 2.0 - 1.0
    if d_in == 1:
        z_norm = np.zeros_like(src_index[..., 2], dtype=np.float32)
    else:
        z_norm = (src_index[..., 2] / (d_in - 1)) * 2.0 - 1.0

    grid = np.stack([x_norm, y_norm, z_norm], axis=-1)
    grid_tensor = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32)).unsqueeze(0)
    sampled = F.grid_sample(
        src,
        grid_tensor,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    resampled = sampled[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    return VolumeData(
        path=volume.path,
        data=np.ascontiguousarray(resampled, dtype=np.float32),
        spacing=reference.spacing,
        affine=get_volume_affine(reference),
        source_format=volume.source_format,
        nrrd_header=volume.nrrd_header,
    )


def align_modalities(volumes: list[VolumeData]) -> tuple[np.ndarray, list[dict]]:
    if not volumes:
        raise ValueError("At least one volume is required.")
    reference = volumes[0]
    ref_shape = reference.data.shape
    aligned: list[np.ndarray] = []
    entries: list[dict] = []
    for volume in volumes:
        original_shape = volume.data.shape
        original_spacing = volume.spacing
        if volume.data.shape != ref_shape or volume.spacing != reference.spacing:
            volume = resample_volume_to_reference(volume, reference)
        aligned.append(volume.data)
        entries.append(
            {
                "path": str(volume.path),
                "shape": list(volume.data.shape),
                "spacing": list(volume.spacing) if volume.spacing is not None else None,
                "format": volume.source_format,
                "original_shape": list(original_shape),
                "original_spacing": list(original_spacing) if original_spacing is not None else None,
                "resampled_to_reference": bool(original_shape != ref_shape or original_spacing != reference.spacing),
            }
        )
    return np.stack(aligned, axis=0), entries


def fill_channels(channels: np.ndarray, required_channels: int, mode: str) -> np.ndarray:
    if channels.ndim != 4:
        raise ValueError(f"Expected C,D,H,W array, got {channels.shape}")
    if channels.shape[0] == required_channels:
        return channels
    if channels.shape[0] > required_channels:
        return channels[:required_channels]
    if channels.shape[0] <= 0:
        raise ValueError("No channels available to fill.")

    filled = [channels[i] for i in range(channels.shape[0])]
    while len(filled) < required_channels:
        if mode == "duplicate-first":
            filled.append(channels[0].copy())
        elif mode == "duplicate-last":
            filled.append(filled[-1].copy())
        else:
            filled.append(np.zeros_like(channels[0], dtype=np.float32))
    return np.stack(filled, axis=0)


def center_crop_or_pad(array: np.ndarray, target_shape: tuple[int, int, int], pad_value: int | float = 0) -> np.ndarray:
    if array.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions, got {array.shape}")

    result = array
    spatial_offset = result.ndim - 3
    for axis_offset, target in enumerate(target_shape):
        axis = spatial_offset + axis_offset
        current = result.shape[axis]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slicer = [slice(None)] * result.ndim
            slicer[axis] = slice(start, end)
            result = result[tuple(slicer)]
        elif current < target:
            before = (target - current) // 2
            after = target - current - before
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (before, after)
            result = np.pad(result, pad_width, mode="constant", constant_values=pad_value)
    return np.ascontiguousarray(result)


def normalize_nonzero_per_channel(channels: np.ndarray) -> np.ndarray:
    result = channels.copy()
    chunk_size = 1_000_000
    for c in range(result.shape[0]):
        values = result[c]
        flat = values.reshape(-1)
        count = 0
        sum_value = 0.0
        sum_square = 0.0
        for start in range(0, flat.size, chunk_size):
            chunk = flat[start:start + chunk_size]
            mask = chunk != 0
            if not np.any(mask):
                continue
            nonzero = chunk[mask].astype(np.float64, copy=False)
            count += int(mask.sum())
            sum_value += float(nonzero.sum(dtype=np.float64))
            sum_square += float((nonzero * nonzero).sum(dtype=np.float64))

        if count <= 0:
            continue

        mean = sum_value / float(count)
        variance = max((sum_square / float(count)) - (mean * mean), 0.0)
        std = math.sqrt(variance)
        if std < 1e-6:
            std = 1.0

        for start in range(0, flat.size, chunk_size):
            chunk = flat[start:start + chunk_size]
            mask = chunk != 0
            if not np.any(mask):
                continue
            chunk_values = chunk[mask].astype(np.float32, copy=False)
            chunk[mask] = ((chunk_values - mean) / std).astype(np.float32, copy=False)

        result[c] = values
    return result


def build_label_map(channel_masks: np.ndarray) -> np.ndarray:
    if channel_masks.shape[0] != 3:
        raise ValueError(f"BraTS label map expects 3 channels, got {channel_masks.shape}")
    tc = channel_masks[0] > 0
    wt = channel_masks[1] > 0
    et = channel_masks[2] > 0
    label = np.zeros(channel_masks.shape[1:], dtype=np.uint8)
    label[wt] = 2
    label[tc] = 1
    label[et] = 4
    return label


def resolve_task_mode(requested: str, output_channels: int, metadata: dict) -> str:
    if requested != "auto":
        return requested
    task_name = str(metadata.get("task") or "").lower()
    bundle_name = str(metadata.get("name") or "").lower()
    if output_channels == 3 and ("brats" in task_name or "brats" in bundle_name):
        return "brats-multilabel"
    return "multiclass"


def build_multiclass_label_map(class_probs: np.ndarray) -> np.ndarray:
    if class_probs.ndim != 4:
        raise ValueError(f"Expected C,D,H,W probabilities, got {class_probs.shape}")
    return np.argmax(class_probs, axis=0).astype(np.uint16, copy=False)


def compute_ov0_patch_starts(axis_size: int, roi_size: int) -> list[int]:
    axis_size = int(axis_size)
    roi_size = max(1, min(int(roi_size), axis_size))
    if axis_size <= roi_size:
        return [0]
    starts = list(range(0, axis_size - roi_size + 1, roi_size))
    last_start = axis_size - roi_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def compute_ov0_owned_interval(
    starts: list[int],
    axis_index: int,
    roi_size: int,
    axis_size: int,
) -> tuple[int, int]:
    start = int(starts[axis_index])
    owned_start = 0 if axis_index <= 0 else int((starts[axis_index - 1] + start + roi_size) // 2)
    owned_end = (
        int(axis_size)
        if axis_index >= len(starts) - 1
        else int((start + starts[axis_index + 1] + roi_size) // 2)
    )
    owned_start = max(owned_start, start)
    owned_end = min(owned_end, start + roi_size, int(axis_size))
    if owned_end <= owned_start:
        raise RuntimeError(
            f"Invalid ov0 owned interval: axis_size={axis_size}, roi_size={roi_size}, starts={starts}, index={axis_index}"
        )
    return owned_start, owned_end


def estimate_tensor_bytes(shape: Iterable[int], dtype: np.dtype | str = np.float32) -> int:
    count = 1
    for dim in shape:
        count *= max(1, int(dim))
    return int(count * np.dtype(dtype).itemsize)


def infer_multiclass_labelmap_patchwise_ov0(
    network,
    input_tensor: torch.Tensor,
    roi_size: tuple[int, int, int],
    output_channels: int,
    preserve_labels: set[int] | None,
    yield_every_patches: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    if input_tensor.ndim != 5 or int(input_tensor.shape[0]) != 1:
        raise ValueError(f"Expected input tensor with shape [1,C,D,H,W], got {tuple(int(v) for v in input_tensor.shape)}")

    spatial_shape = tuple(int(v) for v in input_tensor.shape[2:])
    roi_dhw = tuple(max(1, min(int(roi_size[i]), spatial_shape[i])) for i in range(3))
    start_lists = [compute_ov0_patch_starts(spatial_shape[i], roi_dhw[i]) for i in range(3)]
    patch_grid_dhw = [len(start_lists[0]), len(start_lists[1]), len(start_lists[2])]
    patch_count = int(patch_grid_dhw[0] * patch_grid_dhw[1] * patch_grid_dhw[2])
    label_map = np.zeros(spatial_shape, dtype=np.uint16 if output_channels > 255 else np.uint8)

    peak_rss_mb = 0.0
    peak_private_mb = 0.0
    peak_cuda_allocated_mb = 0.0
    process = psutil.Process(os.getpid())
    patch_ms_total = 0.0
    patch_ms_max = 0.0
    patch_index = 0

    for z_index, z_start in enumerate(start_lists[0]):
        z_owned_start, z_owned_end = compute_ov0_owned_interval(start_lists[0], z_index, roi_dhw[0], spatial_shape[0])
        z_slice = slice(int(z_start), int(z_start + roi_dhw[0]))
        z_owned_global = slice(z_owned_start, z_owned_end)
        z_owned_local = slice(z_owned_start - int(z_start), z_owned_end - int(z_start))

        for y_index, y_start in enumerate(start_lists[1]):
            y_owned_start, y_owned_end = compute_ov0_owned_interval(start_lists[1], y_index, roi_dhw[1], spatial_shape[1])
            y_slice = slice(int(y_start), int(y_start + roi_dhw[1]))
            y_owned_global = slice(y_owned_start, y_owned_end)
            y_owned_local = slice(y_owned_start - int(y_start), y_owned_end - int(y_start))

            for x_index, x_start in enumerate(start_lists[2]):
                x_owned_start, x_owned_end = compute_ov0_owned_interval(start_lists[2], x_index, roi_dhw[2], spatial_shape[2])
                x_slice = slice(int(x_start), int(x_start + roi_dhw[2]))
                x_owned_global = slice(x_owned_start, x_owned_end)
                x_owned_local = slice(x_owned_start - int(x_start), x_owned_end - int(x_start))

                patch_t0 = time.perf_counter()
                logits_patch = network(input_tensor[:, :, z_slice, y_slice, x_slice])
                patch_shape = tuple(int(v) for v in logits_patch.shape)
                expected_patch_shape = (1, int(output_channels), int(roi_dhw[0]), int(roi_dhw[1]), int(roi_dhw[2]))
                if patch_shape != expected_patch_shape:
                    raise RuntimeError(
                        f"Unexpected patch output shape: expected {expected_patch_shape}, got {patch_shape}"
                    )

                patch_labels = torch.argmax(logits_patch, dim=1)[0]
                if preserve_labels:
                    preserve_mask = torch.zeros_like(patch_labels, dtype=torch.bool)
                    for label_value in preserve_labels:
                        preserve_mask |= patch_labels == int(label_value)
                    patch_labels = torch.where(
                        preserve_mask,
                        patch_labels,
                        torch.zeros_like(patch_labels),
                    )

                owned_patch = patch_labels[z_owned_local, y_owned_local, x_owned_local]
                owned_np = owned_patch.detach().cpu().numpy().astype(label_map.dtype, copy=False)
                label_map[z_owned_global, y_owned_global, x_owned_global] = owned_np

                patch_elapsed_ms = (time.perf_counter() - patch_t0) * 1000.0
                patch_ms_total += patch_elapsed_ms
                patch_ms_max = max(patch_ms_max, patch_elapsed_ms)
                patch_index += 1

                memory = process.memory_info()
                peak_rss_mb = max(peak_rss_mb, float(memory.rss / (1024 * 1024)))
                if hasattr(memory, "private"):
                    peak_private_mb = max(peak_private_mb, float(memory.private / (1024 * 1024)))
                if torch.cuda.is_available() and input_tensor.device.type == "cuda":
                    peak_cuda_allocated_mb = max(
                        peak_cuda_allocated_mb,
                        float(torch.cuda.memory_allocated(input_tensor.device) / (1024 * 1024)),
                    )

                del owned_np, owned_patch, patch_labels, logits_patch
                if torch.cuda.is_available() and input_tensor.device.type == "cuda":
                    torch.cuda.empty_cache()
                if yield_every_patches > 0 and (patch_index % yield_every_patches) == 0:
                    gc.collect()
                    time.sleep(0.01)

    return label_map, {
        "patch_count": patch_count,
        "patch_grid_dhw": patch_grid_dhw,
        "roi_size_dhw": list(roi_dhw),
        "ownership_mode": "center-ownership",
        "sw_overlap": 0.0,
        "patch_time_mean_ms": float(patch_ms_total / patch_count) if patch_count > 0 else 0.0,
        "patch_time_max_ms": float(patch_ms_max),
        "peak_process_rss_mb": float(peak_rss_mb),
        "peak_process_private_mb": float(peak_private_mb) if peak_private_mb > 0.0 else None,
        "peak_cuda_allocated_mb": float(peak_cuda_allocated_mb) if peak_cuda_allocated_mb > 0.0 else None,
        "preserve_labels": sorted(int(v) for v in preserve_labels) if preserve_labels else None,
    }


def save_nifti(path: Path, array: np.ndarray, reference: VolumeData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = reference.affine if reference.affine is not None else np.eye(4, dtype=np.float32)
    image = nib.Nifti1Image(np.ascontiguousarray(array), affine)
    if reference.spacing is not None:
        image.header.set_zooms(tuple(float(v) for v in reference.spacing[:3]))
    nib.save(image, str(path))


def numpy_to_nrrd_type(dtype: np.dtype) -> str:
    normalized = np.dtype(dtype)
    mapping = {
        np.dtype(np.int8): "int8",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.int16): "int16",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.int32): "int32",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype for NRRD export: {dtype}")
    return mapping[normalized]


def save_nrrd(path: Path, array: np.ndarray, reference: VolumeData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.nrrd_header or {}
    array = np.ascontiguousarray(array)
    dtype = np.dtype(array.dtype).newbyteorder("<")
    payload = np.asarray(array, dtype=dtype).ravel(order="F").tobytes(order="C")

    lines = [
        "NRRD0005",
        "# Generated by MonaiNcnnBaseline.py",
        f"type: {numpy_to_nrrd_type(dtype)}",
        "dimension: 3",
        f"sizes: {array.shape[0]} {array.shape[1]} {array.shape[2]}",
    ]
    if "space" in header:
        lines.append(f"space: {header['space']}")
    if "space directions" in header:
        lines.append(f"space directions: {header['space directions']}")
    elif "spacings" in header:
        lines.append(f"spacings: {header['spacings']}")
    elif reference.spacing is not None:
        lines.append("spacings: " + " ".join(str(float(v)) for v in reference.spacing))
    if "kinds" in header:
        lines.append(f"kinds: {header['kinds']}")
    else:
        lines.append("kinds: domain domain domain")
    if "space origin" in header:
        lines.append(f"space origin: {header['space origin']}")
    if "space units" in header:
        lines.append(f"space units: {header['space units']}")
    lines.extend(
        [
            "endian: little",
            "encoding: gzip",
            "",
            "",
        ]
    )

    with path.open("wb") as f:
        f.write("\n".join(lines).encode("ascii"))
        f.write(gzip.compress(payload))


def save_original_format_volume(path: Path, array: np.ndarray, reference: VolumeData) -> None:
    if reference.source_format == "nrrd":
        save_nrrd(path, np.asarray(array), reference)
        return
    if reference.source_format == "nifti":
        save_nifti(path, np.asarray(array), reference)
        return
    raise ValueError(f"Unsupported reference format for export: {reference.source_format}")


def save_array_bin(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(path)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_label_values(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_label_subset_specs(text: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for raw_part in text.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        name_part, sep, values_part = part.partition("=")
        if not sep:
            raise ValueError(
                "Each --label-subsets entry must use name=value1,value2 format; got: " + part
            )
        subset_name = make_safe_name(name_part, "label_subset")
        subset_values = parse_label_values(values_part)
        if not subset_values:
            raise ValueError("Label subset has no values: " + part)
        specs.append({"label_name": subset_name, "label_values": subset_values})
    return specs


def make_safe_name(text: str, fallback: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip()).strip("_")
    return safe or fallback


def should_use_patchwise_ov0_multiclass(
    infer_mode: str,
    task_mode: str,
    save_multiclass_logits: bool,
    save_multiclass_probs: bool,
    sw_overlap: float,
) -> bool:
    return (
        infer_mode == "monai-sliding-window"
        and task_mode == "multiclass"
        and not save_multiclass_logits
        and not save_multiclass_probs
        and abs(float(sw_overlap)) <= 1e-8
    )


def describe_array(name: str, array: np.ndarray) -> dict:
    flat = np.asarray(array).astype(np.float32, copy=False).reshape(-1)
    finite = np.isfinite(flat)
    finite_values = flat[finite]
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(flat.size),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(flat).sum()),
        "inf": int(np.isinf(flat).sum()),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "mean": float(finite_values.mean()) if finite_values.size else None,
    }


def describe_tensor_shape(name: str, shape: Iterable[int], dtype: str, count: int) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(v) for v in shape],
        "dtype": dtype,
        "count": int(count),
        "finite": None,
        "nan": None,
        "inf": None,
        "min": None,
        "max": None,
        "mean": None,
    }


def restore_to_original_shape(array: np.ndarray, original_shape: tuple[int, int, int], pad_value: int | float = 0) -> np.ndarray:
    if len(original_shape) != 3:
        raise ValueError(f"Expected 3D original shape, got {original_shape}")
    return center_crop_or_pad(array, original_shape, pad_value=pad_value)


def build_resource_stats(device: str) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    stats: dict[str, Any] = {
        "process_private_mb": float(memory.private / (1024 * 1024)) if hasattr(memory, "private") else None,
        "process_rss_mb": float(memory.rss / (1024 * 1024)),
        "process_vms_mb": float(memory.vms / (1024 * 1024)),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device": str(device),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda_memory": None,
        "python_temp_rt_count": 0,
        "python_compute_buffer_count": 0,
        "note": "Python baseline does not allocate Unity RenderTexture or ComputeBuffer objects.",
    }

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        current = torch.cuda.current_device()
        stats["torch_cuda_memory"] = {
            "device_index": current,
            "device_name": torch.cuda.get_device_name(current),
            "allocated_mb": float(torch.cuda.memory_allocated(current) / (1024 * 1024)),
            "reserved_mb": float(torch.cuda.memory_reserved(current) / (1024 * 1024)),
            "max_allocated_mb": float(torch.cuda.max_memory_allocated(current) / (1024 * 1024)),
            "max_reserved_mb": float(torch.cuda.max_memory_reserved(current) / (1024 * 1024)),
        }

    return stats


def capture_resource_snapshot(stage: str, device: str) -> dict[str, Any]:
    snapshot = build_resource_stats(device)
    snapshot["stage"] = stage
    return snapshot


def apply_torch_runtime_limits(args: argparse.Namespace) -> dict[str, Any]:
    requested_threads = int(args.torch_threads) if int(args.torch_threads) > 0 else None
    requested_interop_threads = int(args.torch_interop_threads) if int(args.torch_interop_threads) > 0 else None
    if args.low_power_mode:
        if requested_threads is None:
            requested_threads = 1
        if requested_interop_threads is None:
            requested_interop_threads = 1
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")

    if requested_threads is not None:
        torch.set_num_threads(max(1, int(requested_threads)))
    if requested_interop_threads is not None:
        torch.set_num_interop_threads(max(1, int(requested_interop_threads)))

    return {
        "low_power_mode": bool(args.low_power_mode),
        "requested_torch_threads": requested_threads,
        "requested_torch_interop_threads": requested_interop_threads,
        "effective_torch_threads": int(torch.get_num_threads()),
        "effective_torch_interop_threads": int(torch.get_num_interop_threads()),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads_env": os.environ.get("NUMEXPR_NUM_THREADS"),
        "omp_wait_policy_env": os.environ.get("OMP_WAIT_POLICY"),
        "kmp_blocktime_env": os.environ.get("KMP_BLOCKTIME"),
    }


def main() -> int:
    args = parse_args()
    runtime_limits = apply_torch_runtime_limits(args)
    bundle_root = Path(args.bundle_root).resolve()
    output_root = Path(args.output_dir).resolve()
    input_paths = [Path(path).resolve() for path in args.input]
    target_shape = parse_shape(args.target_shape)
    roi_size = parse_shape(args.roi_size)
    total_t0 = time.perf_counter()
    timings: dict[str, int] = {}
    resource_snapshots: list[dict[str, Any]] = []

    load_model_t0 = time.perf_counter()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    network, inference_config, metadata = resolve_model(bundle_root)
    timings["load_model_ms"] = int(round((time.perf_counter() - load_model_t0) * 1000.0))
    resource_snapshots.append(capture_resource_snapshot("after_load_model", args.device))

    required_channels = int(inference_config["network_def"].get("in_channels", 1))
    output_channels = int(inference_config["network_def"].get("out_channels", 1))

    load_inputs_t0 = time.perf_counter()
    volumes = [load_volume(path) for path in input_paths]
    stacked, input_entries = align_modalities(volumes)
    timings["load_inputs_ms"] = int(round((time.perf_counter() - load_inputs_t0) * 1000.0))
    resource_snapshots.append(capture_resource_snapshot("after_load_inputs", args.device))

    prepare_input_t0 = time.perf_counter()
    filled = fill_channels(stacked, required_channels, args.channel_fill)
    if args.normalize_nonzero:
        filled = normalize_nonzero_per_channel(filled)
    prepared = center_crop_or_pad(filled, target_shape) if args.infer_mode == "ncnn-fixed" else filled
    timings["prepare_input_ms"] = int(round((time.perf_counter() - prepare_input_t0) * 1000.0))
    resource_snapshots.append(capture_resource_snapshot("after_prepare_input", args.device))

    subset_specs = parse_label_subset_specs(args.label_subsets)
    if args.label_subset_values.strip():
        subset_specs.append(
            {
                "label_name": make_safe_name(args.label_subset_name or "label_subset", "label_subset"),
                "label_values": parse_label_values(args.label_subset_values),
            }
        )

    case_name = args.case_name.strip() if args.case_name else input_paths[0].stem.replace(".nii", "")
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    network = network.to(device)
    inference_t0 = time.perf_counter()
    patchwise_stats: dict[str, Any] | None = None
    with torch.no_grad():
        input_tensor = torch.from_numpy(prepared).unsqueeze(0).to(device=device, dtype=torch.float32)
        task_mode = resolve_task_mode(args.task_mode, output_channels, metadata)
        if should_use_patchwise_ov0_multiclass(
            args.infer_mode,
            task_mode,
            args.save_multiclass_logits,
            args.save_multiclass_probs,
            args.sw_overlap,
        ):
            logits = None
            label_map = None
            label_map, patchwise_stats = infer_multiclass_labelmap_patchwise_ov0(
                network=network,
                input_tensor=input_tensor,
                roi_size=roi_size,
                output_channels=output_channels,
                preserve_labels=None,
                yield_every_patches=1,
            )
        elif args.infer_mode == "monai-sliding-window":
            from monai.inferers import SlidingWindowInferer

            inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=1,
                overlap=float(args.sw_overlap),
            )
            logits = inferer(input_tensor, network)
            label_map = None
        else:
            logits = network(input_tensor)
            label_map = None
    timings["inference_ms"] = int(round((time.perf_counter() - inference_t0) * 1000.0))
    resource_snapshots.append(capture_resource_snapshot("after_inference", args.device))

    postprocess_t0 = time.perf_counter()
    if logits is not None:
        logits_shape = tuple(int(v) for v in logits.shape)
        if logits_shape[1] != output_channels:
            raise RuntimeError(f"Unexpected output channel count: expected {output_channels}, got {logits_shape}")
        logits_dtype_name = str(logits.dtype)
        logits_element_count = int(torch.numel(logits[0]))
    else:
        logits_shape = (
            1,
            int(output_channels),
            int(input_tensor.shape[2]),
            int(input_tensor.shape[3]),
            int(input_tensor.shape[4]),
        )
        logits_dtype_name = "patchwise-argmax-only"
        logits_element_count = int(output_channels * input_tensor.shape[2] * input_tensor.shape[3] * input_tensor.shape[4])

    logits_np = None
    if task_mode == "brats-multilabel":
        if logits is None:
            raise RuntimeError("Patchwise ov0 mode currently supports multiclass tasks only.")
        logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        probs = torch.sigmoid(logits)
        masks = (probs >= args.threshold).to(torch.uint8)
        probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
        masks_np = masks.detach().cpu().numpy().astype(np.uint8, copy=False)
        label_map = build_label_map(masks_np[0])
    else:
        probs_np = None
        if logits is not None and args.save_multiclass_logits:
            logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        if logits is not None and args.save_multiclass_probs:
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
            label_map = build_multiclass_label_map(probs_np[0])
        elif logits is not None:
            label_map = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint16, copy=False)
        elif label_map is None:
            raise RuntimeError("Expected patchwise multiclass label_map to be available.")
        masks_np = None

    original_shape = tuple(int(v) for v in volumes[0].data.shape)
    if tuple(int(v) for v in label_map.shape) == original_shape:
        restored_label_map = label_map.copy()
        restored_masks = masks_np[0].astype(np.uint8, copy=False) if masks_np is not None else None
    else:
        restored_label_map = restore_to_original_shape(label_map, original_shape, pad_value=0)
        restored_masks = restore_to_original_shape(masks_np[0], original_shape, pad_value=0).astype(np.uint8, copy=False) if masks_np is not None else None

    if restored_label_map.dtype == np.uint8 and output_channels > 255:
        restored_label_map = restored_label_map.astype(np.uint16, copy=False)
    elif output_channels > 255:
        restored_label_map = restored_label_map.astype(np.uint16, copy=False)
    else:
        restored_label_map = restored_label_map.astype(np.uint8, copy=False)
    timings["postprocess_ms"] = int(round((time.perf_counter() - postprocess_t0) * 1000.0))
    resource_snapshots.append(capture_resource_snapshot("after_postprocess", args.device))

    manifest = {
        "bundle_root": str(bundle_root),
        "case_name": case_name,
        "infer_mode": args.infer_mode,
        "task": metadata.get("task"),
        "description": metadata.get("description"),
        "label_classes": metadata.get("label_classes"),
        "task_mode": task_mode,
        "inputs": input_entries,
        "channel_fill": args.channel_fill,
        "normalize_nonzero": bool(args.normalize_nonzero),
        "threshold": float(args.threshold),
        "target_shape_dhw": list(target_shape),
        "sliding_window_roi_dhw": list(roi_size),
        "sliding_window_overlap": float(args.sw_overlap),
        "low_power_runtime": runtime_limits,
        "model_input_shape_ncdhw": list(input_tensor.shape),
        "original_volume_shape_dhw": list(original_shape),
        "unity_buffer_shape_whdc": [prepared.shape[3], prepared.shape[2], prepared.shape[1], prepared.shape[0]],
        "model_output_shape_ncdhw": list(logits_shape),
        "unity_output_shape_whdc": [logits_shape[4], logits_shape[3], logits_shape[2], logits_shape[1]],
        "patchwise_multiclass_ov0": patchwise_stats,
        "files": {
            "input_tensor_f32_bin": "input_tensor_ncdhw_f32.bin",
            "logits_f32_bin": "logits_ncdhw_f32.bin" if logits_np is not None else None,
            "probs_f32_bin": "probs_ncdhw_f32.bin" if probs_np is not None else None,
            "masks_u8_bin": "masks_ncdhw_u8.bin" if masks_np is not None else None,
            "labelmap_u8_bin": "labelmap_dhw_u8.bin",
            "restored_labelmap": None,
            "restored_masks_dir": "restored_masks",
        },
        "stats": {
            "input_tensor": describe_array("input_tensor", input_tensor[0].detach().cpu().numpy()),
            "logits": describe_array("logits", logits_np[0]) if logits_np is not None else describe_tensor_shape("logits", logits_shape[1:], logits_dtype_name, logits_element_count),
            "probs": describe_array("probs", probs_np[0]) if probs_np is not None else None,
            "labelmap": describe_array("labelmap", label_map),
            "restored_labelmap": describe_array("restored_labelmap", restored_label_map),
        },
        "timings_ms": timings,
        "resource_stats": None,
        "resource_snapshots": resource_snapshots,
    }
    if masks_np is not None:
        manifest["stats"]["masks"] = describe_array("masks", masks_np[0])
    if restored_masks is not None:
        manifest["stats"]["restored_masks"] = describe_array("restored_masks", restored_masks)

    export_t0 = time.perf_counter()
    save_array_bin(case_dir / "input_tensor_ncdhw_f32.bin", input_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False))
    if logits_np is not None:
        save_array_bin(case_dir / "logits_ncdhw_f32.bin", logits_np[0])
    if probs_np is not None:
        save_array_bin(case_dir / "probs_ncdhw_f32.bin", probs_np[0])
    if masks_np is not None:
        save_array_bin(case_dir / "masks_ncdhw_u8.bin", masks_np[0])
    save_array_bin(case_dir / "labelmap_dhw_u8.bin", label_map)
    np.save(case_dir / "input_tensor_ncdhw_f32.npy", input_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False))
    if logits_np is not None:
        np.save(case_dir / "logits_ncdhw_f32.npy", logits_np[0])
    if probs_np is not None:
        np.save(case_dir / "probs_ncdhw_f32.npy", probs_np[0])
    if masks_np is not None:
        np.save(case_dir / "masks_ncdhw_u8.npy", masks_np[0])
    np.save(case_dir / "labelmap_dhw_u8.npy", label_map)

    if args.save_restored:
        restored_masks_dir = case_dir / "restored_masks"
        restored_masks_dir.mkdir(parents=True, exist_ok=True)
        reference_volume = volumes[0]
        if reference_volume.source_format == "nrrd":
            labelmap_export_name = "labelmap_restored.nrrd"
            mask_ext = ".nrrd"
        else:
            labelmap_export_name = "labelmap_restored.nii.gz"
            mask_ext = ".nii.gz"

        save_original_format_volume(case_dir / labelmap_export_name, restored_label_map, reference_volume)
        manifest["files"]["restored_labelmap"] = labelmap_export_name

        if restored_masks is not None and args.save_channel_masks and restored_masks.shape[0] <= max(1, args.max_saved_mask_channels):
            label_names = metadata.get("network_data_format", {}).get("outputs", {}).get("pred", {}).get("channel_def", {})
            restored_mask_files: list[str] = []
            for channel_index in range(restored_masks.shape[0]):
                channel_name = str(label_names.get(str(channel_index), f"class_{channel_index}"))
                safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in channel_name).strip("_") or f"class_{channel_index}"
                file_name = f"{channel_index:03d}_{safe_name}{mask_ext}"
                save_original_format_volume(restored_masks_dir / file_name, restored_masks[channel_index], reference_volume)
                restored_mask_files.append(f"restored_masks/{file_name}")
            manifest["files"]["restored_mask_channels"] = restored_mask_files

        if subset_specs:
            subset_dir = case_dir / "label_subsets"
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_ext = ".nrrd" if reference_volume.source_format == "nrrd" else ".nii.gz"
            manifest_subsets: list[dict[str, Any]] = []
            for subset_spec in subset_specs:
                subset_name = str(subset_spec["label_name"])
                subset_values = [int(value) for value in subset_spec["label_values"]]
                subset_processed = np.isin(label_map, subset_values).astype(np.uint8, copy=False)
                subset_restored = np.isin(restored_label_map, subset_values).astype(np.uint8, copy=False)
                subset_labelmap = np.where(
                    np.isin(label_map, subset_values),
                    label_map,
                    np.zeros_like(label_map),
                ).astype(restored_label_map.dtype if restored_label_map.dtype.itemsize >= label_map.dtype.itemsize else label_map.dtype, copy=False)
                subset_restored_labelmap = np.where(
                    np.isin(restored_label_map, subset_values),
                    restored_label_map,
                    np.zeros_like(restored_label_map),
                ).astype(restored_label_map.dtype, copy=False)
                processed_subset_name = f"{subset_name}_mask_dhw_u8.bin"
                save_array_bin(subset_dir / processed_subset_name, subset_processed)
                processed_subset_labelmap_name = f"{subset_name}_labelmap_dhw_u16.bin"
                save_array_bin(subset_dir / processed_subset_labelmap_name, subset_labelmap.astype(np.uint16, copy=False))
                restored_subset_name = f"{subset_name}_mask_restored{subset_ext}"
                save_original_format_volume(subset_dir / restored_subset_name, subset_restored, reference_volume)
                restored_subset_labelmap_name = f"{subset_name}_labelmap_restored{subset_ext}"
                save_original_format_volume(
                    subset_dir / restored_subset_labelmap_name,
                    subset_restored_labelmap.astype(np.uint16, copy=False),
                    reference_volume,
                )
                manifest_subsets.append(
                    {
                        "label_values": subset_values,
                        "label_name": subset_name,
                        "processed_binary_mask": f"label_subsets/{processed_subset_name}",
                        "restored_binary_mask": f"label_subsets/{restored_subset_name}",
                        "processed_labelmap": f"label_subsets/{processed_subset_labelmap_name}",
                        "restored_labelmap": f"label_subsets/{restored_subset_labelmap_name}",
                    }
                )
            manifest["label_subsets"] = manifest_subsets
            if manifest_subsets:
                manifest["label_subset"] = manifest_subsets[0]

    timings["export_ms"] = int(round((time.perf_counter() - export_t0) * 1000.0))
    timings["total_elapsed_ms"] = int(round((time.perf_counter() - total_t0) * 1000.0))
    resource_snapshots.append(capture_resource_snapshot("after_export", args.device))
    resource_stats = build_resource_stats(args.device)
    manifest["timings_ms"] = timings
    manifest["resource_stats"] = resource_stats
    manifest["resource_snapshots"] = resource_snapshots
    save_json(case_dir / "baseline_manifest.json", manifest)
    save_json(case_dir / "timings.json", timings)
    save_json(case_dir / "resource_stats.json", resource_stats)
    save_json(case_dir / "resource_snapshots.json", {"snapshots": resource_snapshots})

    if args.save_nifti and volumes[0].affine is not None:
        nii = nib.Nifti1Image(label_map.astype(np.uint8), volumes[0].affine)
        nib.save(nii, str(case_dir / "labelmap_dhw_u8.nii.gz"))

    summary_lines = [
        f"case={case_name}",
        f"bundle={bundle_root}",
        f"device={device}",
        f"input_count={len(input_paths)}",
        f"infer_mode={args.infer_mode}",
        f"channel_fill={args.channel_fill}",
        f"target_shape_dhw={target_shape[0]},{target_shape[1]},{target_shape[2]}",
        f"sliding_window_roi_dhw={roi_size[0]},{roi_size[1]},{roi_size[2]}",
        f"sliding_window_overlap={args.sw_overlap}",
        f"original_shape_dhw={original_shape[0]},{original_shape[1]},{original_shape[2]}",
        f"input_tensor_shape={tuple(input_tensor.shape)}",
        f"logits_shape={tuple(logits_shape)}",
        f"threshold={args.threshold}",
        f"task_mode={task_mode}",
        f"output_channels={output_channels}",
        f"save_multiclass_logits={bool(args.save_multiclass_logits)}",
        f"save_multiclass_probs={bool(args.save_multiclass_probs)}",
        f"patchwise_multiclass_ov0={bool(patchwise_stats is not None)}",
        f"timings_json={case_dir / 'timings.json'}",
        f"resource_stats_json={case_dir / 'resource_stats.json'}",
        f"resource_snapshots_json={case_dir / 'resource_snapshots.json'}",
    ]
    if task_mode == "brats-multilabel":
        summary_lines.append("labels=0:background,1:tumor core,2:whole tumor,4:enhancing tumor")
        summary_lines.append("warning=brats_mri_segmentation is a tumor-subregion model, not a skull/ventricle model")
    save_summary(case_dir / "summary.txt", summary_lines)

    print(str(case_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
