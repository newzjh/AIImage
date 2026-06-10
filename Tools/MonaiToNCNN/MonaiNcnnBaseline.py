from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
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
    for c in range(result.shape[0]):
        values = result[c]
        mask = values != 0
        if not np.any(mask):
            continue
        nonzero = values[mask]
        mean = float(nonzero.mean())
        std = float(nonzero.std())
        if std < 1e-6:
            std = 1.0
        values = values.copy()
        values[mask] = (nonzero - mean) / std
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


def restore_to_original_shape(array: np.ndarray, original_shape: tuple[int, int, int], pad_value: int | float = 0) -> np.ndarray:
    if len(original_shape) != 3:
        raise ValueError(f"Expected 3D original shape, got {original_shape}")
    return center_crop_or_pad(array, original_shape, pad_value=pad_value)


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    output_root = Path(args.output_dir).resolve()
    input_paths = [Path(path).resolve() for path in args.input]
    target_shape = parse_shape(args.target_shape)
    roi_size = parse_shape(args.roi_size)

    network, inference_config, metadata = resolve_model(bundle_root)
    required_channels = int(inference_config["network_def"].get("in_channels", 1))
    output_channels = int(inference_config["network_def"].get("out_channels", 1))

    volumes = [load_volume(path) for path in input_paths]
    stacked, input_entries = align_modalities(volumes)
    filled = fill_channels(stacked, required_channels, args.channel_fill)
    if args.normalize_nonzero:
        filled = normalize_nonzero_per_channel(filled)

    case_name = args.case_name.strip() if args.case_name else input_paths[0].stem.replace(".nii", "")
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    network = network.to(device)
    prepared = center_crop_or_pad(filled, target_shape) if args.infer_mode == "ncnn-fixed" else filled
    with torch.no_grad():
        input_tensor = torch.from_numpy(prepared).unsqueeze(0).to(device=device, dtype=torch.float32)
        if args.infer_mode == "monai-sliding-window":
            from monai.inferers import SlidingWindowInferer

            inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=1,
                overlap=float(args.sw_overlap),
            )
            logits = inferer(input_tensor, network)
        else:
            logits = network(input_tensor)

    logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    if logits_np.shape[1] != output_channels:
        raise RuntimeError(f"Unexpected output channel count: expected {output_channels}, got {logits_np.shape}")
    task_mode = resolve_task_mode(args.task_mode, output_channels, metadata)

    if task_mode == "brats-multilabel":
        probs = torch.sigmoid(logits)
        masks = (probs >= args.threshold).to(torch.uint8)
        probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
        masks_np = masks.detach().cpu().numpy().astype(np.uint8, copy=False)
        label_map = build_label_map(masks_np[0])
    else:
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
        label_map = build_multiclass_label_map(probs_np[0])
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
        "model_input_shape_ncdhw": list(input_tensor.shape),
        "original_volume_shape_dhw": list(original_shape),
        "unity_buffer_shape_whdc": [prepared.shape[3], prepared.shape[2], prepared.shape[1], prepared.shape[0]],
        "model_output_shape_ncdhw": list(logits_np.shape),
        "unity_output_shape_whdc": [logits_np.shape[4], logits_np.shape[3], logits_np.shape[2], logits_np.shape[1]],
        "files": {
            "input_tensor_f32_bin": "input_tensor_ncdhw_f32.bin",
            "logits_f32_bin": "logits_ncdhw_f32.bin",
            "probs_f32_bin": "probs_ncdhw_f32.bin",
            "masks_u8_bin": "masks_ncdhw_u8.bin" if masks_np is not None else None,
            "labelmap_u8_bin": "labelmap_dhw_u8.bin",
            "restored_labelmap": None,
            "restored_masks_dir": "restored_masks",
        },
        "stats": {
            "input_tensor": describe_array("input_tensor", input_tensor[0].detach().cpu().numpy()),
            "logits": describe_array("logits", logits_np[0]),
            "probs": describe_array("probs", probs_np[0]),
            "labelmap": describe_array("labelmap", label_map),
            "restored_labelmap": describe_array("restored_labelmap", restored_label_map),
        },
    }
    if masks_np is not None:
        manifest["stats"]["masks"] = describe_array("masks", masks_np[0])
    if restored_masks is not None:
        manifest["stats"]["restored_masks"] = describe_array("restored_masks", restored_masks)

    save_array_bin(case_dir / "input_tensor_ncdhw_f32.bin", input_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False))
    save_array_bin(case_dir / "logits_ncdhw_f32.bin", logits_np[0])
    save_array_bin(case_dir / "probs_ncdhw_f32.bin", probs_np[0])
    if masks_np is not None:
        save_array_bin(case_dir / "masks_ncdhw_u8.bin", masks_np[0])
    save_array_bin(case_dir / "labelmap_dhw_u8.bin", label_map)
    np.save(case_dir / "input_tensor_ncdhw_f32.npy", input_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False))
    np.save(case_dir / "logits_ncdhw_f32.npy", logits_np[0])
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

    save_json(case_dir / "baseline_manifest.json", manifest)

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
        f"logits_shape={tuple(logits_np.shape)}",
        f"threshold={args.threshold}",
        f"task_mode={task_mode}",
        f"output_channels={output_channels}",
    ]
    if task_mode == "brats-multilabel":
        summary_lines.append("labels=0:background,1:tumor core,2:whole tumor,4:enhancing tumor")
        summary_lines.append("warning=brats_mri_segmentation is a tumor-subregion model, not a skull/ventricle model")
    save_summary(case_dir / "summary.txt", summary_lines)

    print(str(case_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
