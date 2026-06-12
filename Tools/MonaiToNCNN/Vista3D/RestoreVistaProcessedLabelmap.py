from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a processed Vista labelmap back to the original NIfTI reference shape.")
    parser.add_argument("--labelmap-bin", required=True, help="Processed DHW labelmap binary path (.bin, uint16).")
    parser.add_argument("--baseline-manifest", required=True, help="Official Vista baseline manifest JSON.")
    parser.add_argument("--output", required=True, help="Output restored NIfTI path (.nii.gz).")
    parser.add_argument("--dtype", default="uint16", choices=("uint8", "uint16"), help="Input binary element type.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def center_crop_or_pad(array: np.ndarray, target_shape: tuple[int, int, int], pad_value: int = 0) -> np.ndarray:
    result = np.full(target_shape, pad_value, dtype=array.dtype)

    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for src_size, dst_size in zip(array.shape, target_shape):
        copy_size = min(int(src_size), int(dst_size))
        src_start = max(0, (int(src_size) - copy_size) // 2)
        dst_start = max(0, (int(dst_size) - copy_size) // 2)
        src_slices.append(slice(src_start, src_start + copy_size))
        dst_slices.append(slice(dst_start, dst_start + copy_size))

    result[tuple(dst_slices)] = array[tuple(src_slices)]
    return result


def main() -> int:
    args = parse_args()
    labelmap_bin = Path(args.labelmap_bin).resolve()
    manifest_path = Path(args.baseline_manifest).resolve()
    output_path = Path(args.output).resolve()

    manifest = load_json(manifest_path)
    processed_shape = tuple(int(v) for v in manifest["processed_volume_shape_dhw"])
    original_shape = tuple(int(v) for v in manifest["original_volume_shape_dhw"])
    input_path = Path(manifest["inputs"][0]["path"]).resolve()

    dtype = np.uint16 if args.dtype == "uint16" else np.uint8
    raw = np.fromfile(labelmap_bin, dtype=dtype)
    expected = int(np.prod(processed_shape))
    if raw.size != expected:
        raise ValueError(f"Processed labelmap voxel count mismatch: expected {expected}, got {raw.size}")

    processed = raw.reshape(processed_shape)
    restored = center_crop_or_pad(processed, original_shape, pad_value=0).astype(np.uint16, copy=False)

    reference = nib.load(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ascontiguousarray(restored), reference.affine, reference.header)
    image.set_data_dtype(np.uint16)
    nib.save(image, str(output_path))

    sidecar = {
        "labelmap_bin": str(labelmap_bin),
        "baseline_manifest": str(manifest_path),
        "input_reference": str(input_path),
        "processed_shape_dhw": list(processed_shape),
        "restored_shape_dhw": list(original_shape),
        "output": str(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
