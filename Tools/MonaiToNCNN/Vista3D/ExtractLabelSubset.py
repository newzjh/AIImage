from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

try:
    import nrrd  # type: ignore
except ImportError:
    nrrd = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a binary label subset from a restored labelmap.")
    parser.add_argument("--labelmap", required=True, help="Input labelmap path (.nii, .nii.gz, or .nrrd).")
    parser.add_argument("--label-value", required=True, type=int, help="Foreground label value to extract.")
    parser.add_argument("--output", required=True, help="Output binary labelmap path.")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_output_path(labelmap_path: Path, output_path: Path) -> Path:
    lower_name = labelmap_path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return output_path if output_path.name.lower().endswith(".nii.gz") else output_path.with_name(output_path.stem + ".nii.gz")
    if lower_name.endswith(".nii"):
        return output_path if output_path.suffix.lower() == ".nii" else output_path.with_suffix(".nii")
    if lower_name.endswith(".nrrd"):
        return output_path if output_path.suffix.lower() == ".nrrd" else output_path.with_suffix(".nrrd")
    raise ValueError(f"Unsupported labelmap format: {labelmap_path}")


def load_labelmap(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    lower_name = path.name.lower()
    if lower_name.endswith((".nii", ".nii.gz")):
        image = nib.load(str(path))
        array = np.asarray(image.get_fdata())
        return array, {"kind": "nifti", "image": image}
    if lower_name.endswith(".nrrd"):
        if nrrd is None:
            raise RuntimeError("NRRD subset extraction requires the optional 'pynrrd' package.")
        array, header = nrrd.read(str(path))
        return np.asarray(array), {"kind": "nrrd", "header": dict(header)}
    raise ValueError(f"Unsupported labelmap format: {path}")


def save_labelmap(output_path: Path, subset: np.ndarray, meta: dict[str, Any]) -> None:
    kind = str(meta.get("kind") or "")
    if kind == "nifti":
        image = meta["image"]
        out_image = nib.Nifti1Image(np.ascontiguousarray(subset), image.affine, image.header)
        out_image.set_data_dtype(np.uint8)
        nib.save(out_image, str(output_path))
        return
    if kind == "nrrd":
        if nrrd is None:
            raise RuntimeError("NRRD subset extraction requires the optional 'pynrrd' package.")
        header = dict(meta.get("header") or {})
        header["type"] = "uint8"
        header.setdefault("encoding", "gzip")
        nrrd.write(str(output_path), np.ascontiguousarray(subset), header=header)
        return
    raise ValueError(f"Unsupported output metadata kind: {kind}")


def main() -> int:
    args = parse_args()
    labelmap_path = Path(args.labelmap).resolve()
    output_path = normalize_output_path(labelmap_path, Path(args.output).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array, meta = load_labelmap(labelmap_path)
    subset = (array == int(args.label_value)).astype(np.uint8, copy=False)
    save_labelmap(output_path, subset, meta)

    save_json(
        output_path.with_suffix(output_path.suffix + ".json"),
        {
            "labelmap": str(labelmap_path),
            "label_value": int(args.label_value),
            "output": str(output_path),
            "voxel_count": int(subset.sum()),
            "shape": list(subset.shape),
            "format": str(meta.get("kind") or ""),
        },
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
