from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch


class BaselineError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run an official Vista3D fixed-label prompt baseline and dump Unity-friendly tensors.")
    parser.add_argument(
        "--bundle-root",
        default=str(root / "bundle_cache" / "model-zoo-dev" / "models" / "vista3d"),
        help="Path to the Vista3D MONAI bundle root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "manual_test" / "vista3d_ct_philips_heart_baseline"),
        help="Directory where case outputs are written.",
    )
    parser.add_argument(
        "--input",
        default=r"E:\Projects\CTData\sliceexampledata2\CT_Philips\CT_Philips.nii.gz",
        help="Input NIfTI CT path.",
    )
    parser.add_argument(
        "--label-prompt",
        type=int,
        default=115,
        help="Requested Vista3D label prompt.",
    )
    parser.add_argument(
        "--label-name",
        default="heart",
        help="Human-readable label name for manifests.",
    )
    parser.add_argument(
        "--case-name",
        default="ct_philips_heart",
        help="Output case folder name.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--ncnn-manifest",
        default=str(root / "outputs" / "vista3d_ct_philips_heart" / "manifest.json"),
        help="Optional export manifest from Vista3DFixedPromptExport.py.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Vista3DBaseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_bundle_pythonpath(bundle_root: Path) -> None:
    bundle_path = str(bundle_root.resolve())
    if bundle_path not in sys.path:
        sys.path.insert(0, bundle_path)


def expand_label_prompt(requested_label: int, config: dict[str, Any]) -> list[int]:
    subclass = config.get("subclass") or {}
    values = subclass.get(str(int(requested_label)))
    if not values:
        return [int(requested_label)]
    return [int(value) for value in values]


def describe_array(name: str, array: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(array).reshape(-1)
    if flat.size == 0:
        return {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "count": 0,
            "finite": 0,
            "nan": 0,
            "inf": 0,
            "min": None,
            "max": None,
            "mean": None,
        }

    float_flat = flat.astype(np.float32, copy=False)
    finite = np.isfinite(float_flat)
    finite_values = float_flat[finite]
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(flat.size),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(float_flat).sum()),
        "inf": int(np.isinf(float_flat).sum()),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "mean": float(finite_values.mean()) if finite_values.size else None,
    }


def save_array_bin(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(path)


def save_nifti(path: Path, array: np.ndarray, reference_image: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ascontiguousarray(array), reference_image.affine, reference_image.header)
    image.set_data_dtype(array.dtype)
    nib.save(image, str(path))


def apply_official_postprocess(
    parser,
    prepared_image,
    transformed_labels: list[int],
    logits_tensor: torch.Tensor,
    case_output_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray | None, str | None]:
    try:
        parser["output_dir"] = str(case_output_dir / "official_eval")
        postprocessing = parser.get_parsed_content("postprocessing")
        data = {
            "pred": logits_tensor.detach().cpu(),
            "image": prepared_image,
            "label_prompt": torch.tensor(transformed_labels, dtype=torch.long),
        }
        processed = postprocessing(data)
        restored = processed.get("pred")
        if restored is None:
            return None, None

        restored_array = restored.detach().cpu().numpy() if hasattr(restored, "detach") else np.asarray(restored)
        if restored_array.ndim == 4 and restored_array.shape[0] == 1:
            restored_array = restored_array[0]

        output_root = case_output_dir / "official_eval"
        saved_files = sorted(output_root.rglob("*_trans.nii.gz"))
        saved_path = str(saved_files[0]) if saved_files else None
        return np.asarray(restored_array), saved_path
    except Exception as error:
        logger.warning("Official postprocessing/inversion failed: %s", error)
        return None, None


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).resolve()
    output_root = Path(args.output_dir).resolve()
    case_dir = output_root / args.case_name.strip()
    case_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(case_dir / "baseline.log")

    try:
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            raise BaselineError(f"Input CT not found: {input_path}")
        if not str(input_path).lower().endswith((".nii", ".nii.gz")):
            raise BaselineError("Vista3D official baseline currently supports NIfTI CT inputs only.")

        config = load_json(bundle_root / "configs" / "inference.json")
        metadata = load_json(bundle_root / "configs" / "metadata.json")
        ensure_bundle_pythonpath(bundle_root)

        from monai.bundle import ConfigParser
        from monai.networks import copy_model_state

        parser = ConfigParser(config)
        parser["bundle_root"] = str(bundle_root)
        parser["device"] = torch.device(args.device)
        parser["input_dict"] = {"image": str(input_path), "label_prompt": [int(args.label_prompt)]}
        parser["output_dir"] = str(case_dir / "official_eval")

        preprocessing = parser.get_parsed_content("preprocessing")
        inferer = parser.get_parsed_content("inferer")

        prepared = preprocessing({"image": str(input_path), "label_prompt": [int(args.label_prompt)]})
        prepared_image = prepared["image"]
        transformed_labels = [int(value) for value in prepared.get("label_prompt", expand_label_prompt(args.label_prompt, config))]

        network = parser.get_parsed_content("network_def")
        checkpoint_path = bundle_root / "models" / "model.pt"
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict) else payload
        copy_model_state(dst=network, src=state_dict)
        network = network.to(args.device).eval()

        inputs = prepared_image.unsqueeze(0).to(device=args.device, dtype=torch.float32)
        class_vector = torch.tensor([[value] for value in transformed_labels], dtype=torch.long, device=args.device)

        logger.info(
            "Run Vista3D baseline | input=%s | requested_label=%s | transformed_labels=%s | prepared_shape=%s | device=%s",
            input_path,
            args.label_prompt,
            transformed_labels,
            tuple(prepared_image.shape),
            args.device,
        )

        with torch.no_grad():
            logits = inferer(
                inputs=inputs,
                network=network,
                point_coords=None,
                point_labels=None,
                class_vector=class_vector,
            )

        logits_tensor = logits[0].detach().cpu()
        raw_logits_tensor = logits_tensor.clone()
        logits_np = raw_logits_tensor.numpy().astype(np.float32, copy=False)
        processed_shape = tuple(int(value) for value in logits_np.shape[1:])

        if logits_np.shape[0] == 1:
            probs_np = torch.sigmoid(raw_logits_tensor).numpy().astype(np.float32, copy=False)
            masks_np = (probs_np >= 0.5).astype(np.uint8, copy=False)
            labelmap_processed = (masks_np[0] * int(transformed_labels[0])).astype(np.uint8, copy=False)
            task_mode = "binary_label_prompt"
            foreground_label_value = int(transformed_labels[0])
        else:
            probs_np = torch.softmax(raw_logits_tensor, dim=0).numpy().astype(np.float32, copy=False)
            best_index = np.argmax(np.maximum(logits_np, 0.0), axis=0).astype(np.int32, copy=False)
            background = np.all(logits_np <= 0.0, axis=0)
            labelmap_processed = np.zeros(best_index.shape, dtype=np.uint8)
            for index, label_value in enumerate(transformed_labels):
                labelmap_processed[best_index == index] = np.uint8(label_value)
            labelmap_processed[background] = 0
            masks_np = None
            task_mode = "multiclass"
            foreground_label_value = None

        reference_image = nib.load(str(input_path))
        restored_labelmap, restored_path = apply_official_postprocess(
            parser=parser,
            prepared_image=prepared_image,
            transformed_labels=transformed_labels,
            logits_tensor=raw_logits_tensor.clone(),
            case_output_dir=case_dir,
            logger=logger,
        )

        if restored_labelmap is None:
            restored_labelmap = None
            restored_path = None

        input_tensor_np = inputs[0].detach().cpu().numpy().astype(np.float32, copy=False)
        save_array_bin(case_dir / "input_tensor_ncdhw_f32.bin", input_tensor_np)
        save_array_bin(case_dir / "logits_ncdhw_f32.bin", logits_np)
        save_array_bin(case_dir / "probs_ncdhw_f32.bin", probs_np)
        if masks_np is not None:
            save_array_bin(case_dir / "masks_ncdhw_u8.bin", masks_np)
        save_array_bin(case_dir / "labelmap_dhw_u8.bin", labelmap_processed)

        if restored_labelmap is not None:
            restored_output_path = case_dir / "labelmap_restored.nii.gz"
            save_nifti(restored_output_path, restored_labelmap.astype(np.uint16, copy=False), reference_image)
            restored_path = str(restored_output_path)

        export_manifest_path = Path(args.ncnn_manifest).resolve()
        ncnn_assets = None
        if export_manifest_path.exists():
            export_manifest = load_json(export_manifest_path)
            ncnn_assets = {
                "model_param_path": export_manifest.get("param_path"),
                "model_bin_path": export_manifest.get("bin_path"),
                "pnnx_param_path": export_manifest.get("pnnx_param_path"),
                "bundle_manifest_path": str(export_manifest_path),
            }

        manifest = {
            "bundle_root": str(bundle_root),
            "case_name": args.case_name.strip(),
            "infer_mode": "monai-sliding-window",
            "task": metadata.get("task"),
            "description": metadata.get("description"),
            "label_classes": metadata.get("label_classes"),
            "task_mode": task_mode,
            "prompt": {
                "requested_label_prompt": [int(args.label_prompt)],
                "transformed_label_prompt": transformed_labels,
                "label_name": args.label_name,
                "foreground_label_value": foreground_label_value,
            },
            "inputs": [
                {
                    "path": str(input_path),
                    "shape": [int(value) for value in reference_image.shape],
                    "spacing": [float(value) for value in reference_image.header.get_zooms()[:3]],
                    "format": "nifti",
                    "original_shape": [int(value) for value in reference_image.shape],
                    "original_spacing": [float(value) for value in reference_image.header.get_zooms()[:3]],
                    "resampled_to_reference": False,
                }
            ],
            "normalize_nonzero": False,
            "threshold": 0.5,
            "processed_volume_shape_dhw": list(processed_shape),
            "sliding_window_roi_dhw": [int(value) for value in config.get("patch_size", [128, 128, 128])],
            "sliding_window_overlap": float(parser.get_parsed_content("inferer").overlap),
            "model_input_shape_ncdhw": list(inputs.shape),
            "network_patch_shape_ncdhw": [1, 1, *[int(value) for value in config.get("patch_size", [128, 128, 128])]],
            "original_volume_shape_dhw": [int(value) for value in reference_image.shape],
            "unity_buffer_shape_whdc": [
                int(input_tensor_np.shape[3]),
                int(input_tensor_np.shape[2]),
                int(input_tensor_np.shape[1]),
                int(input_tensor_np.shape[0]),
            ],
            "model_output_shape_ncdhw": [1, *list(logits_np.shape)],
            "unity_output_shape_whdc": [
                int(logits_np.shape[3]),
                int(logits_np.shape[2]),
                int(logits_np.shape[1]),
                int(logits_np.shape[0]),
            ],
            "files": {
                "input_tensor_f32_bin": "input_tensor_ncdhw_f32.bin",
                "logits_f32_bin": "logits_ncdhw_f32.bin",
                "probs_f32_bin": "probs_ncdhw_f32.bin",
                "masks_u8_bin": "masks_ncdhw_u8.bin" if masks_np is not None else None,
                "labelmap_u8_bin": "labelmap_dhw_u8.bin",
                "restored_labelmap": Path(restored_path).name if restored_path else None,
                "official_restored_labelmap": restored_path,
            },
            "stats": {
                "input_tensor": describe_array("input_tensor", input_tensor_np),
                "logits": describe_array("logits", logits_np),
                "probs": describe_array("probs", probs_np),
                "labelmap": describe_array("labelmap", labelmap_processed),
                "restored_labelmap": describe_array("restored_labelmap", restored_labelmap) if restored_labelmap is not None else None,
            },
            "ncnn_assets": ncnn_assets,
        }
        if masks_np is not None:
            manifest["stats"]["masks"] = describe_array("masks", masks_np)

        (case_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_lines = [
            f"case={args.case_name.strip()}",
            f"input={input_path}",
            f"device={args.device}",
            f"requested_label_prompt={args.label_prompt}",
            f"transformed_label_prompt={','.join(str(value) for value in transformed_labels)}",
            f"label_name={args.label_name}",
            f"task_mode={task_mode}",
            f"prepared_shape_dhw={processed_shape[0]},{processed_shape[1]},{processed_shape[2]}",
            f"original_shape_dhw={reference_image.shape[0]},{reference_image.shape[1]},{reference_image.shape[2]}",
            f"network_patch_shape_dhw={config.get('patch_size', [128, 128, 128])[0]},{config.get('patch_size', [128, 128, 128])[1]},{config.get('patch_size', [128, 128, 128])[2]}",
            f"sliding_window_overlap={float(parser.get_parsed_content('inferer').overlap):0.3f}",
            f"restored_labelmap={restored_path or ''}",
        ]
        (case_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        logger.info("Vista3D baseline complete | case_dir=%s", case_dir)
        print(str(case_dir))
        return 0
    except Exception as error:
        logger.exception("Vista3D baseline failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
