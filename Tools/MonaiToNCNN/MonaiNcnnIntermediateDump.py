from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Dump selected intermediate tensors from the pnnx-exported BraTS model.")
    parser.add_argument(
        "--input-bin",
        required=True,
        help="Path to input_tensor_ncdhw_f32.bin.",
    )
    parser.add_argument(
        "--input-shape",
        default="1,4,224,224,144",
        help="Input tensor shape in N,C,D,H,W order.",
    )
    parser.add_argument(
        "--model-py",
        default=str(root / "outputs" / "brats_mri_segmentation" / "brats_mri_segmentation.sim_pnnx.py"),
        help="Path to the pnnx-exported Python model file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write intermediate dumps into.",
    )
    return parser.parse_args()


def parse_shape(text: str) -> tuple[int, int, int, int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 5:
        raise ValueError(f"Expected 5 integers for input shape, got: {text}")
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def load_model_module(model_py: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("monai_pnnx_model", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_f32(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=np.float32).tofile(path)


def main() -> int:
    args = parse_args()
    input_shape = parse_shape(args.input_shape)
    input_path = Path(args.input_bin).resolve()
    model_py = Path(args.model_py).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.fromfile(input_path, dtype=np.float32)
    expected = int(np.prod(np.asarray(input_shape, dtype=np.int64)))
    if data.size != expected:
        raise RuntimeError(f"Input element count mismatch: expected {expected}, got {data.size}")
    x = torch.from_numpy(data.reshape(input_shape))

    module = load_model_module(model_py)
    net = module.Model()
    net.float()
    net.eval()

    refs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        v_0 = x
        v_1 = net.conv3d_0(v_0)
        refs["1"] = v_1.detach().cpu().numpy()[0]

        v_2 = net.gn_0(v_1)
        v_3 = net.gn_0(v_1)
        refs["3"] = v_3.detach().cpu().numpy()[0]

        v_4 = torch.nn.functional.relu(v_3)
        refs["4"] = v_4.detach().cpu().numpy()[0]

        v_5 = net.conv3d_1(v_4)
        refs["5"] = v_5.detach().cpu().numpy()[0]

        v_6 = net.gn_1(v_5)
        refs["6"] = v_6.detach().cpu().numpy()[0]

        out = net(v_0)
        refs["out0"] = out.detach().cpu().numpy()[0]

    manifest: dict[str, dict[str, object]] = {}
    for name, array in refs.items():
        c, d, h, w = array.shape
        file_name = f"{name}_ncdhw_like_f32.bin"
        save_f32(output_dir / file_name, array)
        manifest[name] = {
            "shape_cdhw": [int(c), int(d), int(h), int(w)],
            "element_count": int(array.size),
            "file": file_name,
        }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
