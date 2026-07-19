"""Attribute the first vision-encoder residual mismatch around gemm_3."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


TAG_FP16 = 0x01306B47
TAG_INT8 = 0x000D4B38
TAG_FLOAT32_EXTRA_SCALE = 0x0002C056


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NcnnBinReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = path.open("rb")

    def close(self) -> None:
        self.stream.close()

    @property
    def position(self) -> int:
        return self.stream.tell()

    def read_exact(self, count: int) -> bytes:
        value = self.stream.read(count)
        if len(value) != count:
            raise EOFError(f"short read at {self.position}: {len(value)} != {count}")
        return value

    def read_raw_float32(self, count: int) -> np.ndarray:
        return np.frombuffer(self.read_exact(count * 4), dtype="<f4").copy()

    def read_ncnn_array(self, count: int) -> np.ndarray:
        flag = struct.unpack("<I", self.read_exact(4))[0]
        if flag == TAG_FP16:
            byte_count = count * 2
            values = np.frombuffer(self.read_exact((byte_count + 3) & ~3)[:byte_count], dtype="<f2")
            return values.astype(np.float32)
        if flag == TAG_INT8:
            byte_count = count
            values = np.frombuffer(self.read_exact((byte_count + 3) & ~3)[:byte_count], dtype=np.int8)
            return values.astype(np.float32)
        if flag == TAG_FLOAT32_EXTRA_SCALE:
            return self.read_raw_float32(count)

        flag_bytes = flag.to_bytes(4, "little")
        if sum(flag_bytes) != 0:
            table = self.read_raw_float32(256)
            byte_count = count
            indices = np.frombuffer(self.read_exact((byte_count + 3) & ~3)[:byte_count], dtype=np.uint8)
            return table[indices]
        if flag_bytes[0] == 0:
            return self.read_raw_float32(count)
        raise ValueError(f"unsupported ncnn encoding flag 0x{flag:08x} at {self.position - 4}")


def read_gemm(reader: NcnnBinReader, n: int, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    start = reader.position
    weight = reader.read_ncnn_array(n * k).reshape(n, k)
    bias = reader.read_ncnn_array(n)
    return weight, bias, {"start": start, "end": reader.position}


def load_gemm3(bin_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    reader = NcnnBinReader(bin_path)
    offsets: dict[str, object] = {}
    try:
        # ln_63 stores affine vectors as raw float32 (ncnn load_type=1).
        offsets["ln_63_start"] = reader.position
        reader.read_raw_float32(768)
        reader.read_raw_float32(768)
        offsets["ln_63_end"] = reader.position

        for name, n, k in (("gemm_0", 2304, 768), ("gemm_1", 768, 768)):
            _, _, offsets[name] = read_gemm(reader, n, k)

        offsets["ln_64_start"] = reader.position
        reader.read_raw_float32(768)
        reader.read_raw_float32(768)
        offsets["ln_64_end"] = reader.position

        _, _, offsets["gemm_2"] = read_gemm(reader, 3072, 768)
        weight, bias, offsets["gemm_3"] = read_gemm(reader, 768, 3072)
        return weight, bias, offsets
    finally:
        reader.close()


def sequential_float32(a: np.ndarray, b: np.ndarray, bias: np.float32) -> np.float32:
    acc = np.float32(0.0)
    for index in range(a.size):
        acc = np.float32(acc + np.float32(a[index] * b[index]))
    return np.float32(acc + bias)


def split_float32(a: np.ndarray, b: np.ndarray, bias: np.float32, lanes: int) -> np.float32:
    accumulators = np.zeros(lanes, dtype=np.float32)
    for index in range(a.size):
        lane = index % lanes
        accumulators[lane] = np.float32(accumulators[lane] + np.float32(a[index] * b[index]))
    acc = np.float32(0.0)
    for value in accumulators:
        acc = np.float32(acc + value)
    return np.float32(acc + bias)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    weight, bias, offsets = load_gemm3(args.model_bin)
    unity76 = np.fromfile(args.unity_blob76, dtype="<f4").reshape(3072, 3072)
    reference76 = np.fromfile(args.reference_blob76, dtype="<f4").reshape(3072, 3072)
    unity71 = np.fromfile(args.unity_blob71, dtype="<f4").reshape(3072, 768)
    reference71 = np.fromfile(args.reference_blob71, dtype="<f4").reshape(3072, 768)
    unity78 = np.fromfile(args.unity_blob78, dtype="<f4").reshape(3072, 768)
    reference78 = np.fromfile(args.reference_blob78, dtype="<f4").reshape(3072, 768)
    unity77 = np.float32(unity78 - unity71)
    reference77 = np.float32(reference78 - reference71)

    absolute78 = np.abs(unity78 - reference78)
    allowed78 = np.float32(args.atol) + np.float32(args.rtol) * np.abs(reference78)
    failures = np.flatnonzero(absolute78.reshape(-1) > allowed78.reshape(-1))
    worst = int(np.argmax(absolute78))
    indices = list(dict.fromkeys([*(int(value) for value in failures), worst]))

    details: list[dict[str, object]] = []
    for flat_index in indices:
        row, col = divmod(flat_index, 768)
        unity_input = unity76[row]
        reference_input = reference76[row]
        col_weight = weight[col]
        input_delta_projection = float(
            np.dot(
                unity_input.astype(np.float64) - reference_input.astype(np.float64),
                col_weight.astype(np.float64),
            )
        )
        details.append(
            {
                "flat_index": flat_index,
                "row": row,
                "column": col,
                "unity78": float(unity78[row, col]),
                "reference78": float(reference78[row, col]),
                "residual_error": float(unity78[row, col] - reference78[row, col]),
                "unity77_reconstructed": float(unity77[row, col]),
                "reference77_reconstructed": float(reference77[row, col]),
                "gemm_error": float(unity77[row, col] - reference77[row, col]),
                "input_delta_projection_float64": input_delta_projection,
                "unity_input_sequential_float32": float(
                    sequential_float32(unity_input, col_weight, bias[col])
                ),
                "reference_input_sequential_float32": float(
                    sequential_float32(reference_input, col_weight, bias[col])
                ),
                "unity_input_split4_float32": float(split_float32(unity_input, col_weight, bias[col], 4)),
                "unity_input_split8_float32": float(split_float32(unity_input, col_weight, bias[col], 8)),
                "unity_input_dot_float64": float(
                    np.dot(unity_input.astype(np.float64), col_weight.astype(np.float64)) + float(bias[col])
                ),
            }
        )

    return {
        "schema": "qwen35.analysis.vision-gemm3-error/v1",
        "model_bin": str(args.model_bin.resolve()),
        "model_bin_sha256": sha256_file(args.model_bin),
        "gemm3_shape": {"m": 3072, "n": 768, "k": 3072, "transB": True},
        "weight_offsets": offsets,
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "residual_failure_count": int(failures.size),
        "details": details,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (
                ("unity_blob76", args.unity_blob76),
                ("reference_blob76", args.reference_blob76),
                ("unity_blob71", args.unity_blob71),
                ("reference_blob71", args.reference_blob71),
                ("unity_blob78", args.unity_blob78),
                ("reference_blob78", args.reference_blob78),
            )
        },
    }


def main() -> int:
    tool_dir = Path(__file__).resolve().parent
    reports = tool_dir / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bin", type=Path, default=tool_dir / "_models/qwen3.5_0.8b/qwen3.5_vision_encoder.ncnn.bin")
    parser.add_argument("--unity-blob76", type=Path, default=reports / "unity_vision_encoder_blob76_probe.f32")
    parser.add_argument("--reference-blob76", type=Path, default=reports / "ncnn_vision_encoder_blob76_reference.f32")
    parser.add_argument("--unity-blob71", type=Path, default=reports / "unity_vision_encoder_blob71_probe.f32")
    parser.add_argument("--reference-blob71", type=Path, default=reports / "ncnn_vision_encoder_blob71_reference.f32")
    parser.add_argument("--unity-blob78", type=Path, default=reports / "unity_vision_encoder_blob78_probe.f32")
    parser.add_argument("--reference-blob78", type=Path, default=reports / "ncnn_vision_encoder_blob78_reference.f32")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--output", type=Path, default=reports / "vision_encoder_gemm3_error_analysis.json")
    args = parser.parse_args()

    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"failure_count": report["residual_failure_count"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
