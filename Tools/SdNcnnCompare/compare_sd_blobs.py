#!/usr/bin/env python3
"""Compare Stable Diffusion NCNN official dumps with Unity logical dumps."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


HEADER_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")


@dataclass(frozen=True)
class MatDump:
    values: list[float]
    dims: int = 1
    w: int = 0
    h: int = 1
    d: int = 1
    c: int = 1
    elempack: int = 1

    @property
    def shape_text(self) -> str:
        return f"dims={self.dims} w={self.w} h={self.h} d={self.d} c={self.c} elempack={self.elempack}"


def read_dump(path: Path) -> MatDump:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    dims = 1
    w = 0
    h = 1
    d = 1
    c = 1
    elempack = 1
    start = 0

    if lines and lines[0].startswith("#"):
        fields = dict(HEADER_RE.findall(lines[0]))
        dims = int(fields.get("dims", dims))
        w = int(fields.get("w", w))
        h = int(fields.get("h", h))
        d = int(fields.get("d", d))
        c = int(fields.get("c", c))
        elempack = int(fields.get("elempack", elempack))
        start = 1

    values: list[float] = []
    for line in lines[start:]:
        text = line.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            values.append(float("nan"))

    if w <= 0:
        w = len(values)
    return MatDump(values, dims, w, h, d, c, elempack)


def unpack_ncnn(mat: MatDump) -> list[float]:
    pack = max(1, mat.elempack)
    raw = mat.values
    logical_count = mat.w * max(1, mat.h) * max(1, mat.d) * max(1, mat.c) * pack
    if pack == 1 and len(raw) == logical_count:
        return raw

    if mat.dims == 1:
        logical_w = mat.w * pack
        out = [0.0] * logical_w
        for x in range(mat.w):
            base = x * pack
            for p in range(pack):
                out[x * pack + p] = raw[base + p]
        return out

    if mat.dims == 2:
        logical_h = mat.h * pack
        out = [0.0] * (mat.w * logical_h)
        for y_pack in range(mat.h):
            for x in range(mat.w):
                base = (y_pack * mat.w + x) * pack
                for p in range(pack):
                    y = y_pack * pack + p
                    out[y * mat.w + x] = raw[base + p]
        return out

    if mat.dims == 3:
        logical_c = mat.c * pack
        wh = mat.w * mat.h
        cstep = len(raw) // max(1, mat.c * pack)
        if cstep < wh:
            cstep = wh
        out = [0.0] * (wh * logical_c)
        for c_pack in range(mat.c):
            for i in range(wh):
                base = (c_pack * cstep + i) * pack
                for p in range(pack):
                    c = c_pack * pack + p
                    out[c * wh + i] = raw[base + p]
        return out

    if mat.dims == 4:
        logical_c = mat.c * pack
        whd = mat.w * mat.h * mat.d
        cstep = len(raw) // max(1, mat.c * pack)
        if cstep < whd:
            cstep = whd
        out = [0.0] * (whd * logical_c)
        for c_pack in range(mat.c):
            for i in range(whd):
                base = (c_pack * cstep + i) * pack
                for p in range(pack):
                    c = c_pack * pack + p
                    out[c * whd + i] = raw[base + p]
        return out

    return raw


def summarize(a: list[float], b: list[float]) -> dict[str, float | int]:
    n = min(len(a), len(b))
    finite = 0
    nonfinite = 0
    max_abs = -1.0
    max_idx = -1
    sum_abs = 0.0
    sum_sq = 0.0
    for i in range(n):
        av = a[i]
        bv = b[i]
        if not math.isfinite(av) or not math.isfinite(bv):
            nonfinite += 1
            continue
        diff = abs(av - bv)
        finite += 1
        sum_abs += diff
        sum_sq += diff * diff
        if diff > max_abs:
            max_abs = diff
            max_idx = i

    return {
        "n": n,
        "len_a": len(a),
        "len_b": len(b),
        "finite": finite,
        "nonfinite": nonfinite,
        "max_abs": max_abs if max_abs >= 0 else float("nan"),
        "mae": sum_abs / finite if finite else float("nan"),
        "rmse": math.sqrt(sum_sq / finite) if finite else float("nan"),
        "max_idx": max_idx,
    }


def format_float(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.9g}"


def resolve_blob_names(official_dir: Path, unity_dir: Path, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit

    names: set[str] = set()
    for path in official_dir.glob("official_unet_blob_*.txt"):
        name = path.name[len("official_unet_blob_") : -len(".txt")]
        if (unity_dir / f"unity_unet_blob_{name}.txt").exists():
            names.add(name)

    def sort_key(name: str) -> tuple[int, int | str]:
        return (0, int(name)) if name.isdigit() else (1, name)

    return sorted(names, key=sort_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", required=True, type=Path)
    parser.add_argument("--unity-dir", required=True, type=Path)
    parser.add_argument("--blob", action="append", default=[])
    parser.add_argument("--prefix", default="unet_blob")
    parser.add_argument("--top", type=int, default=0)
    args = parser.parse_args()

    names = resolve_blob_names(args.official_dir, args.unity_dir, args.blob)
    if not names:
        print("No matching blob dumps found.")
        return 1

    print("blob\tshape\tcount\tmax_abs\tmae\trmse\tmax_idx\tlen_official\tlen_unity\tnonfinite")
    rows = []
    for name in names:
        official_path = args.official_dir / f"official_{args.prefix}_{name}.txt"
        unity_path = args.unity_dir / f"unity_{args.prefix}_{name}.txt"
        if not official_path.exists() or not unity_path.exists():
            continue
        official = read_dump(official_path)
        unity = read_dump(unity_path)
        official_values = unpack_ncnn(official)
        unity_values = unity.values
        stats = summarize(official_values, unity_values)
        rows.append((name, official.shape_text, stats))

    if args.top > 0:
        rows.sort(key=lambda row: float(row[2]["mae"]), reverse=True)
        rows = rows[: args.top]

    for name, shape, stats in rows:
        print(
            "\t".join(
                [
                    name,
                    shape,
                    format_float(stats["n"]),
                    format_float(stats["max_abs"]),
                    format_float(stats["mae"]),
                    format_float(stats["rmse"]),
                    format_float(stats["max_idx"]),
                    format_float(stats["len_a"]),
                    format_float(stats["len_b"]),
                    format_float(stats["nonfinite"]),
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
