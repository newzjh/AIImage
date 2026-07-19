"""Dump selected decoder blobs for the first text-prefix prefill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_ncnn_baseline import DecoderCache, Qwen35TextBaseline, _require_success


DEFAULT_BLOBS = "90,95,96,97,98,99,106,114,117,120,124,125,128,133,136,143,151,160,174,198,202,212,236,240,241"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blobs", default=DEFAULT_BLOBS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--decode-index", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    runner = Qwen35TextBaseline(args.model_dir.resolve(), args.threads)
    runner.decoder.opt.lightmode = False
    token_ids = runner.tokenizer.encode(prompt)
    prefix_ids = token_ids[:-1]
    if args.decode_index == 0:
        input_ids = prefix_ids
        position = 0
        cache = DecoderCache()
    else:
        _, cache = runner._decode(runner._embed(prefix_ids), len(prefix_ids), 0, DecoderCache())
        input_ids = [token_ids[-1]]
        position = len(prefix_ids)

    embeddings = runner._embed(input_ids)
    past = cache.kv[0][0].h if cache.kv else 0
    cos_cache, sin_cache = runner._rope(len(input_ids), position)
    mask = np.zeros((len(input_ids), past + len(input_ids)), dtype=np.float32)
    for row in range(len(input_ids)):
        mask[row, past + row + 1 :] = np.float32(-1e38)

    extractor = runner.decoder.create_extractor()
    for name, value in (
        ("in0", embeddings),
        ("in1", runner.ncnn.Mat(mask)),
        ("in2", cos_cache),
        ("in3", sin_cache),
    ):
        _require_success(extractor.input(name, value), f"feeding decoder {name}")
    for index, (key_cache, value_cache) in enumerate(cache.kv):
        _require_success(extractor.input(f"cache_k{index}", key_cache), f"feeding cache_k{index}")
        _require_success(extractor.input(f"cache_v{index}", value_cache), f"feeding cache_v{index}")
    for index, value in enumerate(cache.conv):
        _require_success(extractor.input(f"cache_conv{index}", value), f"feeding cache_conv{index}")
    for index, value in enumerate(cache.gdr):
        _require_success(extractor.input(f"cache_gdr{index}", value), f"feeding cache_gdr{index}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "qwen35.reference.decoder-checkpoints/v2",
        "decode_index": args.decode_index,
        "prompt_token_ids": token_ids,
        "prefix_token_ids": prefix_ids,
        "input_token_ids": input_ids,
        "ncnn_lightmode": False,
        "checkpoints": {},
        "valid": True,
    }
    for blob_name in [value.strip() for value in args.blobs.replace(";", ",").split(",") if value.strip()]:
        result, value = extractor.extract(blob_name)
        _require_success(result, f"extracting decoder blob {blob_name}")
        values = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
        output_path = args.output_dir / f"reference.decode{args.decode_index}.blob_{blob_name}.f32"
        values.tofile(output_path)
        report["checkpoints"][blob_name] = {
            "count": int(values.size),
            "shape": list(np.asarray(value).shape),
            "path": str(output_path.resolve()),
            "finite": bool(np.all(np.isfinite(values))),
        }
        report["valid"] = report["valid"] and bool(np.all(np.isfinite(values)))

    report_path = args.output_dir / (
        "reference.decoder_prefix_checkpoints.json"
        if args.decode_index == 0
        else f"reference.decoder_decode{args.decode_index}_checkpoints.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"valid": report["valid"], "checkpoint_count": len(report["checkpoints"])}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
