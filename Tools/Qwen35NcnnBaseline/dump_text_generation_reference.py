"""Dump the ncnn/Python final hidden row and logits for Unity alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_ncnn_baseline import DecoderCache, Qwen35TextBaseline, _require_success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    runner = Qwen35TextBaseline(args.model_dir.resolve(), args.threads)
    token_ids = runner.tokenizer.encode(prompt)
    cache = DecoderCache()
    position = 0
    if len(token_ids) > 1:
        _, cache = runner._decode(runner._embed(token_ids[:-1]), len(token_ids) - 1, position, cache)
        position += len(token_ids) - 1
    hidden, cache = runner._decode(runner._embed([token_ids[-1]]), 1, position, cache)
    position += 1

    extractor = runner.projection.create_extractor()
    _require_success(extractor.input("in0", hidden), "feeding LM head")
    logits_mat = runner._extract(extractor, "out0")
    hidden_values = np.ascontiguousarray(np.asarray(hidden, dtype=np.float32).reshape(-1))
    logits = np.ascontiguousarray(np.asarray(logits_mat, dtype=np.float32).reshape(-1))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    hidden_path = args.output_prefix.with_suffix(".decoder_out0.f32")
    logits_path = args.output_prefix.with_suffix(".logits.f32")
    report_path = args.output_prefix.with_suffix(".json")
    hidden_values.tofile(hidden_path)
    logits.tofile(logits_path)
    top_ids = np.argsort(logits[: len(runner.tokenizer.id_to_token)])[-10:][::-1]
    report = {
        "schema": "qwen35.reference.text-generation-checkpoint/v1",
        "prompt": prompt,
        "token_ids": token_ids,
        "position": position,
        "hidden_count": int(hidden_values.size),
        "logits_count": int(logits.size),
        "top_ids": [int(value) for value in top_ids],
        "top_values": [float(logits[value]) for value in top_ids],
        "top_tokens": [runner.tokenizer.decode([int(value)], skip_special_tokens=False) for value in top_ids],
        "hidden_path": str(hidden_path.resolve()),
        "logits_path": str(logits_path.resolve()),
        "valid": hidden_values.size == 1024 and logits.size == 248320,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
