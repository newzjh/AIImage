"""CPU text-generation baseline using ncnn Python and Python custom layers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ncnn_custom_layers import register_qwen35_custom_layers
from qwen35_tokenizer import ByteLevelBpeTokenizer


@dataclass
class DecoderCache:
    kv: list[tuple[Any, Any]] = field(default_factory=list)
    conv: list[Any] = field(default_factory=list)
    gdr: list[Any] = field(default_factory=list)


def _require_success(result: int, action: str) -> None:
    if result != 0:
        raise RuntimeError(f"ncnn failed while {action}: error {result}")


class Qwen35TextBaseline:
    def __init__(self, model_dir: Path, threads: int) -> None:
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError("missing ncnn Python binding; run: python -m pip install -r requirements.txt") from exc
        self.ncnn = ncnn
        self.model_dir = model_dir
        self.config = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        self.tokenizer = ByteLevelBpeTokenizer.from_model_dir(model_dir)
        setting = self.config["setting"]
        self.attn_count = int(setting["attn_cnt"])
        self.conv_count = int(setting["sconv_cnt"])
        self.gdr_count = int(setting["gdr_cnt"])
        self.rope_dim = int(setting["rope"]["rope_head_dim"])
        self.rope_theta = np.float32(setting["rope"]["rope_theta"])

        params = self.config["params"]
        self.embed = self._load_net(params["embed_token_param"], params["embed_token_bin"], threads)
        self.decoder = self._load_net(params["decoder_param"], params["decoder_bin"], threads, custom=True)
        self.projection = self._load_net(params["proj_out_param"], params["proj_out_bin"], threads)

    def _load_net(self, param_name: str, bin_name: str, threads: int, *, custom: bool = False) -> Any:
        param_path = self.model_dir / param_name
        bin_path = self.model_dir / bin_name
        for path in (param_path, bin_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing model file: {path}")
        net = self.ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = threads
        if custom:
            register_qwen35_custom_layers(net)
        _require_success(net.load_param(str(param_path)), f"loading {param_name}")
        _require_success(net.load_model(str(bin_path)), f"loading {bin_name}")
        return net

    def _extract(self, extractor: Any, name: str) -> Any:
        result, value = extractor.extract(name)
        _require_success(result, f"extracting {name}")
        return value

    def _embed(self, token_ids: list[int]) -> Any:
        ids = np.ascontiguousarray(np.asarray(token_ids, dtype=np.int32).reshape(1, -1))
        extractor = self.embed.create_extractor()
        _require_success(extractor.input("in0", self.ncnn.Mat(ids)), "feeding token ids")
        return self._extract(extractor, "out0")

    def _rope(self, seq_len: int, position: int) -> tuple[Any, Any]:
        frequency_index = np.arange(self.rope_dim // 2, dtype=np.float32)
        inv_frequency = np.power(
            self.rope_theta,
            -frequency_index * np.float32(2.0 / self.rope_dim),
        )
        positions = np.arange(position, position + seq_len, dtype=np.float32)[:, None]
        angles = positions * inv_frequency[None, :]
        # The Python binding wraps NumPy storage without taking ownership. Clone
        # both caches so single-row decode does not alias recycled NumPy memory.
        cos_cache = self.ncnn.Mat(np.ascontiguousarray(np.cos(angles))).clone()
        sin_cache = self.ncnn.Mat(np.ascontiguousarray(np.sin(angles))).clone()
        return cos_cache, sin_cache

    def _decode(self, embeddings: Any, seq_len: int, position: int, cache: DecoderCache) -> tuple[Any, DecoderCache]:
        past = cache.kv[0][0].h if cache.kv else 0
        mask = np.zeros((seq_len, past + seq_len), dtype=np.float32)
        for row in range(seq_len):
            mask[row, past + row + 1 :] = np.float32(-1e38)
        cos_cache, sin_cache = self._rope(seq_len, position)
        extractor = self.decoder.create_extractor()
        for name, value in (
            ("in0", embeddings),
            ("in1", self.ncnn.Mat(mask)),
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

        next_cache = DecoderCache()
        for index in range(self.attn_count):
            next_cache.kv.append(
                (self._extract(extractor, f"out_cache_k{index}"), self._extract(extractor, f"out_cache_v{index}"))
            )
        next_cache.conv = [self._extract(extractor, f"out_cache_conv{index}") for index in range(self.conv_count)]
        next_cache.gdr = [self._extract(extractor, f"out_cache_gdr{index}") for index in range(self.gdr_count)]
        return self._extract(extractor, "out0"), next_cache

    def _greedy(self, hidden: Any) -> int:
        extractor = self.projection.create_extractor()
        _require_success(extractor.input("in0", hidden), "feeding LM head")
        logits = np.asarray(self._extract(extractor, "out0"), dtype=np.float32).reshape(-1)
        return int(np.argmax(logits))

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt produced no tokens")
        cache = DecoderCache()
        position = 0
        if len(token_ids) > 1:
            hidden, cache = self._decode(self._embed(token_ids[:-1]), len(token_ids) - 1, position, cache)
            position += len(token_ids) - 1
            del hidden
        hidden, cache = self._decode(self._embed([token_ids[-1]]), 1, position, cache)
        position += 1
        next_id = self._greedy(hidden)
        generated: list[int] = []
        eos_id = self.tokenizer.token_to_id.get("<|im_end|>", -1)
        for _ in range(max_new_tokens):
            generated.append(next_id)
            piece = self.tokenizer.decode([next_id], skip_special_tokens=False)
            print(piece, end="", flush=True)
            if next_id == eos_id:
                break
            hidden, cache = self._decode(self._embed([next_id]), 1, position, cache)
            position += 1
            next_id = self._greedy(hidden)
        print()
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "_models" / "qwen3.5_0.8b")
    parser.add_argument("--prompt", default="请用一句话介绍 ncnn。")
    parser.add_argument("--raw-prompt", action="store_true", help="do not apply the Qwen chat wrapper")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.threads <= 0:
        raise ValueError("max-new-tokens and threads must be positive")
    prompt = args.prompt
    if not args.raw_prompt:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    print("Loading the 3.18 GiB CPU baseline. Python custom operators prioritize correctness, not speed.")
    runner = Qwen35TextBaseline(args.model_dir.resolve(), args.threads)
    runner.generate(prompt, args.max_new_tokens)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"baseline failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
