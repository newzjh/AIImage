"""Byte-level BPE tokenizer matching the ncnn_llm Qwen3.5 path."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    encoder: dict[int, str] = {}
    decoder: dict[str, int] = {}
    extra = 0
    for byte in range(256):
        printable = 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255
        codepoint = byte if printable else 256 + extra
        if not printable:
            extra += 1
        encoded = chr(codepoint)
        encoder[byte] = encoded
        decoder[encoded] = byte
    return encoder, decoder


class ByteLevelBpeTokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> None:
        with vocab_path.open("r", encoding="utf-8") as stream:
            self.id_to_token = [line.rstrip("\r\n") for line in stream if line.rstrip("\r\n")]
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self.merge_ranks: dict[tuple[str, str], int] = {}
        with merges_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) >= 2:
                    self.merge_ranks.setdefault((fields[0], fields[1]), len(self.merge_ranks))
        configured_special_tokens: list[str] = []
        for token in special_tokens:
            if not token or token in configured_special_tokens:
                continue
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.id_to_token)
                self.id_to_token.append(token)
            configured_special_tokens.append(token)
        self.special_tokens = sorted(configured_special_tokens, key=len, reverse=True)
        self.special_ids = {self.token_to_id[token] for token in self.special_tokens}
        self.byte_encoder, self.byte_decoder = _byte_maps()

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "ByteLevelBpeTokenizer":
        config = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        tokenizer = config["tokenizer"]
        return cls(
            model_dir / tokenizer["vocab_file"],
            model_dir / tokenizer["merges_file"],
            list(tokenizer.get("additional_special_tokens", [])),
        )

    @lru_cache(maxsize=8192)
    def _bpe(self, piece: str) -> tuple[str, ...]:
        symbols = list(piece)
        while len(symbols) >= 2:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        return tuple(symbols)

    def _encode_normal(self, text: str) -> list[int]:
        encoded = "".join(self.byte_encoder[byte] for byte in text.encode("utf-8"))
        result: list[int] = []
        unknown = self.token_to_id.get("<unk>")
        for token in self._bpe(encoded):
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                result.append(token_id)
                continue
            for char in token:
                char_id = self.token_to_id.get(char, unknown)
                if char_id is not None:
                    result.append(char_id)
        return result

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        normal_start = 0
        index = 0
        while index < len(text):
            matched = next((token for token in self.special_tokens if text.startswith(token, index)), None)
            if matched is None:
                index += 1
                continue
            if normal_start < index:
                result.extend(self._encode_normal(text[normal_start:index]))
            result.append(self.token_to_id[matched])
            index += len(matched)
            normal_start = index
        if normal_start < len(text):
            result.extend(self._encode_normal(text[normal_start:]))
        return result

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        combined = ""
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                continue
            if skip_special_tokens and token_id in self.special_ids:
                continue
            token = self.id_to_token[token_id]
            combined += {"\\t": "\t", "\\n": "\n", "\\r": "\r"}.get(token, token)
        raw = bytearray()
        for char in combined:
            byte = self.byte_decoder.get(char)
            if byte is not None:
                raw.append(byte)
        return raw.decode("utf-8", errors="replace")
