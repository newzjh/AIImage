from __future__ import annotations

import json
import unittest
from pathlib import Path

from qwen35_tokenizer import ByteLevelBpeTokenizer


class TokenizerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = Path(__file__).resolve().parents[1] / "_models" / "qwen3.5_0.8b"
        if not cls.model_dir.is_dir():
            raise unittest.SkipTest("Qwen3.5 model assets are not available")
        cls.config = json.loads((cls.model_dir / "model.json").read_text(encoding="utf-8"))
        cls.tokenizer = ByteLevelBpeTokenizer.from_model_dir(cls.model_dir)

    def test_missing_special_tokens_are_appended_in_model_order(self) -> None:
        base_vocab_size = sum(
            1
            for line in (self.model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
            if line.rstrip("\r\n")
        )
        specials = self.config["tokenizer"]["additional_special_tokens"]
        for offset, token in enumerate(specials):
            self.assertEqual(self.tokenizer.token_to_id[token], base_vocab_size + offset)
        self.assertEqual(len(self.tokenizer.id_to_token), base_vocab_size + len(specials))

    def test_chat_and_vision_markers_are_atomic(self) -> None:
        text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>"
        ids = self.tokenizer.encode(text)
        expected_specials = [
            "<|im_start|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|im_end|>",
        ]
        for token in expected_specials:
            self.assertEqual(ids.count(self.tokenizer.token_to_id[token]), 1)
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=False), text)

    def test_known_ascii_ids(self) -> None:
        self.assertEqual(self.tokenizer.encode("Hello, ncnn!"), [9419, 11, 24330, 7136, 0])


if __name__ == "__main__":
    unittest.main()
