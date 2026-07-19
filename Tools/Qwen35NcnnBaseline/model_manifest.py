"""Pinned Qwen3.5 0.8B model manifest from the SDU ncnn mirror."""

from __future__ import annotations

from dataclasses import dataclass


BASE_URL = "https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3.5_0.8b/"


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    groups: frozenset[str]

    @property
    def url(self) -> str:
        return BASE_URL + self.name


FILES = (
    ModelFile("merges.txt", 3_353_259, frozenset({"metadata", "text"})),
    ModelFile("model.json", 2_315, frozenset({"metadata", "text", "vision"})),
    ModelFile("qwen3.5_decoder.ncnn.bin", 1_992_454_120, frozenset({"text", "weights"})),
    ModelFile("qwen3.5_decoder.ncnn.param", 72_827, frozenset({"metadata", "text"})),
    ModelFile("qwen3.5_embed_token.ncnn.bin", 1_017_118_724, frozenset({"text", "weights"})),
    ModelFile("qwen3.5_embed_token.ncnn.param", 165, frozenset({"metadata", "text"})),
    ModelFile("qwen3.5_proj_out.ncnn.param", 179, frozenset({"metadata", "text"})),
    ModelFile("qwen3.5_vision_embed_patch.ncnn.bin", 4_721_668, frozenset({"vision", "weights"})),
    ModelFile("qwen3.5_vision_embed_patch.ncnn.param", 213, frozenset({"metadata", "vision"})),
    ModelFile("qwen3.5_vision_embed_pos.ncnn.bin", 7_077_888, frozenset({"vision", "weights"})),
    ModelFile("qwen3.5_vision_embed_pos.ncnn.param", 360, frozenset({"metadata", "vision"})),
    ModelFile("qwen3.5_vision_encoder.ncnn.bin", 390_572_432, frozenset({"vision", "weights"})),
    ModelFile("qwen3.5_vision_encoder.ncnn.param", 22_132, frozenset({"metadata", "vision"})),
    ModelFile("vocab.txt", 3_111_730, frozenset({"metadata", "text"})),
)


def files_for_group(group: str) -> tuple[ModelFile, ...]:
    if group == "all":
        return FILES
    if group == "metadata":
        return tuple(item for item in FILES if "metadata" in item.groups)
    if group == "text":
        return tuple(item for item in FILES if "text" in item.groups)
    if group == "vision":
        return tuple(
            item
            for item in FILES
            if "vision" in item.groups
            or item.name in {"model.json", "vocab.txt", "merges.txt"}
        )
    raise ValueError(f"unsupported model group: {group}")

