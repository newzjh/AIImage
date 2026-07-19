#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline


DEFAULT_BLOBS = [
    "33",
    "34",
    "155",
    "163",
    "183",
    "184",
    "156",
    "162",
    "188",
    "189",
    "191",
    "192",
    "195",
    "196",
    "197",
    "198",
    "199",
    "200",
    "201",
    "205",
    "206",
    "207",
    "208",
    "209",
    "210",
    "211",
    "212",
    "213",
    "214",
    "216",
    "218",
    "219",
    "222",
    "223",
    "224",
    "225",
    "227",
    "229",
    "231",
    "232",
    "233",
    "234",
    "235",
    "236",
    "237",
    "238",
    "239",
    "240",
    "241",
    "242",
    "243",
    "244",
    "245",
    "246",
    "247",
    "248",
    "249",
    "250",
    "251",
]

EXTRA_DEBUG_BLOBS = [
    "252",
    "253",
    "254",
    "255",
    "256",
    "257",
    "258",
    "260",
    "261",
    "262",
    "263",
    "264",
    "265",
    "266",
    "267",
    "268",
    "270",
    "271",
    "272",
    "273",
    "274",
    "275",
    "276",
    "277",
    "278",
    "279",
    "280",
    "281",
    "282",
    "283",
    "284",
    "285",
    "286",
    "288",
    "290",
    "291",
    "292",
    "293",
    "294",
    "295",
    "296",
    "297",
    "298",
    "299",
    "300",
    "301",
    "302",
    "303",
    "304",
    "306",
    "308",
    "309",
    "310",
    "311",
    "312",
    "313",
    "314",
    "315",
    "316",
    "317",
    "318",
    "319",
    "320",
    "321",
    "322",
    "323",
]

SUPPORTED_BLOBS = DEFAULT_BLOBS + EXTRA_DEBUG_BLOBS + ["out0"]

HEADER_SHAPES: dict[str, tuple[int, int, int, int, int, int]] = {
    "33": (2, 768, 77, 1, 1, 1),
    "34": (2, 768, 77, 1, 1, 1),
    "155": (1, 1280, 1, 1, 1, 1),
    "163": (1, 1280, 1, 1, 1, 1),
    "183": (1, 1280, 1, 1, 1, 1),
    "184": (1, 1280, 1, 1, 1, 1),
    "156": (3, 64, 64, 1, 320, 1),
    "162": (3, 64, 64, 1, 320, 1),
    "188": (3, 64, 64, 1, 320, 1),
    "189": (3, 64, 64, 1, 320, 1),
    "191": (3, 64, 64, 1, 320, 1),
    "192": (3, 64, 64, 1, 320, 1),
    "195": (3, 64, 64, 1, 320, 1),
    "196": (3, 64, 64, 1, 320, 1),
    "197": (3, 320, 64, 1, 64, 1),
    "198": (2, 320, 4096, 1, 1, 1),
    "199": (2, 320, 4096, 1, 1, 1),
    "200": (2, 320, 4096, 1, 1, 1),
    "201": (2, 320, 4096, 1, 1, 1),
    "205": (2, 320, 4096, 1, 1, 1),
    "206": (2, 320, 4096, 1, 1, 1),
    "207": (2, 320, 4096, 1, 1, 1),
    "208": (3, 40, 8, 1, 4096, 1),
    "209": (3, 40, 4096, 1, 8, 1),
    "210": (3, 40, 8, 1, 4096, 1),
    "211": (3, 40, 4096, 1, 8, 1),
    "212": (3, 40, 8, 1, 4096, 1),
    "213": (3, 40, 4096, 1, 8, 1),
    "214": (3, 40, 4096, 1, 8, 1),
    "216": (2, 320, 4096, 1, 1, 1),
    "218": (2, 320, 4096, 1, 1, 1),
    "220": (2, 320, 4096, 1, 1, 1),
    "219": (2, 320, 4096, 1, 1, 1),
    "222": (2, 320, 4096, 1, 1, 1),
    "223": (2, 320, 4096, 1, 1, 1),
    "224": (2, 320, 77, 1, 1, 1),
    "225": (2, 320, 77, 1, 1, 1),
    "227": (3, 40, 4096, 1, 8, 1),
    "229": (3, 40, 77, 1, 8, 1),
    "231": (3, 40, 77, 1, 8, 1),
    "232": (3, 40, 4096, 1, 8, 1),
    "233": (3, 4096, 8, 1, 40, 1),
    "234": (2, 320, 4096, 1, 1, 1),
    "235": (2, 320, 4096, 1, 1, 1),
    "236": (2, 320, 4096, 1, 1, 1),
    "237": (2, 320, 4096, 1, 1, 1),
    "238": (2, 320, 4096, 1, 1, 1),
    "239": (2, 320, 4096, 1, 1, 1),
    "240": (2, 320, 4096, 1, 1, 1),
    "241": (2, 2560, 4096, 1, 1, 1),
    "242": (2, 1280, 4096, 1, 1, 1),
    "243": (2, 1280, 4096, 1, 1, 1),
    "244": (2, 1280, 4096, 1, 1, 1),
    "245": (2, 1280, 4096, 1, 1, 1),
    "246": (2, 320, 4096, 1, 1, 1),
    "247": (2, 320, 4096, 1, 1, 1),
    "248": (3, 320, 64, 1, 64, 1),
    "249": (3, 64, 64, 1, 320, 1),
    "250": (3, 64, 64, 1, 320, 1),
    "251": (3, 64, 64, 1, 320, 1),
    "252": (3, 64, 64, 1, 320, 1),
    "253": (3, 64, 64, 1, 320, 1),
    "254": (3, 64, 64, 1, 320, 1),
    "255": (3, 64, 64, 1, 320, 1),
    "256": (3, 64, 64, 1, 320, 1),
    "257": (3, 64, 64, 1, 320, 1),
    "258": (1, 320, 1, 1, 1, 1),
    "260": (3, 64, 64, 1, 320, 1),
    "261": (3, 64, 64, 1, 320, 1),
    "262": (3, 64, 64, 1, 320, 1),
    "263": (3, 64, 64, 1, 320, 1),
    "264": (3, 64, 64, 1, 320, 1),
    "265": (3, 64, 64, 1, 320, 1),
    "266": (3, 64, 64, 1, 320, 1),
    "267": (3, 64, 64, 1, 320, 1),
    "268": (3, 64, 64, 1, 320, 1),
    "270": (2, 320, 4096, 1, 1, 1),
    "271": (2, 320, 4096, 1, 1, 1),
    "272": (2, 320, 4096, 1, 1, 1),
    "273": (2, 320, 4096, 1, 1, 1),
    "274": (2, 320, 4096, 1, 1, 1),
    "275": (2, 320, 4096, 1, 1, 1),
    "276": (2, 320, 4096, 1, 1, 1),
    "277": (2, 320, 4096, 1, 1, 1),
    "278": (2, 320, 4096, 1, 1, 1),
    "279": (2, 320, 4096, 1, 1, 1),
    "280": (3, 40, 8, 1, 4096, 1),
    "281": (3, 40, 4096, 1, 8, 1),
    "282": (3, 40, 8, 1, 4096, 1),
    "283": (3, 40, 4096, 1, 8, 1),
    "284": (3, 40, 8, 1, 4096, 1),
    "285": (3, 40, 4096, 1, 8, 1),
    "286": (3, 40, 4096, 1, 8, 1),
    "288": (2, 320, 4096, 1, 1, 1),
    "290": (2, 320, 4096, 1, 1, 1),
    "291": (2, 320, 4096, 1, 1, 1),
    "292": (2, 320, 4096, 1, 1, 1),
    "293": (2, 320, 4096, 1, 1, 1),
    "294": (2, 320, 4096, 1, 1, 1),
    "295": (2, 320, 4096, 1, 1, 1),
    "296": (2, 320, 77, 1, 1, 1),
    "297": (2, 320, 77, 1, 1, 1),
    "298": (3, 40, 8, 1, 4096, 1),
    "299": (3, 40, 4096, 1, 8, 1),
    "300": (3, 40, 8, 1, 77, 1),
    "301": (3, 40, 77, 1, 8, 1),
    "302": (3, 40, 8, 1, 77, 1),
    "303": (3, 40, 77, 1, 8, 1),
    "304": (3, 40, 4096, 1, 8, 1),
    "306": (2, 320, 4096, 1, 1, 1),
    "308": (2, 320, 4096, 1, 1, 1),
    "309": (2, 320, 4096, 1, 1, 1),
    "310": (2, 320, 4096, 1, 1, 1),
    "311": (2, 320, 4096, 1, 1, 1),
    "312": (2, 320, 4096, 1, 1, 1),
    "313": (2, 2560, 4096, 1, 1, 1),
    "314": (2, 1280, 4096, 1, 1, 1),
    "315": (2, 1280, 4096, 1, 1, 1),
    "316": (2, 1280, 4096, 1, 1, 1),
    "317": (2, 1280, 4096, 1, 1, 1),
    "318": (2, 320, 4096, 1, 1, 1),
    "319": (2, 320, 4096, 1, 1, 1),
    "320": (3, 320, 64, 1, 64, 1),
    "321": (3, 64, 64, 1, 320, 1),
    "322": (3, 64, 64, 1, 320, 1),
    "323": (3, 64, 64, 1, 320, 1),
    "out0": (3, 64, 64, 1, 4, 1),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run SD15 inpaint Python baseline on a dumped sample and export mapped raw-blob reference dumps."
    )
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"E:\Projects\AIImage\Tools\sd15inpainting2ncnnExporter\output\diffusers"),
    )
    ap.add_argument("--prompt-kind", choices=("cond", "uncond"), default="cond")
    ap.add_argument("--blob", action="append", default=[])
    ap.add_argument("--out-dir", type=Path)
    return ap.parse_args()


def load_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path} expected {expected} floats but found {arr.size}")
    return torch.from_numpy(arr.reshape(shape))


def flatten_tensor(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def write_float_lines(path: Path, values: np.ndarray, header_shape: tuple[int, int, int, int, int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header_shape is not None:
            dims, w, h, d, c, elempack = header_shape
            f.write(f"# dims={dims} w={w} h={h} d={d} c={c} elempack={elempack}\n")
        for value in values:
            f.write(f"{float(value):.9g}\n")


def write_stats(path: Path, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    lines = [
        f"count={values.size}",
        f"nonfinite={int(values.size - finite.size)}",
    ]
    if finite.size > 0:
        lines.extend(
            [
                f"min={float(finite.min()):.9g}",
                f"max={float(finite.max()):.9g}",
                f"mean={float(finite.mean()):.9g}",
                f"mean_abs={float(np.abs(finite).mean()):.9g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clone_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().clone()


def add_blob(captured: dict[str, torch.Tensor], blob: str, value: torch.Tensor) -> None:
    if blob not in captured:
        captured[blob] = clone_tensor(value)


def add_prompt_alias_blobs(captured: dict[str, torch.Tensor], prompt: torch.Tensor | None) -> None:
    if prompt is None or not isinstance(prompt, torch.Tensor):
        return
    add_blob(captured, "33", prompt)
    add_blob(captured, "34", prompt)


def attach_capture_hooks(
    pipe: StableDiffusionInpaintPipeline,
    captured: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    unet = pipe.unet

    def pre_hook(blob: str) -> Callable:
        def _hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if args and isinstance(args[0], torch.Tensor):
                add_blob(captured, blob, args[0])

        return _hook

    def out_hook(blob: str) -> Callable:
        def _hook(_mod: torch.nn.Module, _args: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                out = out[0]
            if isinstance(out, torch.Tensor):
                add_blob(captured, blob, out)

        return _hook

    attn = unet.down_blocks[0].attentions[0]
    attn_next = unet.down_blocks[0].attentions[1]
    res0 = unet.down_blocks[0].resnets[0]
    res1 = unet.down_blocks[0].resnets[1]
    tb = attn.transformer_blocks[0]
    tb_next = attn_next.transformer_blocks[0]

    handles.append(unet.conv_in.register_forward_hook(out_hook("156")))
    handles.append(res0.conv1.register_forward_hook(out_hook("162")))
    handles.append(res0.norm2.register_forward_pre_hook(pre_hook("188")))
    handles.append(res0.norm2.register_forward_hook(out_hook("189")))
    handles.append(res0.conv2.register_forward_hook(out_hook("191")))
    handles.append(attn.norm.register_forward_pre_hook(pre_hook("192")))

    def norm1_pre_hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            add_blob(captured, "198", args[0])
            add_blob(captured, "199", args[0])
            add_blob(captured, "200", args[0])

    def norm2_pre_hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            add_blob(captured, "219", args[0])
            add_blob(captured, "220", args[0])

    handles.append(attn.transformer_blocks[0].norm1.register_forward_pre_hook(norm1_pre_hook))
    handles.append(tb.norm1.register_forward_hook(out_hook("201")))
    handles.append(tb.norm2.register_forward_hook(out_hook("222")))
    handles.append(attn.norm.register_forward_hook(out_hook("195")))

    def proj_in_out_hook(_mod: torch.nn.Module, _args: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            out = out[0]
        if isinstance(out, torch.Tensor):
            add_blob(captured, "196", out)
            # pnnx emits this as permute_458 (NCHW -> NHWC) before reshape_491.
            add_blob(captured, "197", out.permute(0, 2, 3, 1).contiguous())

    handles.append(attn.proj_in.register_forward_hook(proj_in_out_hook))
    handles.append(tb.norm2.register_forward_pre_hook(norm2_pre_hook))
    handles.append(tb.norm3.register_forward_hook(out_hook("240")))
    handles.append(tb.ff.net[0].proj.register_forward_hook(out_hook("241")))
    handles.append(tb.ff.net[2].register_forward_hook(out_hook("246")))
    handles.append(attn.proj_out.register_forward_hook(out_hook("250")))
    handles.append(attn.register_forward_hook(out_hook("251")))

    handles.append(res1.norm1.register_forward_hook(out_hook("255")))
    handles.append(res1.conv1.register_forward_hook(out_hook("257")))
    handles.append(res1.time_emb_proj.register_forward_hook(out_hook("258")))

    def res1_norm2_pre_hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            add_blob(captured, "260", args[0])

    handles.append(res1.norm2.register_forward_pre_hook(res1_norm2_pre_hook))
    handles.append(res1.norm2.register_forward_hook(out_hook("261")))
    handles.append(res1.conv2.register_forward_hook(out_hook("263")))
    handles.append(res1.register_forward_hook(out_hook("264")))
    handles.append(attn_next.norm.register_forward_hook(out_hook("267")))
    handles.append(attn_next.proj_in.register_forward_hook(out_hook("268")))

    def next_norm1_pre_hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            add_blob(captured, "270", args[0])
            add_blob(captured, "271", args[0])
            add_blob(captured, "272", args[0])

    def next_norm2_pre_hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            add_blob(captured, "291", args[0])
            add_blob(captured, "292", args[0])
            add_blob(captured, "293", args[0])

    def next_norm3_pre_hook(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if args and isinstance(args[0], torch.Tensor):
            add_blob(captured, "309", args[0])
            add_blob(captured, "310", args[0])
            add_blob(captured, "311", args[0])

    handles.append(tb_next.norm1.register_forward_pre_hook(next_norm1_pre_hook))
    handles.append(tb_next.norm1.register_forward_hook(out_hook("273")))
    handles.append(tb_next.attn1.to_q.register_forward_hook(out_hook("277")))
    handles.append(tb_next.attn1.to_k.register_forward_hook(out_hook("278")))
    handles.append(tb_next.attn1.to_v.register_forward_hook(out_hook("279")))
    handles.append(tb_next.attn1.to_out[0].register_forward_hook(out_hook("290")))
    handles.append(tb_next.norm2.register_forward_pre_hook(next_norm2_pre_hook))
    handles.append(tb_next.norm2.register_forward_hook(out_hook("294")))
    handles.append(tb_next.attn2.to_q.register_forward_hook(out_hook("295")))
    handles.append(tb_next.attn2.to_k.register_forward_hook(out_hook("296")))
    handles.append(tb_next.attn2.to_v.register_forward_hook(out_hook("297")))
    handles.append(tb_next.attn2.to_out[0].register_forward_hook(out_hook("308")))
    handles.append(tb_next.norm3.register_forward_pre_hook(next_norm3_pre_hook))
    handles.append(tb_next.norm3.register_forward_hook(out_hook("312")))
    handles.append(tb_next.ff.net[0].proj.register_forward_hook(out_hook("313")))
    handles.append(tb_next.ff.net[2].register_forward_hook(out_hook("318")))
    handles.append(attn_next.proj_out.register_forward_hook(out_hook("322")))
    handles.append(attn_next.register_forward_hook(out_hook("323")))

    class _Marker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    marker_199 = _Marker()
    marker_205 = _Marker()
    marker_206 = _Marker()
    marker_207 = _Marker()
    marker_208 = _Marker()
    marker_209 = _Marker()
    marker_210 = _Marker()
    marker_211 = _Marker()
    marker_212 = _Marker()
    marker_213 = _Marker()
    marker_214 = _Marker()
    marker_216 = _Marker()
    marker_218 = _Marker()
    marker_223 = _Marker()
    marker_224 = _Marker()
    marker_225 = _Marker()
    marker_227 = _Marker()
    marker_229 = _Marker()
    marker_231 = _Marker()
    marker_232 = _Marker()
    marker_233 = _Marker()
    marker_234 = _Marker()
    marker_235 = _Marker()
    marker_236 = _Marker()
    marker_237 = _Marker()

    def attn1_processor(
        attn_module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn_module.spatial_norm is not None:
            hidden_states = attn_module.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = 0

        sequence_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]

        if attention_mask is not None:
            attention_mask = attn_module.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn_module.heads, -1, attention_mask.shape[-1])

        if attn_module.group_norm is not None:
            hidden_states = attn_module.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = marker_199(hidden_states)
        query = attn_module.to_q(hidden_states)
        query = marker_205(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn_module.norm_cross:
            encoder_hidden_states = attn_module.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn_module.to_k(encoder_hidden_states)
        key = marker_206(key)
        value = attn_module.to_v(encoder_hidden_states)
        value = marker_207(value)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_module.heads

        query = query.view(batch_size, -1, attn_module.heads, head_dim)
        query = marker_208(query)
        query = query.transpose(1, 2)
        query = marker_209(query)
        key = key.view(batch_size, -1, attn_module.heads, head_dim)
        key = marker_210(key)
        key = key.transpose(1, 2)
        key = marker_211(key)
        value = value.view(batch_size, -1, attn_module.heads, head_dim)
        value = marker_212(value)
        value = value.transpose(1, 2)
        value = marker_213(value)

        if attn_module.norm_q is not None:
            query = attn_module.norm_q(query)
        if attn_module.norm_k is not None:
            key = attn_module.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = marker_214(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, attn_module.heads * head_dim)
        hidden_states = marker_216(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn_module.to_out[0](hidden_states)
        hidden_states = marker_218(hidden_states)
        hidden_states = attn_module.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn_module.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn_module.rescale_output_factor
        return hidden_states

    original_attn1_processor = tb.attn1.processor
    tb.attn1.set_processor(attn1_processor)

    def attn2_processor(
        attn_module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn_module.spatial_norm is not None:
            hidden_states = attn_module.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = 0

        sequence_length = hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]

        if attention_mask is not None:
            attention_mask = attn_module.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn_module.heads, -1, attention_mask.shape[-1])

        if attn_module.group_norm is not None:
            hidden_states = attn_module.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn_module.to_q(hidden_states)
        query = marker_223(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            add_prompt_alias_blobs(captured, encoder_hidden_states)
            if attn_module.norm_cross:
                encoder_hidden_states = attn_module.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn_module.to_k(encoder_hidden_states)
        key = marker_224(key)
        value = attn_module.to_v(encoder_hidden_states)
        value = marker_225(value)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_module.heads

        query = query.view(batch_size, -1, attn_module.heads, head_dim).transpose(1, 2)
        query = marker_227(query)
        key = key.view(batch_size, -1, attn_module.heads, head_dim).transpose(1, 2)
        key = marker_229(key)
        value = value.view(batch_size, -1, attn_module.heads, head_dim).transpose(1, 2)
        value = marker_231(value)

        if attn_module.norm_q is not None:
            query = attn_module.norm_q(query)
        if attn_module.norm_k is not None:
            key = attn_module.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = marker_232(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = marker_233(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, -1, attn_module.heads * head_dim)
        hidden_states = marker_234(hidden_states)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = marker_235(hidden_states)

        hidden_states = attn_module.to_out[0](hidden_states)
        hidden_states = marker_236(hidden_states)
        hidden_states = attn_module.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn_module.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn_module.rescale_output_factor
        hidden_states = marker_237(hidden_states)
        return hidden_states

    original_attn2_processor = tb.attn2.processor
    tb.attn2.set_processor(attn2_processor)

    handles.append(unet.time_embedding.linear_2.register_forward_hook(out_hook("155")))
    handles.append(marker_199.register_forward_hook(out_hook("199")))
    handles.append(marker_205.register_forward_hook(out_hook("205")))
    handles.append(marker_206.register_forward_hook(out_hook("206")))
    handles.append(marker_207.register_forward_hook(out_hook("207")))
    handles.append(marker_208.register_forward_hook(out_hook("208")))
    handles.append(marker_209.register_forward_hook(out_hook("209")))
    handles.append(marker_210.register_forward_hook(out_hook("210")))
    handles.append(marker_211.register_forward_hook(out_hook("211")))
    handles.append(marker_212.register_forward_hook(out_hook("212")))
    handles.append(marker_213.register_forward_hook(out_hook("213")))
    handles.append(marker_214.register_forward_hook(out_hook("214")))
    handles.append(marker_216.register_forward_hook(out_hook("216")))
    handles.append(marker_218.register_forward_hook(out_hook("218")))
    handles.append(marker_223.register_forward_hook(out_hook("223")))
    handles.append(marker_224.register_forward_hook(out_hook("224")))
    handles.append(marker_225.register_forward_hook(out_hook("225")))
    handles.append(marker_227.register_forward_hook(out_hook("227")))
    handles.append(marker_229.register_forward_hook(out_hook("229")))
    handles.append(marker_231.register_forward_hook(out_hook("231")))
    handles.append(marker_232.register_forward_hook(out_hook("232")))
    handles.append(marker_233.register_forward_hook(out_hook("233")))
    handles.append(marker_234.register_forward_hook(out_hook("234")))
    handles.append(marker_235.register_forward_hook(out_hook("235")))
    handles.append(marker_236.register_forward_hook(out_hook("236")))
    handles.append(marker_237.register_forward_hook(out_hook("237")))
    def restore() -> None:
        tb.attn1.set_processor(original_attn1_processor)
        tb.attn2.set_processor(original_attn2_processor)
        for handle in handles:
            handle.remove()

    setattr(pipe, "_sd_capture_restore", restore)
    return handles


def finalize_captured_blobs(captured: dict[str, torch.Tensor]) -> None:
    if "236" in captured and "220" in captured:
        captured["237"] = clone_tensor(captured["236"] + captured["220"])
    if "237" in captured:
        add_blob(captured, "238", captured["237"])
        add_blob(captured, "239", captured["237"])
    if "241" in captured and ("242" not in captured or "243" not in captured):
        left, right = torch.chunk(captured["241"], 2, dim=-1)
        captured["242"] = clone_tensor(left)
        captured["243"] = clone_tensor(right)
    if "243" in captured:
        captured["244"] = clone_tensor(F.gelu(captured["243"]))
    if "242" in captured and "244" in captured:
        captured["245"] = clone_tensor(captured["242"] * captured["244"])
    if "246" in captured and "238" in captured:
        captured["247"] = clone_tensor(captured["246"] + captured["238"])
    if "247" in captured:
        seq_len = captured["247"].shape[-2]
        channels = captured["247"].shape[-1]
        side = int(round(seq_len**0.5))
        if side * side == seq_len:
            reshaped = captured["247"].reshape(captured["247"].shape[0], side, side, channels)
            captured["248"] = clone_tensor(reshaped)
            captured["249"] = clone_tensor(reshaped.permute(0, 3, 1, 2).contiguous())
    if "251" in captured:
        add_blob(captured, "252", captured["251"])
        add_blob(captured, "253", captured["251"])
        add_blob(captured, "254", captured["251"])
    if "255" in captured:
        captured["256"] = clone_tensor(F.silu(captured["255"]))
    if "261" in captured:
        captured["262"] = clone_tensor(F.silu(captured["261"]))
    if "264" in captured:
        add_blob(captured, "265", captured["264"])
        add_blob(captured, "266", captured["264"])
    if "273" in captured:
        add_blob(captured, "274", captured["273"])
        add_blob(captured, "275", captured["273"])
        add_blob(captured, "276", captured["273"])
    if "313" in captured and ("314" not in captured or "315" not in captured):
        left, right = torch.chunk(captured["313"], 2, dim=-1)
        captured["314"] = clone_tensor(left)
        captured["315"] = clone_tensor(right)
    if "315" in captured:
        captured["316"] = clone_tensor(F.gelu(captured["315"]))
    if "314" in captured and "316" in captured:
        captured["317"] = clone_tensor(captured["314"] * captured["316"])
    if "318" in captured and "310" in captured:
        captured["319"] = clone_tensor(captured["318"] + captured["310"])
    if "319" in captured:
        seq_len = captured["319"].shape[-2]
        channels = captured["319"].shape[-1]
        side = int(round(seq_len**0.5))
        if side * side == seq_len:
            reshaped = captured["319"].reshape(captured["319"].shape[0], side, side, channels)
            captured["320"] = clone_tensor(reshaped)
            captured["321"] = clone_tensor(reshaped.permute(0, 3, 1, 2).contiguous())
    # 163 is the Swish output after time_embedding.linear_2, and 184 is one of its split aliases.
    if "155" in captured:
        captured["163"] = clone_tensor(F.silu(captured["155"]))
        captured["183"] = clone_tensor(captured["163"])
        captured["184"] = clone_tensor(captured["163"])


@torch.no_grad()
def main() -> int:
    args = parse_args()
    dump_dir: Path = args.dump_dir
    out_dir: Path = args.out_dir or (dump_dir / "python_ref_pnnx")

    requested = args.blob or DEFAULT_BLOBS
    unsupported = sorted(set(requested) - set(SUPPORTED_BLOBS))
    if unsupported:
        raise KeyError(f"unsupported blob requests: {', '.join(unsupported)}")

    prompt_path = dump_dir / f"prompt_{args.prompt_kind}_f32.bin"
    sample = load_f32(dump_dir / "unity_unet_in0_f32.bin", (1, 9, 64, 64)).float()
    timestep = load_f32(dump_dir / "unity_unet_timestep_f32.bin", (1,)).float()
    prompt = load_f32(prompt_path, (1, 77, 768)).float()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()

    captured: dict[str, torch.Tensor] = {}
    attach_capture_hooks(pipe, captured)
    add_prompt_alias_blobs(captured, prompt)

    try:
        out = pipe.unet(sample, timestep, encoder_hidden_states=prompt, return_dict=False)[0]
    finally:
        restore = getattr(pipe, "_sd_capture_restore", None)
        if restore is not None:
            restore()

    captured["out0"] = clone_tensor(out)
    finalize_captured_blobs(captured)
    out_dir.mkdir(parents=True, exist_ok=True)
    for blob in requested:
        if blob not in captured:
            raise KeyError(f"captured tensor missing for blob {blob}")
        flat = flatten_tensor(captured[blob])
        write_float_lines(out_dir / f"official_unet_blob_{blob}.txt", flat, HEADER_SHAPES.get(blob))
        write_stats(out_dir / f"official_unet_blob_{blob}_stats.txt", flat)

    final_name = f"official_unet_outout_{args.prompt_kind}.txt"
    final_flat = flatten_tensor(captured["out0"])
    write_float_lines(out_dir / final_name, final_flat, HEADER_SHAPES.get("out0"))
    write_stats(out_dir / f"{Path(final_name).stem}_stats.txt", final_flat)

    print(f"dump_dir={dump_dir}")
    print(f"out_dir={out_dir}")
    print(f"prompt_kind={args.prompt_kind}")
    print(f"exported={','.join(requested)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
