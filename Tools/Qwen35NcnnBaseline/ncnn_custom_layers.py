"""Python ncnn registrations for the two Qwen3.5 custom layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from reference_ops import gated_delta_rule, short_conv


_LIVE_LAYERS: list[Any] = []


def register_qwen35_custom_layers(net: Any) -> None:
    try:
        import ncnn
    except ImportError as exc:
        raise RuntimeError("the ncnn Python package is required; install requirements.txt") from exc

    class GatedDeltaRuleLayer(ncnn.Layer):
        def __init__(self) -> None:
            ncnn.Layer.__init__(self)
            self.one_blob_only = False
            self.support_inplace = False
            _LIVE_LAYERS.append(self)

        def forward(self, bottom_blobs: Any, top_blobs: Any, opt: Any) -> int:
            initial = None if bottom_blobs[7].empty() else np.asarray(bottom_blobs[7], dtype=np.float32)
            output, state = gated_delta_rule(
                np.asarray(bottom_blobs[0], dtype=np.float32).reshape(-1),
                np.asarray(bottom_blobs[1], dtype=np.float32).reshape(-1),
                np.asarray(bottom_blobs[2], dtype=np.float32),
                np.asarray(bottom_blobs[3], dtype=np.float32),
                np.asarray(bottom_blobs[4], dtype=np.float32),
                np.asarray(bottom_blobs[5], dtype=np.float32),
                np.asarray(bottom_blobs[6], dtype=np.float32),
                initial,
            )
            top_blobs[0].clone_from(ncnn.Mat(output), opt.blob_allocator)
            top_blobs[1].clone_from(ncnn.Mat(state), opt.blob_allocator)
            return -100 if top_blobs[0].empty() or top_blobs[1].empty() else 0

    class ShortConvLayer(ncnn.Layer):
        def __init__(self) -> None:
            ncnn.Layer.__init__(self)
            self.one_blob_only = False
            self.support_inplace = False
            _LIVE_LAYERS.append(self)

        def forward(self, bottom_blobs: Any, top_blobs: Any, opt: Any) -> int:
            mixed = np.asarray(bottom_blobs[1], dtype=np.float32)
            if mixed.ndim != 2:
                mixed = mixed.reshape(bottom_blobs[1].h, bottom_blobs[1].w)
            groups = mixed.shape[1]
            kernel_size = bottom_blobs[0].w
            weight = np.asarray(bottom_blobs[0], dtype=np.float32).reshape(groups, kernel_size)
            state = None
            if not bottom_blobs[2].empty():
                state = np.asarray(bottom_blobs[2], dtype=np.float32).reshape(-1, groups)
            output, state_out = short_conv(weight, mixed, state, cache_length=kernel_size)
            top_blobs[0].clone_from(ncnn.Mat(output), opt.blob_allocator)
            top_blobs[1].clone_from(ncnn.Mat(state_out), opt.blob_allocator)
            return -100 if top_blobs[0].empty() or top_blobs[1].empty() else 0

    def destroy(layer: Any) -> None:
        try:
            _LIVE_LAYERS.remove(layer)
        except ValueError:
            pass

    if net.register_custom_layer("GatedDeltaRule", GatedDeltaRuleLayer, destroy) != 0:
        raise RuntimeError("failed to register GatedDeltaRule")
    if net.register_custom_layer("ShortConv", ShortConvLayer, destroy) != 0:
        raise RuntimeError("failed to register ShortConv")

