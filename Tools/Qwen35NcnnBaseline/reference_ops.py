"""FP32 NumPy gold implementations for Qwen3.5 ncnn custom operators."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


Array = np.ndarray


def _f32(value: Array, name: str) -> Array:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _sigmoid(value: Array) -> Array:
    value = np.asarray(value, dtype=np.float32)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (np.float32(1.0) + exp_value)
    return result


def _softplus(value: Array) -> Array:
    value = np.asarray(value, dtype=np.float32)
    return np.maximum(value, np.float32(0.0)) + np.log1p(np.exp(-np.abs(value)))


def short_conv(
    weight: Array,
    mixed_qkv: Array,
    conv_state: Optional[Array] = None,
    *,
    cache_length: Optional[int] = None,
) -> tuple[Array, Array]:
    """Depthwise causal Conv1D + SiLU with explicit streaming state.

    Shapes are weight=[groups, kernel], mixed_qkv=[seq, groups], and
    state=[cache, groups]. The ncnn reference returns kernel elements even
    though only kernel-1 elements are mathematically needed. cache_length
    defaults to kernel to preserve that runtime contract.
    """

    weight = _f32(weight, "weight")
    mixed_qkv = _f32(mixed_qkv, "mixed_qkv")
    if weight.ndim != 2 or mixed_qkv.ndim != 2:
        raise ValueError("weight and mixed_qkv must both be rank 2")
    groups, kernel_size = weight.shape
    seq_len, input_groups = mixed_qkv.shape
    if input_groups != groups or seq_len <= 0 or kernel_size <= 0:
        raise ValueError("invalid ShortConv shapes")

    if cache_length is None:
        cache_length = kernel_size
    if cache_length not in {max(kernel_size - 1, 0), kernel_size}:
        raise ValueError("cache_length must be kernel_size or kernel_size - 1")

    if conv_state is None:
        history = np.zeros((max(kernel_size - 1, 0), groups), dtype=np.float32)
    else:
        history = _f32(conv_state, "conv_state")
        if history.ndim != 2 or history.shape[1] != groups:
            raise ValueError("conv_state must have shape [cache, groups]")
        if history.shape[0] < max(kernel_size - 1, 0):
            raise ValueError("conv_state is shorter than kernel_size - 1")

    stated = np.concatenate((history, mixed_qkv), axis=0)
    prefix_len = history.shape[0]
    output = np.empty((seq_len, groups), dtype=np.float32)
    transposed_weight = weight.T
    for token_index in range(seq_len):
        end = prefix_len + token_index + 1
        window = stated[end - kernel_size : end]
        summed = np.sum(window * transposed_weight, axis=0, dtype=np.float32)
        output[token_index] = summed * _sigmoid(summed)

    state_out = stated[-cache_length:].copy() if cache_length else stated[:0].copy()
    return np.ascontiguousarray(output), np.ascontiguousarray(state_out)


def gated_delta_rule(
    a_log: Array,
    dt_bias: Array,
    b: Array,
    a: Array,
    query: Array,
    key: Array,
    value: Array,
    initial_state: Optional[Array] = None,
    *,
    epsilon: float = 1e-6,
) -> tuple[Array, Array]:
    """Recurrent Gated Delta Rule matching ref/ncnn_llm-main/src/utils/gdr.cpp.

    query/key=[seq, heads, k_dim], value=[seq, heads, v_dim], a/b=[seq,
    heads], a_log/dt_bias=[heads], and state=[heads, k_dim, v_dim]. All
    recurrence and accumulation stays FP32.
    """

    a_log = _f32(a_log, "a_log")
    dt_bias = _f32(dt_bias, "dt_bias")
    b = _f32(b, "b")
    a = _f32(a, "a")
    query = _f32(query, "query")
    key = _f32(key, "key")
    value = _f32(value, "value")

    if query.ndim != 3 or key.shape != query.shape or value.ndim != 3:
        raise ValueError("query/key/value must be [seq, heads, dim]")
    seq_len, heads, k_dim = query.shape
    if value.shape[:2] != (seq_len, heads):
        raise ValueError("value sequence/head dimensions do not match query")
    v_dim = value.shape[2]
    if a_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError("a_log and dt_bias must be [heads]")
    if a.shape != (seq_len, heads) or b.shape != (seq_len, heads):
        raise ValueError("a and b must be [seq, heads]")

    epsilon32 = np.float32(epsilon)
    q_norm = query / np.sqrt(np.sum(query * query, axis=2, keepdims=True, dtype=np.float32) + epsilon32)
    k_norm = key / np.sqrt(np.sum(key * key, axis=2, keepdims=True, dtype=np.float32) + epsilon32)
    beta = _sigmoid(b)
    decay_log = -np.exp(a_log)[None, :] * _softplus(a + dt_bias[None, :])

    if initial_state is None:
        state = np.zeros((heads, k_dim, v_dim), dtype=np.float32)
    else:
        state = _f32(initial_state, "initial_state").copy()
        if state.shape != (heads, k_dim, v_dim):
            raise ValueError("initial_state must be [heads, k_dim, v_dim]")

    output = np.empty((seq_len, heads, v_dim), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(k_dim))
    for token_index in range(seq_len):
        state *= np.exp(decay_log[token_index])[:, None, None]
        kv_memory = np.einsum("hk,hkv->hv", k_norm[token_index], state, optimize=False)
        delta = (value[token_index] - kv_memory) * beta[token_index, :, None]
        state += k_norm[token_index, :, :, None] * delta[:, None, :]
        output[token_index] = np.einsum(
            "hk,hkv->hv", q_norm[token_index], state, optimize=False
        ) * scale

    if output.dtype != np.float32 or state.dtype != np.float32:
        raise AssertionError("GatedDeltaRule left FP32")
    return np.ascontiguousarray(output), np.ascontiguousarray(state)

