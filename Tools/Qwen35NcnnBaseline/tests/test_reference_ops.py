from __future__ import annotations

import unittest

import numpy as np

from reference_ops import gated_delta_rule, short_conv


class ShortConvTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.weight = rng.normal(0, 0.2, (6, 4)).astype(np.float32)
        self.values = rng.normal(0, 0.5, (9, 6)).astype(np.float32)

    def test_prefill_matches_token_decode_for_ncnn_cache(self) -> None:
        expected, expected_state = short_conv(self.weight, self.values)
        state = None
        actual_rows = []
        for row in self.values:
            output, state = short_conv(self.weight, row[None, :], state)
            actual_rows.append(output)
        np.testing.assert_allclose(np.concatenate(actual_rows), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(state, expected_state, rtol=0, atol=0)
        self.assertEqual(state.shape, (self.weight.shape[1], self.weight.shape[0]))

    def test_minimal_cache_has_equivalent_output(self) -> None:
        state = None
        rows = []
        for row in self.values:
            output, state = short_conv(
                self.weight,
                row[None, :],
                state,
                cache_length=self.weight.shape[1] - 1,
            )
            rows.append(output)
        expected, ncnn_state = short_conv(self.weight, self.values)
        np.testing.assert_allclose(np.concatenate(rows), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(state, ncnn_state[-(self.weight.shape[1] - 1) :])

    def test_zero_cache_matches_manual_first_window(self) -> None:
        output, _ = short_conv(self.weight, self.values[:1])
        manual_sum = self.values[0] * self.weight[:, -1]
        manual = manual_sum / (np.float32(1) + np.exp(-manual_sum))
        np.testing.assert_allclose(output[0], manual, rtol=1e-6, atol=1e-6)


class GatedDeltaRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        seq, heads, key_dim, value_dim = 8, 3, 5, 4
        self.args = (
            rng.normal(-1.0, 0.2, heads).astype(np.float32),
            rng.normal(0, 0.2, heads).astype(np.float32),
            rng.normal(0, 0.4, (seq, heads)).astype(np.float32),
            rng.normal(0, 0.4, (seq, heads)).astype(np.float32),
            rng.normal(0, 0.5, (seq, heads, key_dim)).astype(np.float32),
            rng.normal(0, 0.5, (seq, heads, key_dim)).astype(np.float32),
            rng.normal(0, 0.5, (seq, heads, value_dim)).astype(np.float32),
        )

    def test_prefill_matches_token_decode(self) -> None:
        expected, expected_state = gated_delta_rule(*self.args)
        state = None
        rows = []
        for token in range(self.args[2].shape[0]):
            token_args = self.args[:2] + tuple(value[token : token + 1] for value in self.args[2:])
            output, state = gated_delta_rule(*token_args, state)
            rows.append(output)
        np.testing.assert_allclose(np.concatenate(rows), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(state, expected_state, rtol=2e-6, atol=2e-6)

    def test_cache_continuity(self) -> None:
        split = 3
        first_args = self.args[:2] + tuple(value[:split] for value in self.args[2:])
        second_args = self.args[:2] + tuple(value[split:] for value in self.args[2:])
        first, state = gated_delta_rule(*first_args)
        second, state = gated_delta_rule(*second_args, state)
        expected, expected_state = gated_delta_rule(*self.args)
        np.testing.assert_allclose(np.concatenate((first, second)), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(state, expected_state, rtol=2e-6, atol=2e-6)

    def test_fp32_finite_and_deterministic(self) -> None:
        first_out, first_state = gated_delta_rule(*self.args)
        second_out, second_state = gated_delta_rule(*self.args)
        self.assertEqual(first_out.dtype, np.float32)
        self.assertEqual(first_state.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(first_out)))
        self.assertTrue(np.all(np.isfinite(first_state)))
        np.testing.assert_array_equal(first_out, second_out)
        np.testing.assert_array_equal(first_state, second_state)


if __name__ == "__main__":
    unittest.main()

