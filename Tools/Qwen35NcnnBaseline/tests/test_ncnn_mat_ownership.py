from __future__ import annotations

import unittest

import numpy as np

from run_ncnn_baseline import Qwen35TextBaseline


class NcnnMatOwnershipTests(unittest.TestCase):
    def test_single_token_rope_caches_do_not_alias(self) -> None:
        try:
            import ncnn
        except ImportError:
            self.skipTest("ncnn Python binding is not installed")

        runner = Qwen35TextBaseline.__new__(Qwen35TextBaseline)
        runner.ncnn = ncnn
        runner.rope_dim = 64
        runner.rope_theta = np.float32(10_000_000.0)

        cosine, sine = runner._rope(1, 8)
        cosine_values = np.asarray(cosine, dtype=np.float32).copy()
        sine_values = np.asarray(sine, dtype=np.float32).copy()

        self.assertEqual((1, 32), cosine_values.shape)
        self.assertEqual((1, 32), sine_values.shape)
        self.assertFalse(np.shares_memory(cosine_values, sine_values))
        self.assertAlmostEqual(float(np.cos(np.float32(8.0))), float(cosine_values[0, 0]), places=6)
        self.assertAlmostEqual(float(np.sin(np.float32(8.0))), float(sine_values[0, 0]), places=6)
        self.assertGreater(float(np.max(np.abs(cosine_values - sine_values))), 0.5)


if __name__ == "__main__":
    unittest.main()
