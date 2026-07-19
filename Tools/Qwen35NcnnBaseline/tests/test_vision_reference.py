from __future__ import annotations

import unittest

import numpy as np

from vision_reference import reorder_patches_for_merge, rgb_to_duplicated_patches, target_image_size, vision_rope_2d


class VisionReferenceTests(unittest.TestCase):
    def test_target_is_patch_and_merge_aligned(self) -> None:
        height, width = target_image_size(721, 1281)
        self.assertEqual((height, width), (736, 1312))
        self.assertEqual(height % 32, 0)
        self.assertEqual(width % 32, 0)

    def test_merge_order(self) -> None:
        values = np.arange(16, dtype=np.int32)[:, None]
        actual = reorder_patches_for_merge(values, 4, 4).reshape(-1)
        expected = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
        np.testing.assert_array_equal(actual, expected)

    def test_patch_channels_are_duplicated(self) -> None:
        rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
        patches = rgb_to_duplicated_patches(rgb)
        self.assertEqual(patches.shape, (4, 3, 2, 16, 16))
        np.testing.assert_array_equal(patches[:, :, 0], patches[:, :, 1])
        expected = (np.float32(255) / np.float32(255.5) - np.float32(0.5)) / np.float32(0.5)
        np.testing.assert_allclose(patches, expected, rtol=0, atol=1e-7)

    def test_vision_rope_shape_and_origin(self) -> None:
        cosine, sine = vision_rope_2d(4, 6)
        self.assertEqual(cosine.shape, (24, 64))
        self.assertEqual(sine.shape, (24, 64))
        np.testing.assert_array_equal(cosine[0], np.ones(64, dtype=np.float32))
        np.testing.assert_array_equal(sine[0], np.zeros(64, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

