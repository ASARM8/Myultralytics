"""Endpoint ablation: same target and line interior, changed neighbor extension."""
import unittest

import numpy as np

from myscripts.wire_seg.geometry import WireCrop


class EndpointPromptTest(unittest.TestCase):
    def test_only_neighbor_extension_changes(self):
        boxes = [[32, 32, 40, 2, 0], [32, 35, 8, 2, 0]]
        crop = WireCrop(boxes[0], 12)
        target, finite = crop.prompts(boxes, 0, "axes")
        target_inf, infinite = crop.prompts(boxes, 0, "axes_infinite")
        np.testing.assert_array_equal(target, target_inf)
        mid = finite.shape[1] // 2
        np.testing.assert_array_equal(finite[:, mid-2:mid+2], infinite[:, mid-2:mid+2])
        self.assertGreater(float(infinite[:, 0].max()), .9)
        self.assertLess(float(finite[:, 0].max()), 1e-8)
        self.assertTrue(np.all(infinite >= finite))

    def test_remote_segment_supporting_line_is_not_culled(self):
        boxes = [[32, 32, 40, 2, 0], [300, 32, 8, 2, 0]]
        crop = WireCrop(boxes[0], 12)
        target, finite = crop.prompts(boxes, 0, "axes")
        target_inf, infinite = crop.prompts(boxes, 0, "axes_infinite")
        np.testing.assert_array_equal(target, target_inf)
        self.assertEqual(float(finite.max()), 0)
        self.assertGreater(float(infinite.max()), .9)

    def test_target_remains_finite_without_neighbors(self):
        box = [[32, 32, 8, 2, .7]]
        crop = WireCrop(box[0], 12)
        expected = crop.prompts(box, 0, "axes")
        actual = crop.prompts(box, 0, "axes_infinite")
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
