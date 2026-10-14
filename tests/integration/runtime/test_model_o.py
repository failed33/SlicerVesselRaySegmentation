"""The GPU fixture oracle must retain Q16 fractions and signed tie decisions."""

from __future__ import annotations

import unittest

import numpy as np
from model_o import reference
from model_o_inference.constants import FIXED_POINT_SCALE


class Q16ReferenceTest(unittest.TestCase):
    def inputs(self, coordinates: list[float], center: int):
        z = np.arange(4096)[:, None, None]
        volume = np.broadcast_to(np.where(z % 2 == 0, 32768, -32768), (4096, 3, 3)).astype(
            np.float32
        )
        lattice = np.zeros((len(coordinates), 6), np.int32)
        lattice[:, 0] = np.rint(np.array(coordinates) * FIXED_POINT_SCALE).astype(np.int32)
        lattice[:, 1:3] = FIXED_POINT_SCALE
        hu = np.ones(len(coordinates), np.float32)
        geometry = np.column_stack((hu.view(np.uint32), lattice.view(np.uint32)))
        return reference(volume, geometry, np.array([center, 1, 1]), 1.0)

    def test_interpolation_retains_one_q16_unit_at_large_coordinates(self):
        quantum = 1 / FIXED_POINT_SCALE
        for base in (300, 3072):
            outputs = self.inputs([base + 0.5 - quantum, base + 0.5 + quantum], base)
            np.testing.assert_array_equal(outputs["ray_hu"][:, 0], [1, -1])
            np.testing.assert_array_equal(outputs["ray_mask"], True)
            self.assertNotEqual(outputs["origin_zyx_norm"][0, 0], outputs["origin_zyx_norm"][1, 0])

    def test_negative_half_voxel_border_and_even_ties(self):
        quantum = 1 / FIXED_POINT_SCALE
        outputs = self.inputs([-0.5 - quantum, -0.5, -0.5 + quantum, 0.5, 1.5], 0)
        np.testing.assert_array_equal(outputs["ray_mask"][:, 0], [False, True, True, True, True])
        indices = outputs["voxel_indices"][:, 0]
        # The even rounded absolute Z coordinates are 0, 0, 0 and 2.
        np.testing.assert_array_equal(
            indices[1:], [24 * 48**2 + 24 * 48 + 24] * 3 + [26 * 48**2 + 24 * 48 + 24]
        )
        np.testing.assert_array_equal(outputs["ray_hu"][0], 0)
        np.testing.assert_array_equal(outputs["ray_features"][0], 0)


if __name__ == "__main__":
    unittest.main()
