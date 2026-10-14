"""Binary union matches maximum-logit decisions and preserves clipped geometry."""

import numpy as np
import pytest
from model_o_inference.stitching import MaskMerger, mask_from_logits


def test_zero_boundary_includes_positive_subnormals_and_excludes_signed_zero():
    smallest = np.nextafter(np.float32(0), np.float32(1))
    logits = np.array([-80, -smallest, -0.0, 0.0, smallest, 80], np.float32)
    np.testing.assert_array_equal(
        mask_from_logits(logits), [False, False, False, False, True, True]
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_logits_fail_before_thresholding(value):
    with pytest.raises(RuntimeError, match="non-finite"):
        mask_from_logits(np.array([-1, value, 1], np.float32))


def test_binary_union_matches_maximum_logits_with_overlap_and_uncovered_voxels():
    merger = MaskMerger((5, 5, 5), patch_size=2)
    reference = np.full((5, 5, 5), -np.inf, np.float32)
    generator = np.random.default_rng(9)
    for center in [(1, 1, 1), (2, 2, 2), (4, 4, 4)]:
        logits = generator.normal(size=(2,) * 3).astype(np.float32)
        merger.aggregate(mask_from_logits(logits), center)
        slices = tuple(slice(value - 1, value + 1) for value in center)
        np.maximum(reference[slices], logits, out=reference[slices])
    actual = merger.finalize()
    np.testing.assert_array_equal(actual, reference > 0)
    assert actual.dtype == bool and actual.nbytes == 5**3
    actual[:] = True
    assert not merger.finalize().all(), "A preview must not mutate the accumulator"


def test_clipped_corner_outside_patch_and_background_votes():
    merger = MaskMerger((3, 3, 3), patch_size=2)
    merger.aggregate(np.ones((2,) * 3, bool), (0, 0, 0))
    merger.aggregate(np.ones((2,) * 3, bool), (10, 10, 10))
    merger.aggregate(np.zeros((2,) * 3, bool), (1, 1, 1))
    expected = np.zeros((3,) * 3, bool)
    expected[0, 0, 0] = True
    np.testing.assert_array_equal(merger.finalize(), expected)


def test_probability_patches_are_rejected():
    with pytest.raises(TypeError, match="boolean"):
        MaskMerger((3, 3, 3), patch_size=2).aggregate(np.ones((2,) * 3, np.float32), (1, 1, 1))
