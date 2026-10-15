"""Physical sampling policies share Q16 geometry without changing serving defaults."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from model_o_inference.constants import model_o_geometry
from model_o_inference.geometry import OriginSamplingPolicy, RayGeometryConfig
from model_o_inference.lattice import fixed_to_float, quantize_to_fixed
from model_o_inference.planning import PatchPlan, build_patch_plans
from model_o_inference.preparation import RayContext, prepare_ray_context
from model_o_inference.sampling import (
    sample_origin_hu,
    sample_polyline_geometry,
    sample_polylines,
)
from scipy import ndimage


def test_serving_defaults_retain_floor_sampling_and_original_path_tangents() -> None:
    assert model_o_geometry().origin_sampling_policy == "legacy_serving"
    volume = np.full((8, 8, 8), 123.25, dtype=np.float32)
    paths = [[], [[2, 2, 2], [2, 2, 3], [2, 4, 3]]]
    samples = sample_polylines(paths, volume, 1.0, (2.0, 0.5, 0.7))
    np.testing.assert_array_equal([s["position_zyx"] for s in samples], paths[1][::2])
    expected = np.array([0.0, 1.0, 0.7]) / np.sqrt(1.49)
    np.testing.assert_allclose([s["tangent"] for s in samples], [expected, expected])
    assert [s["edge_index"] for s in samples] == [1, 1]
    assert [s["initial_hu"] for s in samples] == [123.25, 123.25]
    short = sample_polylines([[[2, 2, 2], [2, 2, 2.4]]], volume, 1.0, (1.0, 1.0, 1.0))
    np.testing.assert_array_equal([s["position_zyx"] for s in short], [[2, 2, 2]])


@pytest.mark.parametrize(
    "path,spacing",
    [
        ([[0, 0, 0], [0, 0, 1.8]], [1, 1, 1]),
        ([[0, 0, 0], [6, 6, 6]], [2, 0.5, 0.7]),
        ([[0, 0, 0], [2, 0, 0], [2, 3, 2]], [1.7, 0.4, 0.6]),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 4]], [0.5, 0.7, 2]),
    ],
)
def test_max_spacing_includes_endpoints_and_bounds_physical_gaps(
    path: list[list[float]], spacing: list[float]
) -> None:
    line, scale = np.asarray(path, dtype=np.float64), np.asarray(spacing, dtype=np.float64)
    positions, tangents = sample_polyline_geometry(line, 1.0, scale)
    length = np.linalg.norm(np.diff(line, axis=0) * scale, axis=1).sum()
    assert len(positions) == int(np.ceil(length)) + 1
    np.testing.assert_array_equal(positions[[0, -1]], line[[0, -1]])
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1, atol=1e-15)
    assert np.linalg.norm(np.diff(positions, axis=0) * scale, axis=1).max() <= 1 + 1e-14
    quantized = fixed_to_float(quantize_to_fixed(positions))
    tolerance = np.linalg.norm(scale) / (1 << 16)
    assert np.linalg.norm(np.diff(quantized, axis=0) * scale, axis=1).max() <= 1 + tolerance


def test_max_spacing_handles_short_degenerate_and_reversing_paths() -> None:
    spacing = np.ones(3)
    positions, tangents = sample_polyline_geometry(np.array([[4.0, 4, 4]]), 1.0, spacing)
    np.testing.assert_array_equal(positions, [[4, 4, 4]])
    np.testing.assert_array_equal(tangents, [[1, 0, 0]])
    repeated = np.array([[4.0, 4, 4], [4, 4, 4]])
    duplicate, _ = sample_polyline_geometry(repeated, 1.0, spacing)
    np.testing.assert_array_equal(duplicate, positions)
    short, _ = sample_polyline_geometry(np.array([[0.0, 0, 0], [0, 0, 0.4]]), 1.0, spacing)
    np.testing.assert_array_equal(short, [[0, 0, 0], [0, 0, 0.4]])
    _, cusp = sample_polyline_geometry(np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 0]]), 1, spacing)
    np.testing.assert_array_equal(cusp, [[1, 0, 0], [-1, 0, 0], [-1, 0, 0]])


@pytest.mark.parametrize("spacing_mm", [0, -1, float("nan"), float("inf")])
def test_origin_spacing_must_be_finite_and_positive(spacing_mm: float) -> None:
    with pytest.raises(ValueError, match="spacing_mm"):
        sample_polyline_geometry(np.array([[0.0, 0, 0], [1, 0, 0]]), spacing_mm, np.ones(3))


@pytest.mark.parametrize("spacing", [[1, 1], [1, 0, 1], [1, float("nan"), 1]])
def test_voxel_spacing_must_have_three_finite_positive_values(spacing: list[float]) -> None:
    with pytest.raises(ValueError, match="spacing_zyx"):
        sample_polyline_geometry(np.array([[0.0, 0, 0]]), 1, np.asarray(spacing))


@pytest.mark.parametrize("path", [[], [[0, 0]], [[0, float("nan"), 0]]])
def test_nonempty_geometry_rejects_malformed_coordinates(path: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="shape|finite"):
        sample_polyline_geometry(np.asarray(path), 1, np.ones(3))


def test_unknown_origin_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="origin sampling policy"):
        sample_polyline_geometry(
            np.array([[0.0, 0, 0]]), 1, np.ones(3), cast(OriginSamplingPolicy, "unknown")
        )


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_origin_hu_matches_float64_interpolation_without_casting_volume(
    dtype: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    z, y, x = np.indices((8, 8, 8))
    volume = (100 + 40 * z + 2 * y + x).astype(dtype)
    positions = np.array([[2, 3, 4], [2.5, 3.25, 4.5], [-0.25, 3, 4]], dtype=np.float64)
    expected = ndimage.map_coordinates(
        volume.astype(np.float64), positions.T, order=1, mode="nearest"
    )
    native = ndimage.map_coordinates
    seen: list[np.ndarray] = []

    def capture(source: np.ndarray, coordinates: np.ndarray, **kwargs) -> np.ndarray:
        seen.append(source)
        return native(source, coordinates, **kwargs)

    monkeypatch.setattr(ndimage, "map_coordinates", capture)
    actual = sample_origin_hu(volume, positions)
    assert seen[0] is volume
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [190, 211, 110])


def test_preparation_retains_legacy_positions_and_opt_in_q16_source_identity() -> None:
    z, y, x = np.indices((24, 24, 24))
    volume = (100 + 40 * z + 2 * y + x * 0.25).astype(np.float32)
    paths = [np.array([[12.123456, 12, 12.1], [12.123456, 12, 16.8]])]
    spacing = (0.8, 0.5, 0.5)
    legacy = prepare_ray_context(paths, volume, RayGeometryConfig(8, 1.5, 0.1, 1, 0.25, 2), spacing)
    modern = prepare_ray_context(
        paths, volume, RayGeometryConfig(8, 1.5, 0.1, 1, 0.25, 2, "max_spacing"), spacing
    )
    assert len(legacy.sample_positions) == 3
    assert len(modern.sample_positions) == 4
    np.testing.assert_array_equal(legacy.sample_positions[[0, -1]], paths[0])
    for sample, rows in enumerate(modern.sample_ray_indices):
        origin_fp = modern.ray_lattice_fp[rows, :3]
        np.testing.assert_array_equal(origin_fp, np.repeat(origin_fp[:1], 8, axis=0))
        np.testing.assert_array_equal(modern.sample_positions[sample], fixed_to_float(origin_fp[0]))
        assert modern.sample_hu[sample] == modern.ray_origin_hu[rows[0]]
    samples = sample_polylines(paths, volume, 1, spacing, origin_sampling_policy="max_spacing")
    expected_hu = np.asarray([s["initial_hu"] for s in samples], dtype=np.float32)
    np.testing.assert_array_equal(modern.ray_origin_hu, np.repeat(expected_hu, 8))
    assert modern.ray_lattice_fp.dtype == np.int32
    assert modern.ray_origin_hu.dtype == np.float32


@pytest.mark.parametrize("overlap", [None, 0.5])
def test_patch_plans_preserve_branch_identity_and_same_branch_gathers(
    overlap: float | None,
) -> None:
    positions = np.array([[1.9, 2.1, 3.7], [2.9, 2.1, 3.7], [1.9, 2.1, 3.7]])
    context = RayContext(
        np.zeros(6, dtype=np.float32),
        np.zeros((6, 6), dtype=np.int32),
        positions,
        np.array([0, 0, 2], dtype=np.int32),
        np.array([-12.3, 100, 9]),
        [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])],
    )
    paths = [positions[:2], np.empty((0, 3)), positions[2:]]
    plans = build_patch_plans(context, 4, overlap=overlap, polylines_zyx=paths)
    assert [p.branch_id for p in plans] == ([0, 0, 2] if overlap is None else [0, 2])
    for plan in plans:
        np.testing.assert_array_equal(plan.center_zyx, [1, 2, 3] if plan.h0 != 100 else [2, 2, 3])
        np.testing.assert_array_equal(
            plan.ray_indices, [0, 1, 2, 3] if plan.branch_id == 0 else [4, 5]
        )
    assert plans[0].h0 == 12.3
    assert PatchPlan(np.zeros(3), np.array([], dtype=np.int64), 1).branch_id == 0
