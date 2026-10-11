"""Ray direction selection and physical-distance sampling of ordered ZYX polylines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

_NEAR_ZERO = 1e-12
_MAX_UINT16_INDEX_COUNT = int(np.iinfo(np.uint16).max) + 1
_DEGENERATE_TANGENT = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def generate_fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniform directions on the unit sphere.

    Uses the Fibonacci spiral method. Deterministic for a given n.

    Returns:
        (n, 3) float64 array of unit vectors in ZYX order.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    # Latitude: evenly spaced in [-1, 1]
    z = 1 - 2 * indices / (n - 1) if n > 1 else np.array([0.0])
    radius = np.sqrt(1 - z * z)
    # Longitude: golden angle increments
    theta = 2 * np.pi * indices / golden_ratio
    y = radius * np.sin(theta)
    x = radius * np.cos(theta)
    # Return in ZYX order to match volume indexing convention
    return np.column_stack([z, y, x])


def select_top_by_tangent_weight(
    directions: np.ndarray,
    tangent: np.ndarray,
    num_rays: int,
    min_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top num_rays directions weighted by sin(angle_to_tangent).

    Returns:
        (indices, directions) -- indices into the input array, and the
        corresponding direction vectors.
    """
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < _NEAR_ZERO:
        idx = np.arange(min(num_rays, len(directions)))
        return idx, directions[:num_rays].copy()

    tangent_unit = tangent / tangent_norm
    cos_angles = directions @ tangent_unit
    weights = np.sqrt(np.maximum(1 - cos_angles**2, 0.0))

    valid = weights >= min_weight
    valid_idx = np.where(valid)[0]
    valid_weights = weights[valid]

    if len(valid_idx) < num_rays:
        rejected_idx = np.where(~valid)[0]
        rejected_weights = weights[~valid]
        order = np.argsort(-rejected_weights)
        extra_needed = num_rays - len(valid_idx)
        combined_idx = np.concatenate([valid_idx, rejected_idx[order[:extra_needed]]])
        return combined_idx, directions[combined_idx]

    order = np.argsort(-valid_weights)
    selected_idx = valid_idx[order[:num_rays]]
    return selected_idx, directions[selected_idx]


def weight_directions_by_tangent(
    directions: np.ndarray,
    tangent: np.ndarray,
    num_rays: int,
    min_weight: float = 0.1,
) -> np.ndarray:
    """Weight directions by sin(angle_to_tangent) and select top num_rays.

    Args:
        directions: (M, 3) unit vectors (pre-generated oversampled set).
        tangent: (3,) vessel tangent vector.
        num_rays: Number of effective rays to return.
        min_weight: Minimum weight threshold for rejection.

    Returns:
        (num_rays, 3) unit vectors sorted by descending weight.
    """
    _, selected = select_top_by_tangent_weight(directions, tangent, num_rays, min_weight)
    return selected


def compute_survivor_indices(
    base_dirs: np.ndarray,
    tangent: np.ndarray,
    num_rays: int,
    min_weight: float = 0.1,
) -> np.ndarray:
    """Compute survivor indices for tangent-weighted ray selection.

    Returns (num_rays,) uint16 array of indices into base_dirs.
    """
    if num_rays < 0:
        raise ValueError(f"num_rays must be non-negative, got {num_rays}")
    if num_rays > _MAX_UINT16_INDEX_COUNT:
        raise ValueError(f"num_rays exceeds uint16 capacity ({_MAX_UINT16_INDEX_COUNT})")
    if len(base_dirs) > _MAX_UINT16_INDEX_COUNT:
        raise ValueError(
            f"rays_oversample exceeds uint16 survivor-index capacity ({_MAX_UINT16_INDEX_COUNT})"
        )
    if len(base_dirs) < num_rays:
        raise ValueError(f"rays_oversample must be >= num_rays, got {len(base_dirs)} < {num_rays}")
    indices, _ = select_top_by_tangent_weight(base_dirs, tangent, num_rays, min_weight)
    return indices.astype(np.uint16)


def _sample_polyline_geometry(
    path_arr: np.ndarray,
    spacing_mm: float,
    spacing_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return voxel positions and physical unit tangents for one nonempty line."""
    path_phys = path_arr * spacing_arr
    deltas = np.diff(path_phys, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    if len(path_arr) == 1 or total_length < _NEAR_ZERO:
        return path_arr[:1].copy(), _DEGENERATE_TANGENT[None, :].copy()

    n_samples = max(1, int(total_length / spacing_mm) + 1)
    sample_dists = np.linspace(0, total_length, n_samples)

    seg_idx = np.searchsorted(cum_length, sample_dists, side="right") - 1
    np.clip(seg_idx, 0, len(path_arr) - 2, out=seg_idx)
    seg_start = cum_length[seg_idx]
    seg_len = seg_lengths[seg_idx]
    t = np.zeros_like(sample_dists)
    np.divide(sample_dists - seg_start, seg_len, out=t, where=seg_len > _NEAR_ZERO)
    np.clip(t, 0.0, 1.0, out=t)
    positions = path_arr[seg_idx] + t[:, None] * (path_arr[seg_idx + 1] - path_arr[seg_idx])

    lo = np.maximum(seg_idx - 1, 0)
    hi = np.minimum(seg_idx + 2, len(path_arr) - 1)
    tangents = (path_arr[hi] - path_arr[lo]).astype(np.float64) * spacing_arr
    tangent_norms = np.linalg.norm(tangents, axis=1)
    valid = tangent_norms > _NEAR_ZERO
    tangents[valid] /= tangent_norms[valid, None]
    tangents[~valid] = _DEGENERATE_TANGENT
    return positions, tangents


def _sample_polyline(
    path: Sequence[Sequence[float]],
    vol_float: np.ndarray,
    spacing_mm: float,
    spacing_arr: np.ndarray,
    edge_idx: int,
) -> list[dict]:
    """Add shared HU interpolation and line identity to sampled geometry."""
    path_arr = np.array(path, dtype=np.float64)
    positions, tangents = _sample_polyline_geometry(path_arr, spacing_mm, spacing_arr)
    hu_values = ndimage.map_coordinates(vol_float, positions.T, order=1, mode="nearest")
    return [
        {
            "position_zyx": pos,
            "edge_index": edge_idx,
            "tangent": tangent,
            "initial_hu": float(hu),
        }
        for pos, tangent, hu in zip(positions, tangents, hu_values)
    ]


def sample_polylines(
    polylines_zyx: Sequence[Sequence[Sequence[float]]],
    volume: np.ndarray,
    spacing_mm: float,
    spacing_zyx: tuple[float, float, float],
) -> list[dict]:
    """Sample ordered ZYX polylines at regular physical intervals.

    Each sample records position (float ZYX), edge index, local tangent,
    and initial HU (trilinear interpolation from volume at sample position).

    Args:
        polylines_zyx: Ordered lines of ZYX voxel coordinates.
        volume: 3D CT volume (ZYX) for HU sampling.
        spacing_mm: Physical distance between samples.
        spacing_zyx: Voxel spacing in mm.

    Returns:
        List of dicts with keys: position_zyx (3,), edge_index (int),
        tangent (3,) unit vector, initial_hu (float).
    """
    spacing_arr = np.asarray(spacing_zyx, dtype=np.float64)
    vol_float = volume.astype(np.float64) if volume.dtype != np.float64 else volume
    samples = []

    for edge_idx, path in enumerate(polylines_zyx):
        if len(path) == 0:
            continue
        samples.extend(_sample_polyline(path, vol_float, spacing_mm, spacing_arr, edge_idx))

    return samples
