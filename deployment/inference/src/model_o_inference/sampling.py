"""Ray direction selection and physical-distance sampling of ordered ZYX polylines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from scipy import ndimage

from .geometry import OriginSamplingPolicy

_NEAR_ZERO = 1e-12
_MAX_UINT16_INDEX_COUNT = int(np.iinfo(np.uint16).max) + 1


class PolylineSample(TypedDict):
    position_zyx: np.ndarray
    edge_index: int
    tangent: np.ndarray
    initial_hu: float


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


def _validate_polyline(
    path_arr: np.ndarray,
    spacing_mm: float,
    spacing_arr: np.ndarray,
    origin_sampling_policy: OriginSamplingPolicy,
) -> None:
    if origin_sampling_policy not in {"max_spacing", "legacy_serving"}:
        raise ValueError("Unsupported origin sampling policy")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("spacing_mm must be finite and positive")
    if spacing_arr.shape != (3,) or not np.all(np.isfinite(spacing_arr) & (spacing_arr > 0)):
        raise ValueError("spacing_zyx must contain three finite positive values")
    if path_arr.ndim != 2 or path_arr.shape[1] != 3 or not len(path_arr):
        raise ValueError("Each nonempty polyline must have shape (N, 3)")
    if not np.isfinite(path_arr).all():
        raise ValueError("Polyline coordinates must be finite")


def sample_polyline_geometry(
    path_arr: np.ndarray,
    spacing_mm: float,
    spacing_arr: np.ndarray,
    origin_sampling_policy: OriginSamplingPolicy = "max_spacing",
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a nonempty line with explicit physical origin placement.

    ``max_spacing`` uses ceil(length/spacing)+1 and tangents on the resampled
    physical path. Adjacent chord lengths can be shorter at bends. Q16
    quantization happens later. ``legacy_serving`` retains the historical
    floor rule and tangents based on the original neighboring vertices.
    """
    _validate_polyline(path_arr, spacing_mm, spacing_arr, origin_sampling_policy)
    path_phys = path_arr * spacing_arr
    deltas = np.diff(path_phys, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    if len(path_arr) == 1 or total_length < _NEAR_ZERO:
        return path_arr[:1].copy(), _DEGENERATE_TANGENT[None, :].copy()

    n_samples = (
        max(2, int(np.ceil(total_length / spacing_mm)) + 1)
        if origin_sampling_policy == "max_spacing"
        else max(1, int(total_length / spacing_mm) + 1)
    )
    sample_dists = np.linspace(0, total_length, n_samples)

    seg_idx = np.searchsorted(cum_length, sample_dists, side="right") - 1
    np.clip(seg_idx, 0, len(path_arr) - 2, out=seg_idx)
    seg_start = cum_length[seg_idx]
    seg_len = seg_lengths[seg_idx]
    t = np.zeros_like(sample_dists)
    np.divide(sample_dists - seg_start, seg_len, out=t, where=seg_len > _NEAR_ZERO)
    np.clip(t, 0.0, 1.0, out=t)
    positions = path_arr[seg_idx] + t[:, None] * (path_arr[seg_idx + 1] - path_arr[seg_idx])

    if origin_sampling_policy == "max_spacing":
        return positions, _resampled_tangents(positions * spacing_arr)

    lo = np.maximum(seg_idx - 1, 0)
    hi = np.minimum(seg_idx + 2, len(path_arr) - 1)
    tangents = (path_arr[hi] - path_arr[lo]).astype(np.float64) * spacing_arr
    tangent_norms = np.linalg.norm(tangents, axis=1)
    valid = tangent_norms > _NEAR_ZERO
    tangents[valid] /= tangent_norms[valid, None]
    tangents[~valid] = _DEGENERATE_TANGENT
    return positions, tangents


def _resampled_tangents(physical: np.ndarray) -> np.ndarray:
    """Centered differences, one-sided endpoints, and deterministic cusp fallback."""
    tangents = np.empty_like(physical)
    tangents[0] = physical[1] - physical[0]
    tangents[-1] = physical[-1] - physical[-2]
    tangents[1:-1] = physical[2:] - physical[:-2]
    for index in np.flatnonzero(np.linalg.norm(tangents, axis=1) <= _NEAR_ZERO):
        following = physical[min(index + 1, len(physical) - 1)] - physical[index]
        preceding = physical[index] - physical[max(index - 1, 0)]
        tangents[index] = following if np.linalg.norm(following) > _NEAR_ZERO else preceding
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > _NEAR_ZERO
    tangents[valid] /= norms[valid, None]
    tangents[~valid] = _DEGENERATE_TANGENT
    return tangents


def sample_origin_hu(volume: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Trilinear origin HU with float64 output, without a whole-volume cast."""
    return ndimage.map_coordinates(volume, positions.T, output=np.float64, order=1, mode="nearest")


def sample_polylines(
    polylines_zyx: Sequence[np.ndarray | Sequence[Sequence[float]]],
    volume: np.ndarray,
    spacing_mm: float,
    spacing_zyx: tuple[float, float, float],
    *,
    origin_sampling_policy: OriginSamplingPolicy = "legacy_serving",
) -> list[PolylineSample]:
    """Sample ordered ZYX lines and trilinear HU on the supplied CT grid.

    The default retains historical serving geometry. Explicit ``max_spacing``
    bounds continuous polyline arclength before Q16 and includes endpoints;
    a degenerate branch produces one origin. Tangents are normalized in
    physical space. Ownership lookup remains the caller's responsibility.
    """
    spacing_arr = np.asarray(spacing_zyx, dtype=np.float64)
    samples: list[PolylineSample] = []
    for edge_idx, path in enumerate(polylines_zyx):
        if len(path) == 0:
            continue
        positions, tangents = sample_polyline_geometry(
            np.asarray(path, dtype=np.float64), spacing_mm, spacing_arr, origin_sampling_policy
        )
        hu_values = sample_origin_hu(volume, positions)
        samples.extend(
            PolylineSample(
                position_zyx=pos, edge_index=edge_idx, tangent=tangent, initial_hu=float(hu)
            )
            for pos, tangent, hu in zip(positions, tangents, hu_values, strict=True)
        )
    return samples
