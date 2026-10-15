"""Shared CT and ordered-polyline preparation for ModelO ray inputs.

This builds compact case metadata without casting profiles or creating a GPU
device. Polyline order defines the indices used for same-line patch gathering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import RayGeometryConfig
from .lattice import fixed_to_float, quantize_to_fixed
from .sampling import (
    compute_survivor_indices,
    generate_fibonacci_sphere,
    sample_polylines,
)


@dataclass(frozen=True)
class RayContext:
    """Fixed-point ray lattice and sampled polyline metadata for one CT grid."""

    ray_origin_hu: np.ndarray
    ray_lattice_fp: np.ndarray
    sample_positions: np.ndarray
    sample_edges: np.ndarray
    sample_hu: np.ndarray
    sample_ray_indices: list[np.ndarray]

    @property
    def n_rays(self) -> int:
        return int(self.ray_origin_hu.shape[0])


def prepare_ray_context(
    polylines_zyx: Sequence[np.ndarray | Sequence[Sequence[float]]],
    volume: np.ndarray,
    geometry: RayGeometryConfig,
    spacing_zyx: tuple[float, float, float],
) -> RayContext:
    """Sample polylines and select their tangent-weighted Q16 ray lattice.

    Coordinates are continuous voxel positions in the supplied CT's ZYX grid;
    spacing supplies physical distances for sampling and ray steps.
    """
    samples = sample_polylines(
        polylines_zyx,
        volume,
        geometry.sample_spacing_mm,
        spacing_zyx,
        origin_sampling_policy=geometry.origin_sampling_policy,
    )
    num_rays = int(geometry.num_rays)
    total_rays = len(samples) * num_rays
    ray_origin_hu = np.zeros(total_rays, dtype=np.float32)
    ray_lattice_fp = np.zeros((total_rays, 6), dtype=np.int32)
    sample_positions = np.zeros((len(samples), 3), dtype=np.float64)
    sample_edges = np.zeros(len(samples), dtype=np.int32)
    sample_hu = np.zeros(len(samples), dtype=np.float64)
    sample_ray_indices: list[np.ndarray] = []
    step_scale = geometry.step_size_mm / np.asarray(spacing_zyx, dtype=np.float64)

    base_dirs = generate_fibonacci_sphere(int(geometry.num_rays * geometry.rays_oversample_factor))
    for sample_idx, sample in enumerate(samples):
        pos = np.asarray(sample["position_zyx"], dtype=np.float64)
        block = slice(sample_idx * num_rays, (sample_idx + 1) * num_rays)
        survivor_indices = compute_survivor_indices(
            base_dirs,
            np.asarray(sample["tangent"], dtype=np.float64),
            num_rays,
            min_weight=geometry.min_tangent_weight,
        )
        directions = base_dirs[survivor_indices.astype(np.int64)]
        ray_origin_hu[block] = np.float32(sample["initial_hu"])
        ray_lattice_fp[block, :3] = quantize_to_fixed(np.broadcast_to(pos, (num_rays, 3)))
        ray_lattice_fp[block, 3:] = quantize_to_fixed(directions * step_scale)
        sample_positions[sample_idx] = (
            fixed_to_float(ray_lattice_fp[block.start, :3])
            if geometry.origin_sampling_policy == "max_spacing"
            else pos
        )
        sample_edges[sample_idx] = int(sample["edge_index"])
        sample_hu[sample_idx] = (
            float(np.float32(sample["initial_hu"]))
            if geometry.origin_sampling_policy == "max_spacing"
            else sample["initial_hu"]
        )
        sample_ray_indices.append(np.arange(block.start, block.stop, dtype=np.int64))

    return RayContext(
        ray_origin_hu=ray_origin_hu,
        ray_lattice_fp=ray_lattice_fp,
        sample_positions=sample_positions,
        sample_edges=sample_edges,
        sample_hu=sample_hu,
        sample_ray_indices=sample_ray_indices,
    )
