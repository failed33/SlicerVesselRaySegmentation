"""Same-polyline spatial gathering for ModelO patch inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constants import PATCH_SIZE


class PatchSamples(Protocol):
    @property
    def sample_positions(self) -> np.ndarray: ...

    @property
    def sample_edges(self) -> np.ndarray: ...

    @property
    def sample_hu(self) -> np.ndarray: ...

    @property
    def sample_ray_indices(self) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class PatchPlan:
    """One patch: its center plus the gathered same-branch ray indices.

    Mirrors training's ``PatchDescriptor``. ``h0`` is the center sample's
    ``|HU|`` (training normalizes ``ct_patch`` by this).
    """

    center_zyx: np.ndarray  # (3,) int64
    ray_indices: np.ndarray  # (N,) int64
    h0: float
    branch_id: int = 0


def _in_box(points: np.ndarray, center: np.ndarray, patch_size: int) -> np.ndarray:
    delta = points - center
    half = patch_size / 2.0
    return np.all((delta >= -half) & (delta < half), axis=-1)


def _covered(points: np.ndarray, centers: np.ndarray, patch_size: int) -> np.ndarray:
    covered = np.zeros(len(points), dtype=bool)
    for center in centers:
        covered |= _in_box(points, center, patch_size)
    return covered


def _complete_coverage(
    positions: np.ndarray, indices: list[int], endpoints: np.ndarray, patch_size: int
) -> list[int]:
    """Retain every origin in a gather and cover origins/endpoints on the output grid."""
    gathered = _covered(positions, positions[indices], patch_size)
    placed = _covered(positions, positions[indices].astype(np.int64), patch_size)
    for index in range(len(positions)):
        if not gathered[index] or not placed[index]:
            indices.append(index)
            gathered |= _in_box(positions, positions[index], patch_size)
            placed |= _in_box(positions, positions[index].astype(np.int64), patch_size)
    for endpoint in endpoints:
        centers = positions[indices].astype(np.int64)
        if not _covered(endpoint[None], centers, patch_size)[0]:
            candidates = np.flatnonzero(_in_box(endpoint, positions.astype(np.int64), patch_size))
            if not len(candidates):
                raise ValueError("ModelO endpoint cannot be covered by a sampled-origin patch")
            indices.append(int(candidates[0]))
    return sorted(set(indices))


def _sparse_centers(
    positions: np.ndarray, polyline: np.ndarray, core_size: int, stride: float
) -> list[int]:
    """Choose existing origins by voxel arc length, independently for each branch."""
    distances = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    )
    midpoint = int(np.argmin(np.abs(distances - distances[-1] / 2.0)))
    if (
        _in_box(polyline, positions[midpoint].astype(np.int64), core_size).all()
        and _in_box(positions, positions[midpoint], core_size).all()
    ):
        return [midpoint]

    indices = [0]
    while indices[-1] < len(positions) - 1:
        following = int(np.searchsorted(distances, distances[indices[-1]] + stride, side="right"))
        indices.append(max(following - 1, indices[-1] + 1))
    return _complete_coverage(positions, indices, polyline[[0, -1]], core_size)


def _planning_sizes(
    patch_size: int, context_margin_voxels: int, gather_margin_voxels: int
) -> tuple[int, int]:
    for name, value in (
        ("context_margin_voxels", context_margin_voxels),
        ("gather_margin_voxels", gather_margin_voxels),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative integer")
    core_size = patch_size - 2 * context_margin_voxels
    if core_size < 2:
        raise ValueError("context_margin_voxels must leave a core at least two voxels wide")
    return core_size, patch_size + 2 * gather_margin_voxels


def build_patch_plans(
    rays: PatchSamples,
    patch_size: int = PATCH_SIZE,
    *,
    overlap: float | None = None,
    polylines_zyx: Sequence[np.ndarray | Sequence[Sequence[float]]] | None = None,
    context_margin_voxels: int = 0,
    gather_margin_voxels: int = 0,
) -> list[PatchPlan]:
    """Group rays into patches by same-branch spatial proximity.

    By default, mirror ``build_spatial_index`` with one patch per sample. A
    numeric overlap selects sparse centers while retaining all sampled rays.
    Original polylines supply endpoint coverage and the short-branch fit test.
    Overlap specifies the nominal voxel-arc stride ``patch_size * (1-overlap)``.
    A context margin selects sparse centers using the inner width
    ``patch_size - 2*context_margin_voxels``; coverage can add centers beyond the
    nominal schedule. Dense mode ignores the context margin.
    Every selected center gathers same-edge origins within the half-open width
    ``patch_size + 2*gather_margin_voxels``, centered on the continuous origin.
    The output center still truncates that origin to the integer grid, and the
    output crop remains ``patch_size``. Neither margin changes ray sampling.
    """
    core_size, gather_size = _planning_sizes(
        patch_size, context_margin_voxels, gather_margin_voxels
    )
    n_samples = len(rays.sample_positions)
    if n_samples == 0:
        return []

    if overlap is not None and not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be finite and in [0, 1), or None")
    positions = rays.sample_positions
    edges = rays.sample_edges

    # Pre-group sample indices by edge to avoid an O(S^2) per-center scan.
    edge_to_samples: dict[int, np.ndarray] = {
        int(e): np.where(edges == e)[0] for e in np.unique(edges)
    }

    centers = list(range(n_samples))
    if overlap is not None:
        centers = []
        for edge, samples in edge_to_samples.items():
            branch = positions[samples]
            polyline = branch if polylines_zyx is None else np.asarray(polylines_zyx[edge])
            stride = patch_size * (1.0 - overlap)
            centers.extend(samples[_sparse_centers(branch, polyline, core_size, stride)])
        centers.sort()

    plans: list[PatchPlan] = []
    for s in centers:
        center = positions[s]
        same_branch = edge_to_samples[int(edges[s])]
        in_box = _in_box(positions[same_branch], center, gather_size)
        selected = same_branch[in_box]

        gathered = [rays.sample_ray_indices[i] for i in selected]
        ray_indices = np.concatenate(gathered) if gathered else np.array([], dtype=np.int64)
        plans.append(
            PatchPlan(
                # Truncating cast (astype int64), matching build_spatial_index's
                # ``center.astype(np.int64)`` so the patch corner agrees with the
                # training grid and the encode-once torch reference.
                center_zyx=center.astype(np.int64),
                ray_indices=ray_indices,
                h0=abs(float(rays.sample_hu[s])),
                branch_id=int(edges[s]),
            )
        )
    return plans
