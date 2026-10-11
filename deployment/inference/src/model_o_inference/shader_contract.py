"""ModelO WGSL uniform layout and seven tensor shapes."""

# TODO: replace with `SlangPy`

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MAX_TRAVEL_MM, PATCH_SIZE, RAY_LENGTH, STEP_SIZE_MM
from .lattice import FIXED_POINT_SCALE

_FEATURE_COUNT = 7


@dataclass(frozen=True)
class OutputSpec:
    binding: int
    dtype: np.dtype
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize


def config_buffers(
    *,
    volume_shape: tuple[int, int, int],
    n_rays: int,
    rays_oversample: int,
    spacing_zyx: tuple[float, float, float],
    patch_h0: float,
    patch_corner_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    cfg_u32 = np.array(
        [*volume_shape, n_rays, int(rays_oversample), FIXED_POINT_SCALE],
        dtype=np.uint32,
    )
    cfg_f32 = np.array(
        [
            float(spacing_zyx[0]),
            float(spacing_zyx[1]),
            float(spacing_zyx[2]),
            STEP_SIZE_MM,
            MAX_TRAVEL_MM,
            float(patch_h0),
            float(patch_corner_zyx[0]),
            float(patch_corner_zyx[1]),
            float(patch_corner_zyx[2]),
        ],
        dtype=np.float32,
    )
    return cfg_u32, cfg_f32


def output_specs(n_rays: int) -> tuple[OutputSpec, ...]:
    return (
        OutputSpec(5, np.dtype(np.float32), (n_rays, RAY_LENGTH, _FEATURE_COUNT)),
        OutputSpec(6, np.dtype(np.uint32), (n_rays, RAY_LENGTH)),
        OutputSpec(7, np.dtype(np.float32), (n_rays, 3)),
        OutputSpec(8, np.dtype(np.float32), (n_rays, 3)),
        OutputSpec(9, np.dtype(np.uint32), (n_rays, RAY_LENGTH)),
        OutputSpec(10, np.dtype(np.float32), (n_rays, RAY_LENGTH)),
        OutputSpec(11, np.dtype(np.float32), (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)),
    )
