"""Fixed-point ray lattice shared by dataset, CPU, and WebGPU paths.

The discrete voxel a ray sample scatters into is part of the ModelO data
contract. Keep that decision in integer arithmetic so GPU f32 precision and
language-specific float rounding cannot move valid samples between voxels.
"""

from __future__ import annotations

import numpy as np

from .constants import FIXED_POINT_SCALE


def quantize_to_fixed(values: np.ndarray, scale: int = FIXED_POINT_SCALE) -> np.ndarray:
    """Quantize voxel-space floats to signed fixed-point ``int32``."""
    if scale <= 0:
        raise ValueError(f"scale must be positive; got {scale}")
    fixed64 = np.rint(np.asarray(values, dtype=np.float64) * float(scale)).astype(np.int64)
    info = np.iinfo(np.int32)
    if fixed64.size and (fixed64.min() < info.min or fixed64.max() > info.max):
        raise OverflowError("fixed-point ray lattice exceeds int32 range")
    return fixed64.astype(np.int32, copy=False)


def fixed_to_float(values: np.ndarray, scale: int = FIXED_POINT_SCALE) -> np.ndarray:
    """Convert fixed-point coordinates back to voxel-space float64."""
    if scale <= 0:
        raise ValueError(f"scale must be positive; got {scale}")
    return np.asarray(values, dtype=np.float64) / float(scale)


def quantize_ray_lattice(
    origin_zyx: np.ndarray,
    step_zyx_voxel: np.ndarray,
    scale: int = FIXED_POINT_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(origin_fp, step_fp)`` for one ray in canonical fixed-point form."""
    origin = np.asarray(origin_zyx, dtype=np.float64)
    step = np.asarray(step_zyx_voxel, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError(f"origin_zyx must have shape (3,), got {origin.shape}")
    if step.shape != (3,):
        raise ValueError(f"step_zyx_voxel must have shape (3,), got {step.shape}")
    return quantize_to_fixed(origin, scale), quantize_to_fixed(step, scale)


def round_fixed_to_voxel(values_fp: np.ndarray, scale: int = FIXED_POINT_SCALE) -> np.ndarray:
    """Round fixed-point coordinates to integer voxels using half-to-even.

    This is the integer equivalent of NumPy's ``round`` for values represented in
    Q-format. It is deliberately independent of floating-point precision.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive; got {scale}")
    if scale % 2 != 0:
        raise ValueError(f"scale must be even for exact tie handling; got {scale}")
    values = np.asarray(values_fp, dtype=np.int64)
    quotient = values // scale
    remainder = values - quotient * scale
    half = scale // 2
    above = remainder > half
    tie_to_odd = (remainder == half) & ((quotient & 1) != 0)
    return quotient + (above | tie_to_odd).astype(np.int64)


def ray_positions_from_lattice(
    origin_fp: np.ndarray,
    step_fp: np.ndarray,
    n_steps: int,
    scale: int = FIXED_POINT_SCALE,
) -> np.ndarray:
    """Return fixed-lattice sample positions as ``(n_steps, 3)`` float64."""
    positions_fp = ray_positions_fixed(origin_fp, step_fp, n_steps)
    return fixed_to_float(positions_fp, scale)


def ray_positions_fixed(origin_fp: np.ndarray, step_fp: np.ndarray, n_steps: int) -> np.ndarray:
    """Return fixed-point ray sample positions as ``(n_steps, 3)`` int64."""
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative; got {n_steps}")
    origin = np.asarray(origin_fp, dtype=np.int64)
    step = np.asarray(step_fp, dtype=np.int64)
    if origin.shape != (3,):
        raise ValueError(f"origin_fp must have shape (3,), got {origin.shape}")
    if step.shape != (3,):
        raise ValueError(f"step_fp must have shape (3,), got {step.shape}")
    offsets = np.arange(1, n_steps + 1, dtype=np.int64).reshape(-1, 1)
    return origin.reshape(1, 3) + offsets * step.reshape(1, 3)


def ray_voxels_from_lattice(
    origin_fp: np.ndarray,
    step_fp: np.ndarray,
    n_steps: int,
    scale: int = FIXED_POINT_SCALE,
) -> np.ndarray:
    """Return fixed-lattice sample voxels as ``(n_steps, 3)`` int64."""
    return round_fixed_to_voxel(ray_positions_fixed(origin_fp, step_fp, n_steps), scale)


__all__ = [
    "FIXED_POINT_SCALE",
    "fixed_to_float",
    "quantize_ray_lattice",
    "quantize_to_fixed",
    "ray_positions_fixed",
    "ray_positions_from_lattice",
    "ray_voxels_from_lattice",
    "round_fixed_to_voxel",
]
