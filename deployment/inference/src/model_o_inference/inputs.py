"""Backend-independent packed Q16 geometry for ModelO patch inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import STEP_SIZE_MM
from .lattice import quantize_to_fixed


@dataclass(frozen=True)
class RayCase:
    volume: np.ndarray
    spacing: tuple[float, float, float]
    origin: np.ndarray
    directions: np.ndarray
    origin_hu: np.ndarray
    center: np.ndarray
    h0: float

    @property
    def n_rays(self) -> int:
        return len(self.origin)

    @property
    def lattice(self) -> np.ndarray:
        steps = self.directions * STEP_SIZE_MM / np.asarray(self.spacing)
        return quantize_to_fixed(np.concatenate([self.origin, steps], axis=1))

    @property
    def packed_geometry(self) -> np.ndarray:
        return pack_ray_geometry(self.lattice, self.origin_hu)


@dataclass(frozen=True)
class PackedRayCase:
    """Patch metadata and already packed HU/Q16 geometry for native inference."""

    volume: np.ndarray
    spacing: tuple[float, float, float]
    geometry: np.ndarray
    center: np.ndarray
    h0: float

    @property
    def n_rays(self) -> int:
        return len(self.geometry)

    @property
    def packed_geometry(self) -> np.ndarray:
        return self.geometry


def pack_ray_geometry(lattice_fp: np.ndarray, origin_hu: np.ndarray) -> np.ndarray:
    """Pack float32 HU and six int32 Q16 coordinates without float conversion."""
    if lattice_fp.dtype != np.int32 or lattice_fp.ndim != 2 or lattice_fp.shape[1] != 6:
        raise ValueError("Ray lattice must contain six int32 Q16 coordinates per ray")
    if origin_hu.dtype != np.float32 or origin_hu.shape != (len(lattice_fp),):
        raise ValueError("Ray origin HU must contain one float32 value per ray")
    return np.column_stack([origin_hu.view(np.uint32), lattice_fp.view(np.uint32)])
