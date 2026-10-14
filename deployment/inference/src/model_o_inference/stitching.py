"""Finite-logit thresholding and binary union of branch-specific patch masks."""

from __future__ import annotations

import numpy as np

from .constants import LOGIT_THRESHOLD, PATCH_SIZE


def mask_from_logits(logits: np.ndarray) -> np.ndarray:
    """Reject invalid predictions before applying the strict zero boundary."""
    if logits.dtype != np.float32:
        raise TypeError("ModelO logits must be float32")
    if not np.isfinite(logits).all():
        raise RuntimeError("ModelO emitted non-finite logits")
    return logits > LOGIT_THRESHOLD


def _patch_bounds(
    center_zyx: tuple[int, int, int],
    patch_size: int,
    volume_shape: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]] | None:
    """Clipped ``(dst, src)`` slice triples for laying a patch into the volume.

    ``dst`` indexes the patient-space buffer; ``src`` indexes the patch. The
    patch is placed corner-first at ``center - patch_size // 2`` (matching
    ``crop_ct_patch``) and clipped to the volume. Returns ``None`` when the
    patch footprint lies entirely outside the volume (a no-op deposit).
    """
    cz, cy, cx = int(center_zyx[0]), int(center_zyx[1]), int(center_zyx[2])
    half = patch_size // 2
    z0, y0, x0 = cz - half, cy - half, cx - half
    vz, vy, vx = volume_shape
    dz0, dy0, dx0 = max(z0, 0), max(y0, 0), max(x0, 0)
    dz1, dy1, dx1 = (
        min(z0 + patch_size, vz),
        min(y0 + patch_size, vy),
        min(x0 + patch_size, vx),
    )
    if dz1 <= dz0 or dy1 <= dy0 or dx1 <= dx0:
        return None
    dst = (slice(dz0, dz1), slice(dy0, dy1), slice(dx0, dx1))
    src = (
        slice(dz0 - z0, dz1 - z0),
        slice(dy0 - y0, dy1 - y0),
        slice(dx0 - x0, dx1 - x0),
    )
    return dst, src


class MaskMerger:
    """Union boolean branch predictions; uncovered voxels remain background."""

    def __init__(self, volume_shape: tuple[int, int, int], patch_size: int = PATCH_SIZE) -> None:
        if len(volume_shape) != 3 or any(size <= 0 for size in volume_shape):
            raise ValueError("Volume shape must contain three positive dimensions")
        if patch_size <= 0:
            raise ValueError("Patch size must be positive")
        self._volume_shape = volume_shape
        self._patch_size = patch_size
        self._mask = np.zeros(volume_shape, dtype=bool)

    def aggregate(self, patch_mask: np.ndarray, center_zyx: tuple[int, int, int]) -> None:
        if patch_mask.dtype != np.dtype(bool):
            raise TypeError("Patch masks must be boolean")
        if patch_mask.shape != (self._patch_size,) * 3:
            raise ValueError("Patch mask shape does not match patch size")
        bounds = _patch_bounds(center_zyx, self._patch_size, self._volume_shape)
        if bounds is not None:
            destination, source = bounds
            self._mask[destination] |= patch_mask[source]

    def finalize(self) -> np.ndarray:
        """Copy the cumulative mask for final output or a requested preview."""
        return self._mask.copy()
