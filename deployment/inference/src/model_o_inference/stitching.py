"""Patch mergers for ModelO patient-space outputs.

Two leaf mergers share one patch-placement convention but combine overlaps
differently:

* :class:`MaxMerger` -- per-voxel **maximum** across overlapping patches (a
  union). This is the ModelO production combiner: the model is trained
  *per-branch* (``build_entry``'s target is ``(branch_volume == branch_id) &
  gt_mask`` -- every *other* branch passing through a patch box is a negative),
  so a patch centered on branch B votes ~0 for a neighbouring branch C's voxels
  even though C is true vessel. Combining patches by **averaging** would let
  those cross-branch zero votes dilute true positives below threshold and
  collapse recall; the union keeps each branch's own confident prediction.
* :class:`HannMerger` -- Hann-windowed weighted **average** of overlapping
  patches. A standard, correct merge when every patch is an independent
  estimate of the *same* quantity; it is NOT used by the ModelO producer for
  the per-branch reason above, but is retained as a general leaf utility.

Per handoff §6: hand-rolled numpy; no monai (rejected the
``monai.inferers.AvgMerger`` subclass after weighing +750 MB install closure
for ~30 lines of numpy). Both mergers are pure-numpy leaf utilities with no
ONNX, Slicer, or producer machinery in scope -- their only internal import is
``constants.PATCH_SIZE``.

Convention (both): ``aggregate(patch_prob, center_zyx)`` expects ``patch_prob``
to be **sigmoided probabilities** ``(P, P, P) float32`` -- the per-patch sigmoid
is owned by the producer (handoff §6, Phase-3 split of concerns: the mergers are
strictly arithmetic). ``center_zyx`` is the patch *center*; the patch extends
``patch_size // 2`` voxels in each direction (matching
``_extraction.crop_ct_patch`` so mergers and extractor agree on origin).
"""

from __future__ import annotations

import numpy as np

from .constants import PATCH_SIZE

# Small finite divisor used in HannMerger.finalize() to avoid division by zero
# at voxels that no patch ever covered. Those voxels resolve to 0.0 because
# their numerator (``prob_sum``) is also zero -- the eps just keeps the division
# itself well-defined. Tuned to 1e-8 so any voxel with even one Hann tail
# contribution (typical minimum weight ~1e-4 for separable Hann at P=48) still
# divides through normally.
_EPS: float = 1e-8


# ----------------------------------------------------------- shared placement


def _check_init(
    volume_shape: tuple[int, int, int], patch_size: int
) -> tuple[tuple[int, int, int], int]:
    """Validate + normalize merger constructor args (shared by both mergers)."""
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be 3-D ZYX; got {volume_shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive; got {patch_size}")
    shape = (int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2]))
    return shape, int(patch_size)


def _check_patch(patch_prob: np.ndarray, patch_size: int) -> None:
    """Validate an incoming patch's shape + dtype (shared by both mergers).

    The dtype contract is explicit (no silent coercion): callers apply
    ``.astype(np.float32)`` after sigmoid.
    """
    if patch_prob.shape != (patch_size, patch_size, patch_size):
        raise ValueError(
            f"patch_prob must have shape ({patch_size}, {patch_size}, {patch_size}); "
            f"got {patch_prob.shape}"
        )
    if patch_prob.dtype != np.float32:
        raise TypeError(
            f"patch_prob must be float32; got {patch_prob.dtype}. "
            "Apply .astype(np.float32) after sigmoid."
        )


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


# ------------------------------------------------------------- Hann window


def _hann_window_1d(length: int) -> np.ndarray:
    """Symmetric 1D Hann window of the given length, peak at the centre.

    Uses the periodic-style formula ``0.5 * (1 - cos(2*pi*n / (N-1)))`` for
    ``N >= 2``. For ``N == 1`` returns ``[1.0]`` (degenerate but well-defined).
    Corners are 0.0; the central sample is 1.0 when ``N`` is odd, and
    ``cos(pi/(N-1))``-close to 1.0 when ``N`` is even (the two central samples
    are equal and just below 1.0).
    """
    if length <= 0:
        raise ValueError(f"length must be positive; got {length}")
    if length == 1:
        return np.ones((1,), dtype=np.float64)
    n = np.arange(length, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1)))


def _hann_window_3d(patch_size: int) -> np.ndarray:
    """Separable 3D Hann window of shape ``(P, P, P)`` as the outer product
    of three identical 1D Hann windows.

    Corners are 0.0; the central voxel reaches 1.0 (odd ``P``) or ``~1.0``
    (even ``P``) by the same logic as the 1D form.
    """
    w1d = _hann_window_1d(patch_size)
    return (w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]).astype(np.float32)


# ----------------------------------------------------------------- mergers


class MaxMerger:
    """Per-voxel **maximum** (union) patch merger -- the ModelO production combiner.

    Owns a single ``float32`` patient-space buffer ``prob_max``. Each
    ``aggregate(...)`` takes the elementwise max of the buffer and the incoming
    patch over the patch's in-bounds footprint; ``finalize()`` returns a copy.
    Uncovered voxels stay 0.0.

    Union (not average) is required by ModelO's per-branch training contract --
    see the module docstring.

    Args:
        volume_shape: ZYX shape of the patient-space accumulator buffer.
        patch_size: Cube edge length of incoming patches. Defaults to
            ``constants.PATCH_SIZE`` (= 48).
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        patch_size: int = PATCH_SIZE,
    ) -> None:
        self._volume_shape, self._patch_size = _check_init(volume_shape, patch_size)
        self._prob_max: np.ndarray = np.zeros(self._volume_shape, dtype=np.float32)

    def aggregate(
        self,
        patch_prob: np.ndarray,
        center_zyx: tuple[int, int, int],
    ) -> None:
        """Union a patch's per-voxel probabilities into the volume buffer."""
        _check_patch(patch_prob, self._patch_size)
        bounds = _patch_bounds(center_zyx, self._patch_size, self._volume_shape)
        if bounds is None:
            return
        dst, src = bounds
        np.maximum(self._prob_max[dst], patch_prob[src], out=self._prob_max[dst])

    def finalize(self) -> np.ndarray:
        """Return the per-voxel max probability map (fresh ``float32`` copy)."""
        return self._prob_max.copy()


class HannMerger:
    """Hann-windowed weighted-average patch merger (general leaf utility).

    Two float32 buffers (``prob_sum``, ``weight_sum``) of shape
    ``volume_shape``, plus a precomputed Hann window reused across every
    ``aggregate(...)`` call. ``finalize()`` returns ``prob_sum / max(weight_sum,
    eps)``.

    NOTE: not used by the ModelO producer -- averaging dilutes per-branch
    predictions (see module docstring); the producer uses :class:`MaxMerger`.

    Args:
        volume_shape: ZYX shape of the patient-space accumulator buffers.
        patch_size: Cube edge length of incoming patches. Defaults to
            ``constants.PATCH_SIZE`` (= 48).
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        patch_size: int = PATCH_SIZE,
    ) -> None:
        self._volume_shape, self._patch_size = _check_init(volume_shape, patch_size)
        self._prob_sum: np.ndarray = np.zeros(self._volume_shape, dtype=np.float32)
        self._weight_sum: np.ndarray = np.zeros(self._volume_shape, dtype=np.float32)
        # Computed once; reused across every aggregate() call.
        self._window: np.ndarray = _hann_window_3d(self._patch_size)

    def aggregate(
        self,
        patch_prob: np.ndarray,
        center_zyx: tuple[int, int, int],
    ) -> None:
        """Deposit a patch's Hann-weighted probabilities into the buffers."""
        _check_patch(patch_prob, self._patch_size)
        bounds = _patch_bounds(center_zyx, self._patch_size, self._volume_shape)
        if bounds is None:
            return
        dst, src = bounds
        window = self._window[src]
        self._prob_sum[dst] += patch_prob[src] * window
        self._weight_sum[dst] += window

    def finalize(self) -> np.ndarray:
        """Return ``vessel_prob = prob_sum / max(weight_sum, eps)``.

        Voxels with zero coverage end up at ``0.0`` (correct: they had no
        evidence). Always returns a fresh ``float32`` array; calling
        ``finalize()`` multiple times yields equivalent arrays (not consumed).
        """
        return (self._prob_sum / np.maximum(self._weight_sum, _EPS)).astype(np.float32)


__all__ = ["HannMerger", "MaxMerger"]
