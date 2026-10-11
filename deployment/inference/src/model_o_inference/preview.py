"""Optional observation of worker-owned cumulative ModelO segmentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import SimpleITK as sitk


@dataclass(frozen=True)
class PatchProgress:
    """Patch completions and wall time since patch execution began."""

    completed: int
    total: int
    elapsed_seconds: float


class PreviewObserver(Protocol):
    """Observe each merge without allocating a mask until a preview is wanted.

    ``snapshot`` returns an independent uint8 cumulative binary image in the
    submitted CT's physical grid. Call it synchronously inside ``on_progress``;
    retain the returned image if needed, rather than the snapshot callable.
    Observers own preview cadence. Exceptions propagate and release GPU resources.
    """

    def on_progress(self, progress: PatchProgress, snapshot: Callable[[], sitk.Image]) -> None: ...
