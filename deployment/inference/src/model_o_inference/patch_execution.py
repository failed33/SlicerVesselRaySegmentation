"""Device-independent patch submission and collection boundary."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Protocol

from .inputs import PackedRayCase

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort
    from numpy.typing import NDArray


class PatchPipeline[Pending](Protocol):
    @property
    def native_session(self) -> ort.InferenceSession: ...

    def submit(self, case: PackedRayCase, resources: ExitStack) -> Pending: ...

    def collect(self, readback: Pending) -> NDArray[np.bool_]: ...


type PatchFactory = Callable[[ExitStack, PackedRayCase], PatchPipeline[Any]]
