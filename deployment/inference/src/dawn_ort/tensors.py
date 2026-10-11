"""Contiguous native GPU tensors with explicit storage and ownership."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol

import numpy as np
from numpy.typing import DTypeLike


class GpuBufferOwner(Protocol):
    """Owned native buffer reference from one runtime and logical device.

    ``device`` is an identity token shared by the allocator and session. ``retain`` acquires an independent native reference.
    ``close`` releases that reference without destroying the underlying buffer.
    """

    @property
    def device(self) -> object: ...

    @property
    def address(self) -> int: ...

    @property
    def allocated_size(self) -> int: ...

    def retain(self) -> GpuBufferOwner: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DeviceTensor:
    """Contiguous tensor at buffer offset zero, with a strong owner reference.

    Shape describes logical elements. Storage includes the 16-byte allocation
    padding required by the tested ORT WebGPU path. A raw integer handle alone
    cannot establish device identity, storage size or lifetime.
    """

    owner: GpuBufferOwner
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(self, owner: GpuBufferOwner, shape: tuple[int, ...], dtype: DTypeLike) -> None:
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        self.__post_init__()

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError(f"GPU tensor needs positive integer dimensions: {self.shape}")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize

    def validate(self, device: object) -> None:
        if self.owner.device is not device:
            raise ValueError("GPU tensor belongs to a different runtime/device")
        if self.owner.address <= 0:
            raise ValueError("GPU tensor has a null native buffer handle")
        required = (self.nbytes + 15) // 16 * 16
        if self.owner.allocated_size < required:
            raise ValueError(f"GPU tensor requires {required} storage bytes including padding")
