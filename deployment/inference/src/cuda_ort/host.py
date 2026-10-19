"""Own a CUDA primary context and stream; ORT owns tensor allocations."""

from __future__ import annotations

from contextlib import ExitStack
from importlib import import_module
from typing import Any

import numpy as np
import onnxruntime as ort


def checked(result: tuple[Any, ...]) -> Any:
    """Unwrap NVIDIA CUDA Python's status-first return convention."""
    status, *values = result
    if int(status) != 0:
        raise RuntimeError(f"CUDA operation failed: {status}")
    return values[0] if len(values) == 1 else values


class CudaHost:
    """Use logical device zero, honoring CUDA_VISIBLE_DEVICES and MIG isolation."""

    def __init__(self) -> None:
        self.driver = import_module("cuda.bindings.driver")
        driver = self.driver
        checked(driver.cuInit(0))
        device = checked(driver.cuDeviceGet(0))
        with ExitStack() as resources:
            context = checked(driver.cuDevicePrimaryCtxRetain(device))
            resources.callback(lambda: checked(driver.cuDevicePrimaryCtxRelease(device)))
            checked(driver.cuCtxPushCurrent(context))
            resources.callback(lambda: checked(driver.cuCtxPopCurrent()))
            self.stream = checked(driver.cuStreamCreate(0))
            resources.callback(lambda: checked(driver.cuStreamDestroy(self.stream)))
            attributes = driver.CUdevice_attribute
            major = checked(
                driver.cuDeviceGetAttribute(
                    attributes.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
                )
            )
            minor = checked(
                driver.cuDeviceGetAttribute(
                    attributes.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
                )
            )
            self.capability = f"{major}{minor}"
            self._resources = resources.pop_all()

    def empty(self, shape: tuple[int, ...], dtype: np.dtype) -> ort.OrtValue:
        return ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, "cuda", 0)

    def upload(self, target: ort.OrtValue, source: np.ndarray) -> None:
        checked(
            self.driver.cuMemcpyHtoDAsync(
                target.data_ptr(), source.ctypes.data, source.nbytes, self.stream
            )
        )

    def copy(self, target: ort.OrtValue, source: ort.OrtValue, nbytes: int) -> None:
        checked(
            self.driver.cuMemcpyDtoDAsync(target.data_ptr(), source.data_ptr(), nbytes, self.stream)
        )

    def synchronize(self) -> None:
        checked(self.driver.cuStreamSynchronize(self.stream))

    def close(self) -> None:
        try:
            self.synchronize()
        finally:
            self._resources.close()
