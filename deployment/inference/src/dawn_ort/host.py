"""Small ctypes Dawn host for compute and caller-owned inference buffers.

The generated module is build output. Handles here belong to that one library.
Callbacks stay alive until their future completes or the instance shuts down.
"""

from __future__ import annotations

import ctypes as ct
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np

from ._bindings import load_bindings
from .platforms import platform_config

_CONTEXT_IDS = count(1)


def address(handle: Any) -> int:
    return ct.cast(handle, ct.c_void_p).value or 0


@dataclass(eq=False)
class Buffer:
    host: DawnHost
    handle: Any
    size: int
    allocated_size: int

    def __post_init__(self) -> None:
        self.host.buffers.add(self)

    @property
    def device(self) -> object:
        return self.host

    @property
    def address(self) -> int:
        if self.closed:
            raise RuntimeError("Buffer owner has been released")
        return address(self.handle)

    @property
    def closed(self) -> bool:
        return not bool(self.handle)

    def close(self) -> None:
        if not self.closed:
            self.host.w.wgpuBufferRelease(self.handle)
            self.handle = None
            self.host.buffers.remove(self)

    def retain(self) -> Buffer:
        self.address
        self.host.w.wgpuBufferAddRef(self.handle)
        return Buffer(self.host, self.handle, self.size, self.allocated_size)


class Kernel:
    """Own a compiled pipeline and bind group across repeated dispatches."""

    def __init__(self, host: DawnHost, handles: dict[str, Any]) -> None:
        self.host = host
        self.handles = handles

    def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
        """Dispatch workgroup counts; shader geometry belongs to the caller."""
        if not self.handles:
            raise RuntimeError("Compute kernel has been released")
        w = self.host.w
        encoder = w.wgpuDeviceCreateCommandEncoder(self.host.device, None)
        compute = w.wgpuCommandEncoderBeginComputePass(encoder, None)
        w.wgpuComputePassEncoderSetPipeline(compute, self.handles["ComputePipeline"])
        w.wgpuComputePassEncoderSetBindGroup(compute, 0, self.handles["BindGroup"], 0, None)
        w.wgpuComputePassEncoderDispatchWorkgroups(compute, x, y, z)
        w.wgpuComputePassEncoderEnd(compute)
        w.wgpuComputePassEncoderRelease(compute)
        self.host._submit(encoder)

    def close(self) -> None:
        w = self.host.w
        for kind, handle in reversed(self.handles.items()):
            getattr(w, f"wgpu{kind}Release")(handle)
        self.handles.clear()


class DawnHost:
    instance: Any
    adapter: Any
    device: Any
    queue: Any

    def __init__(self, generated: Path) -> None:
        self.w, self.library = load_bindings(generated)
        self.backend = platform_config(sys.platform).backend
        self.context_id = next(_CONTEXT_IDS)
        self.buffers: set[Buffer] = set()
        self.errors: list[str] = []
        self.callbacks: list[Any] = []
        self.wait_count = 0
        self.readback_count = 0
        self.write_bytes = 0
        self.instance = self.adapter = self.device = self.queue = None
        w = self.w
        features = (w.WGPUInstanceFeatureName * 1)(w.WGPUInstanceFeatureName_TimedWaitAny)
        desc = w.WGPUInstanceDescriptor(requiredFeatureCount=1, requiredFeatures=features)
        self.instance = w.wgpuCreateInstance(ct.byref(desc))
        if not self.instance:
            raise RuntimeError("Dawn did not create an instance")
        with ExitStack() as resources:
            resources.callback(w.wgpuInstanceRelease, self.instance)
            options = w.WGPURequestAdapterOptions(
                backendType=getattr(w, f"WGPUBackendType_{self.backend}")
            )
            self.adapter = self._request(
                "RequestAdapter", w.wgpuInstanceRequestAdapter, self.instance, ct.byref(options)
            )
            resources.callback(w.wgpuAdapterRelease, self.adapter)
            limits = w.WGPULimits()
            if w.wgpuAdapterGetLimits(self.adapter, ct.byref(limits)) != w.WGPUStatus_Success:
                raise RuntimeError("Cannot query adapter limits")
            error_callback = w.WGPUUncapturedErrorCallback(
                lambda _dev, kind, msg, _one, _two: self.errors.append(
                    f"{kind}: {self._message(msg)}"
                )
            )
            self.callbacks.append(error_callback)
            supports_f16 = w.wgpuAdapterHasFeature(self.adapter, w.WGPUFeatureName_ShaderF16)
            device_features = (
                (w.WGPUFeatureName * 1)(w.WGPUFeatureName_ShaderF16) if supports_f16 else None
            )
            desc = w.WGPUDeviceDescriptor(
                requiredLimits=ct.pointer(limits),
                requiredFeatureCount=1 if supports_f16 else 0,
                requiredFeatures=device_features,
            )
            desc.uncapturedErrorCallbackInfo.callback = error_callback
            self.device = self._request(
                "RequestDevice", w.wgpuAdapterRequestDevice, self.adapter, ct.byref(desc)
            )
            resources.callback(w.wgpuDeviceRelease, self.device)
            self.queue = w.wgpuDeviceGetQueue(self.device)
            resources.callback(w.wgpuQueueRelease, self.queue)
            self.raise_errors()
            self._resources = resources.pop_all()

    def _request(self, kind: str, function: Any, *args: Any) -> Any:
        w, result = self.w, []
        callback = getattr(w, f"WGPU{kind}Callback")(lambda *values: result.extend(values[:-2]))
        self.callbacks.append(callback)
        info = getattr(w, f"WGPU{kind}CallbackInfo")(
            mode=w.WGPUCallbackMode_WaitAnyOnly, callback=callback
        )
        future = function(*args, info)
        wait = w.WGPUFutureWaitInfo(future=future)
        self.wait_count += 1
        status = w.wgpuInstanceWaitAny(self.instance, 1, ct.byref(wait), 10_000_000_000)
        if status != w.WGPUWaitStatus_Success or not wait.completed:
            raise RuntimeError(f"Dawn future wait failed: {status}")
        self.callbacks.remove(callback)
        if result[0] != getattr(
            w, f"WGPU{'MapAsync' if kind == 'BufferMap' else kind}Status_Success"
        ):
            raise RuntimeError(f"{kind} failed: {self._message(result[-1])}")
        return result[1] if len(result) == 3 else None

    @staticmethod
    def _message(view: Any) -> str:
        if not view.data:
            return ""
        length = -1 if view.length == 2**64 - 1 else view.length
        return ct.string_at(view.data, length).decode("utf-8", errors="replace")

    def raise_errors(self) -> None:
        if self.errors:
            raise RuntimeError("Dawn validation: " + "; ".join(self.errors))

    def buffer(self, size: int, *, staging: bool = False) -> Buffer:
        if type(size) is not int or size < 1:
            raise ValueError("Buffer size must be a positive integer")
        w = self.w
        usage = w.WGPUBufferUsage_CopyDst
        usage |= (
            w.WGPUBufferUsage_MapRead
            if staging
            else (w.WGPUBufferUsage_Storage | w.WGPUBufferUsage_CopySrc)
        )
        # ORT's vectorized f32 kernels access vec4 slots, including the final tail.
        allocated_size = (size + 15) // 16 * 16
        buf = Buffer(
            self, self._create("Buffer", size=allocated_size, usage=usage), size, allocated_size
        )
        self.raise_errors()
        return buf

    def write(self, buf: Buffer, values: np.ndarray) -> None:
        data = np.ascontiguousarray(values)
        if data.nbytes != buf.size:
            raise ValueError("Write size must equal the buffer size")
        # Queue writes require four-byte alignment even for byte tensors.
        if data.nbytes % 4:
            data = np.pad(data.view(np.uint8).reshape(-1), (0, -data.nbytes % 4))
        self.w.wgpuQueueWriteBuffer(self.queue, buf.handle, 0, data.ctypes.data, data.nbytes)
        self.write_bytes += data.nbytes
        self.raise_errors()

    def _submit(self, encoder: Any) -> None:
        w = self.w
        command = w.wgpuCommandEncoderFinish(encoder, None)
        try:
            commands = (w.WGPUCommandBuffer * 1)(command)
            w.wgpuQueueSubmit(self.queue, 1, commands)
        finally:
            w.wgpuCommandBufferRelease(command)
            w.wgpuCommandEncoderRelease(encoder)
        self.raise_errors()

    def compute(self, code: str, buffers: list[Buffer], workgroups: tuple[int, int, int]) -> None:
        kernel = self.kernel(code, buffers)
        try:
            kernel.dispatch(*workgroups)
        finally:
            kernel.close()

    def _create(self, kind: str, **fields: Any) -> Any:
        descriptor = getattr(self.w, f"WGPU{kind}Descriptor")(**fields)
        return getattr(self.w, f"wgpuDeviceCreate{kind}")(self.device, ct.byref(descriptor))

    def kernel(self, code: str, buffers: list[Buffer], *, entry_point: str = "main") -> Kernel:
        w = self.w
        source = w.WGPUShaderSourceWGSL(
            chain=w.WGPUChainedStruct(sType=w.WGPUSType_ShaderSourceWGSL), code=self._view(code)
        )
        shader = self._create(
            "ShaderModule", nextInChain=ct.cast(ct.pointer(source), ct.POINTER(w.WGPUChainedStruct))
        )
        pipeline = self._create(
            "ComputePipeline",
            compute=w.WGPUComputeState(module=shader, entryPoint=self._view(entry_point)),
        )
        layout = w.wgpuComputePipelineGetBindGroupLayout(pipeline, 0)
        entries = (w.WGPUBindGroupEntry * len(buffers))(
            *(
                w.WGPUBindGroupEntry(binding=i, buffer=b.handle, size=b.allocated_size)
                for i, b in enumerate(buffers)
            )
        )
        group = self._create("BindGroup", layout=layout, entryCount=len(buffers), entries=entries)
        kernel = Kernel(
            self,
            {
                "ShaderModule": shader,
                "ComputePipeline": pipeline,
                "BindGroupLayout": layout,
                "BindGroup": group,
            },
        )
        try:
            self.raise_errors()
        except Exception:
            kernel.close()
            raise
        return kernel

    def _view(self, text: str) -> Any:
        data = text.encode()
        view = self.w.WGPUStringView(length=len(data))
        view.data = ct.cast(ct.c_char_p(data), type(view.data))
        return view

    def stage_readback(self, buf: Buffer) -> Buffer:
        w = self.w
        staging = self.buffer(buf.size, staging=True)
        encoder = w.wgpuDeviceCreateCommandEncoder(self.device, None)
        w.wgpuCommandEncoderCopyBufferToBuffer(
            encoder, buf.handle, 0, staging.handle, 0, buf.allocated_size
        )
        self._submit(encoder)
        return staging

    def read_staging(self, staging: Buffer, dtype: Any = np.float32) -> np.ndarray:
        self.readback_count += 1
        w = self.w
        self._request(
            "BufferMap",
            w.wgpuBufferMapAsync,
            staging.handle,
            w.WGPUMapMode_Read,
            0,
            staging.allocated_size,
        )
        try:
            ptr = w.wgpuBufferGetConstMappedRange(staging.handle, 0, staging.allocated_size)
            values = np.frombuffer(ct.string_at(ptr, staging.size), dtype=dtype).copy()
        finally:
            w.wgpuBufferUnmap(staging.handle)
            staging.close()
        self.raise_errors()
        return values

    def read(self, buf: Buffer, dtype: Any = np.float32) -> np.ndarray:
        return self.read_staging(self.stage_readback(buf), dtype)

    def _wait_for_completion(self) -> None:
        self._request("QueueWorkDone", self.w.wgpuQueueOnSubmittedWorkDone, self.queue)

    def sync(self) -> None:
        self._wait_for_completion()
        self.raise_errors()

    def close(self) -> None:
        """Release a completed queue before reporting stored validation errors."""
        if not self.queue:
            return
        self._wait_for_completion()
        for buf in list(self.buffers):
            buf.close()
        self._resources.close()
        self.instance = self.adapter = self.device = self.queue = None
        self.callbacks.clear()
        self.raise_errors()
