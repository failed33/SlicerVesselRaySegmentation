"""Execute the same Slang sampler on CUDA, binding ORT-owned device tensors."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cuda_ort.host import checked

from .constants import PATCH_SIZE
from .inputs import PackedRayCase, RayCase
from .shader_contract import config_dtype, config_record, output_specs
from .shaders import Shader, load_shader

if TYPE_CHECKING:
    import onnxruntime as ort
    from cuda_ort.host import CudaHost


@cache
def _compile_cubin(capability: str) -> bytes:
    """NVRTC compiles generated CUDA; Slang is only needed during packaging."""
    shader = load_shader("patch_inputs", "cuda")
    nvrtc = import_module("cuda.bindings.nvrtc")
    prelude = Path(__file__).with_name("webgpu") / "generated/slang-cuda-prelude.h"
    program = checked(
        nvrtc.nvrtcCreateProgram(
            shader.code.encode(),
            b"patch_inputs.cu",
            1,
            [prelude.read_bytes()],
            [b"slang-cuda-prelude.h"],
        )
    )
    try:
        options = [
            b"--std=c++17",
            b"--fmad=false",
            f"--gpu-architecture=sm_{capability}".encode(),
        ]
        (status,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if int(status):
            log = bytearray(checked(nvrtc.nvrtcGetProgramLogSize(program)))
            checked(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(f"ModelO CUDA compilation failed: {log.decode(errors='replace')}")
        cubin = bytearray(checked(nvrtc.nvrtcGetCUBINSize(program)))
        checked(nvrtc.nvrtcGetCUBIN(program, cubin))
        return bytes(cubin)
    finally:
        checked(nvrtc.nvrtcDestroyProgram(program))


def _parameter_record(shader: Shader, storage: dict[str, tuple[int, int]]) -> np.ndarray:
    """Validate Slang's reflected pointer/count ABI instead of guessing offsets."""
    parameters = shader.metadata["parameters"]
    if set(storage) != {p["name"] for p in parameters}:
        raise ValueError("CUDA shader resources do not match supplied tensors")
    record = np.zeros((len(parameters), 2), np.uint64)
    for index, parameter in enumerate(parameters):
        binding = parameter["binding"]
        if binding["kind"] != "uniform" or binding["size"] != 16 or binding["offset"] != index * 16:
            raise ValueError("CUDA shader requires contiguous 64-bit pointer/count pairs")
        pointer, nbytes = storage[parameter["name"]]
        # Counts are in shader records. Reflection supplies the record stride.
        element = parameter["type"]["resultType"]
        stride = 8 if element["kind"] == "vector" else 4
        if element["kind"] == "struct":
            stride = len(element["fields"]) * 4
        if nbytes % stride:
            raise ValueError("CUDA buffer size must contain whole shader records")
        record[index] = (pointer, nbytes // stride)
    return record


class CudaPatchInputs:
    """Own the sampler module and tensors, borrowing the host's stream and CT."""

    def __init__(self, host: CudaHost, case: RayCase | PackedRayCase, volume: ort.OrtValue) -> None:
        self.host = host
        self._shader = load_shader("patch_inputs", "cuda")
        self.gpu_inputs = {
            spec.name: host.empty(spec.shape, spec.dtype) for spec in output_specs(case.n_rays)
        }
        self._storage = dict(
            self.gpu_inputs,
            config=host.empty((config_dtype().itemsize,), np.dtype(np.uint8)),
            volume_zyx=volume,
            ray_geometry=host.empty((case.n_rays, 7), np.dtype(np.uint32)),
        )
        # Each pipeline loads its own module so simultaneous requests cannot
        # overwrite another request's Slang global parameter block.
        self.module = checked(host.driver.cuModuleLoadData(_compile_cubin(host.capability)))
        try:
            self._kernel = checked(
                host.driver.cuModuleGetFunction(self.module, self._shader.entry_point.encode())
            )
            pointer, size = checked(
                host.driver.cuModuleGetGlobal(self.module, b"SLANG_globalParams")
            )
            record = _parameter_record(
                self._shader,
                {
                    name: (tensor.data_ptr(), tensor.tensor_size_in_bytes())
                    for name, tensor in self._storage.items()
                },
            )
            if size != record.nbytes:
                raise ValueError("CUDA global parameter size does not match reflection")
            checked(host.driver.cuMemcpyHtoD(pointer, record.ctypes.data, record.nbytes))
        except BaseException:
            checked(host.driver.cuModuleUnload(self.module))
            raise

    def generate(self, case: PackedRayCase) -> np.ndarray:
        depth, height, width = case.volume.shape
        z, y, x = case.center - PATCH_SIZE // 2
        config = config_record(
            (depth, height, width), case.n_rays, case.h0, (int(z), int(y), int(x))
        )
        self.host.upload(self._storage["config"], config)
        self.host.upload(self._storage["ray_geometry"], case.geometry)
        checked(
            self.host.driver.cuLaunchKernel(
                self._kernel,
                self._shader.workgroups(case.n_rays + PATCH_SIZE**3),
                1,
                1,
                self._shader.metadata["entryPoints"][0]["threadGroupSize"][0],
                1,
                1,
                0,
                self.host.stream,
                0,
                0,
            )
        )
        return config

    def close(self) -> None:
        try:
            self.host.synchronize()
        finally:
            checked(self.host.driver.cuModuleUnload(self.module))
