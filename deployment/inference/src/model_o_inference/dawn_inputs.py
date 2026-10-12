"""Run the production ModelO ray shader on the Dawn device used by ORT.

Only compact geometry and patch metadata enter through Python. The seven
generated tensors remain in caller-owned buffers, in the shipped ONNX dtypes.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
from dawn_ort import DeviceTensor
from dawn_ort.host import Buffer, DawnHost

from .constants import PATCH_SIZE, STEP_SIZE_MM
from .lattice import quantize_to_fixed
from .shader_contract import config_dtype, config_record, output_specs
from .shaders import load_shader


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


class DawnPatchInputs:
    """Own patch buffers and a shader, optionally borrowing the request's CT."""

    def __init__(
        self,
        host: DawnHost,
        case: RayCase | PackedRayCase,
        *,
        ct_buffer: Buffer | None = None,
        initialize: bool = True,
    ) -> None:
        self.host = host
        self.case = case
        self.specs = output_specs(case.n_rays)
        self.shader = load_shader("patch_inputs")
        if ct_buffer is not None and (
            ct_buffer.host is not host or ct_buffer.closed or ct_buffer.size != case.volume.nbytes
        ):
            raise ValueError(
                "Borrowed CT buffer must be live on this host and match the volume size"
            )
        with ExitStack() as resources:
            config = host.buffer(config_dtype().itemsize)
            resources.callback(config.close)
            volume = ct_buffer
            if volume is None:
                volume = host.buffer(case.volume.nbytes)
                resources.callback(volume.close)
                host.write(volume, case.volume)
            geometry = host.buffer(case.n_rays * 7 * np.dtype(np.uint32).itemsize)
            resources.callback(geometry.close)
            self.storage = {"config": config, "volume_zyx": volume, "ray_geometry": geometry}
            for spec in self.specs:
                buffer = host.buffer(spec.nbytes)
                resources.callback(buffer.close)
                self.storage[spec.name] = buffer
            self.kernel = host.kernel(
                self.shader.code,
                self.shader.bindings(self.storage),
                entry_point=self.shader.entry_point,
            )
            resources.callback(self.kernel.close)
            if initialize:
                self.write_geometry(case.packed_geometry)
                self.generate(case.center, case.h0)
            self._resources = resources.pop_all()

    @property
    def input_buffers(self) -> dict[str, Buffer]:
        return {spec.name: self.storage[spec.name] for spec in self.specs}

    def write_geometry(self, geometry: np.ndarray) -> None:
        """Upload an existing packed lattice for this builder's ray capacity."""
        if geometry.dtype != np.uint32 or geometry.shape != (self.case.n_rays, 7):
            raise ValueError("Packed geometry must contain seven uint32 words per ray")
        self.host.write(self.storage["ray_geometry"], geometry)

    def generate(self, center: np.ndarray, h0: float) -> None:
        depth, height, width = self.case.volume.shape
        z, y, x = center - PATCH_SIZE // 2
        record = config_record(
            (depth, height, width), self.case.n_rays, h0, (int(z), int(y), int(x))
        )
        self.host.write(self.storage["config"], record)
        self.kernel.dispatch(self.shader.workgroups(self.case.n_rays + PATCH_SIZE**3))

    @property
    def gpu_inputs(self) -> dict[str, DeviceTensor]:
        return {
            spec.name: DeviceTensor(self.storage[spec.name], spec.shape, spec.dtype)
            for spec in self.specs
        }

    def close(self) -> None:
        self._resources.close()
