"""Run the production ModelO ray shader on the Dawn device used by ORT.

Only compact geometry and patch metadata enter through Python. The seven
generated tensors remain in caller-owned buffers, in the shipped ONNX dtypes.
"""

from __future__ import annotations

import re
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
from dawn_ort import DeviceTensor
from dawn_ort.host import Buffer, DawnHost

from .constants import INPUT_NAMES, PATCH_SIZE, STEP_SIZE_MM
from .lattice import quantize_to_fixed
from .shader_contract import config_buffers, output_specs
from .shaders import SHADER_PATH


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


# TODO: replace with `SlangPy`
def packed_shader() -> str:
    """Emit BOOL bytes and low/high INT64 words directly, without temporaries.

    L=120 is divisible by four. Each ray owns complete packed mask words, so
    its sequential writes need no inter-thread atomic read/modify/write.
    """
    source = SHADER_PATH.read_text()
    replacements = {
        "voxel_indices_u32: array<u32>": "voxel_indices_u32: array<vec2<u32>>",
        "voxel_indices_u32[idx] = flat;": "voxel_indices_u32[idx] = vec2<u32>(flat, 0u);",
        "voxel_indices_u32[idx] = 0u;": "voxel_indices_u32[idx] = padding_index();",
        "ray_mask_u32[idx] = select(0u, 1u, in_patch);": "write_mask(idx, select(0u, 1u, in_patch));",
        "ray_mask_u32[idx] = 0u;": "write_mask(idx, 0u);",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Production shader changed: expected one {old!r}")
        source = source.replace(old, new)
    source += """
fn write_mask(idx: u32, value: u32) {
    if idx % 4u == 0u { ray_mask_u32[idx / 4u] = 0u; }
    ray_mask_u32[idx / 4u] |= value << ((idx % 4u) * 8u);
}
fn padding_index() -> vec2<u32> {
    // CPU padding stores absolute voxel (0,0,0), even when its mask is false.
    let local = -patch_corner_zyx();
    return vec2<u32>(select(0u, patch_flat(local), local_in_patch(local)), 0u);
}
"""
    return _compact_bindings(source)


def _compact_bindings(source: str) -> str:
    """Fit Dawn Metal's ten-storage-buffer limit without tensor copies."""
    source = source.replace("@group(0) @binding(1) var<storage, read> cfg_f32: array<f32>;", "")
    source = re.sub(
        r"cfg_f32\[(\d+)\]", lambda m: f"bitcast<f32>(cfg_u32[{int(m[1]) + 6}])", source
    )
    source = source.replace(
        "@group(0) @binding(3) var<storage, read> ray_origin_hu: array<f32>;", ""
    )
    source = source.replace("ray_lattice_fp: array<i32>", "ray_geometry: array<u32>")
    source = source.replace("let base = ray * 6u;", "let base = ray * 7u;")
    source = re.sub(
        r"ray_lattice_fp\[base \+ (\d)u\]",
        lambda m: f"bitcast<i32>(ray_geometry[base + {int(m[1]) + 1}u])",
        source,
    )
    source = source.replace("ray_origin_hu[ray]", "bitcast<f32>(ray_geometry[ray * 7u])")
    remap = {0: 0, 2: 1, 4: 2, **{index: index - 2 for index in range(5, 12)}}
    return re.sub(r"@binding\((\d+)\)", lambda m: f"@binding({remap[int(m[1])]})", source)


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
        self.dtypes = {spec.binding: spec.dtype for spec in self.specs}
        self.dtypes.update({6: np.dtype(bool), 9: np.dtype(np.int64)})
        if ct_buffer is not None and (
            ct_buffer.host is not host or ct_buffer.closed or ct_buffer.size != case.volume.nbytes
        ):
            raise ValueError(
                "Borrowed CT buffer must be live on this host and match the volume size"
            )
        with ExitStack() as resources:
            config = host.buffer(15 * np.dtype(np.uint32).itemsize)
            resources.callback(config.close)
            volume = ct_buffer
            if volume is None:
                volume = host.buffer(case.volume.nbytes)
                resources.callback(volume.close)
                host.write(volume, case.volume)
            geometry = host.buffer(case.n_rays * 7 * np.dtype(np.uint32).itemsize)
            resources.callback(geometry.close)
            self.storage = [config, volume, geometry]
            for spec in self.specs:
                size = int(np.prod(spec.shape)) * self.dtypes[spec.binding].itemsize
                buffer = host.buffer(size)
                resources.callback(buffer.close)
                self.storage.append(buffer)
            self.kernel = host.kernel(packed_shader(), self.storage)
            resources.callback(self.kernel.close)
            if initialize:
                self.write_geometry(case.packed_geometry)
                self.generate(case.center, case.h0)
            self._resources = resources.pop_all()

    @property
    def input_buffers(self) -> dict[str, Buffer]:
        return dict(zip(INPUT_NAMES, self.storage[3:], strict=True))

    def write_geometry(self, geometry: np.ndarray) -> None:
        """Upload an existing packed lattice for this builder's ray capacity."""
        if geometry.dtype != np.uint32 or geometry.shape != (self.case.n_rays, 7):
            raise ValueError("Packed geometry must contain seven uint32 words per ray")
        self.host.write(self.storage[2], geometry)

    def generate(self, center: np.ndarray, h0: float) -> None:
        case = self.case
        depth, height, width = case.volume.shape
        z, y, x = center - PATCH_SIZE // 2
        cfg_u32, cfg_f32 = config_buffers(
            volume_shape=(depth, height, width),
            n_rays=case.n_rays,
            rays_oversample=case.n_rays,
            spacing_zyx=case.spacing,
            patch_h0=h0,
            patch_corner_zyx=(int(z), int(y), int(x)),
        )
        self.host.write(self.storage[0], np.concatenate([cfg_u32, cfg_f32.view(np.uint32)]))
        self.kernel.dispatch((case.n_rays + PATCH_SIZE**3 + 63) // 64)

    @property
    def gpu_inputs(self) -> dict[str, DeviceTensor]:
        return {
            name: DeviceTensor(buf, spec.shape, self.dtypes[spec.binding])
            for spec, (name, buf) in zip(self.specs, self.input_buffers.items(), strict=True)
        }

    def close(self) -> None:
        self._resources.close()
