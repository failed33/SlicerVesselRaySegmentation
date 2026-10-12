"""ModelO's semantic ONNX tensor contract and reflected scalar config record."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .constants import FEATURE_COUNT, PATCH_SIZE, RAY_LENGTH
from .shaders import load_shader


@dataclass(frozen=True)
class OutputSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize


@cache
def config_dtype() -> np.dtype:
    """Support the scalar config ABI, rejecting implicit padding/layout changes."""
    parameter = next(
        p for p in load_shader("patch_inputs").metadata["parameters"] if p["name"] == "config"
    )
    fields = parameter["type"]["resultType"]["fields"]
    scalar_types = {"uint32": np.uint32, "int32": np.int32, "float32": np.float32}
    for index, field in enumerate(fields):
        if (
            field["type"]["kind"] != "scalar"
            or field["binding"]["size"] != 4
            or field["binding"]["offset"] != index * 4
        ):
            raise ValueError("Config must remain contiguous 32-bit scalar fields")
    return np.dtype(
        {
            "names": [f["name"] for f in fields],
            "formats": [scalar_types[f["type"]["scalarType"]] for f in fields],
            "offsets": [f["binding"]["offset"] for f in fields],
            "itemsize": len(fields) * 4,
        }
    )


def config_record(
    volume_shape: tuple[int, int, int],
    n_rays: int,
    patch_h0: float,
    patch_corner_zyx: tuple[int, int, int],
) -> np.ndarray:
    values: dict[str, int | float] = dict(
        zip(("depth", "height", "width"), volume_shape, strict=True)
    )
    values.update(dict(zip(("cornerZ", "cornerY", "cornerX"), patch_corner_zyx, strict=True)))
    values.update(rayCount=n_rays, h0=patch_h0)
    dtype = config_dtype()
    if set(values) != set(dtype.names or ()):
        raise ValueError("Shader config fields do not match the application contract")
    result = np.zeros(1, dtype=dtype)
    for name, value in values.items():
        result[name] = value
    return result


def output_specs(n_rays: int) -> tuple[OutputSpec, ...]:
    return (
        OutputSpec("ray_features", np.dtype(np.float32), (n_rays, RAY_LENGTH, FEATURE_COUNT)),
        OutputSpec("ray_mask", np.dtype(bool), (n_rays, RAY_LENGTH)),
        OutputSpec("origin_zyx_norm", np.dtype(np.float32), (n_rays, 3)),
        OutputSpec("direction_zyx_norm", np.dtype(np.float32), (n_rays, 3)),
        OutputSpec("voxel_indices", np.dtype(np.int64), (n_rays, RAY_LENGTH)),
        OutputSpec("ray_hu", np.dtype(np.float32), (n_rays, RAY_LENGTH)),
        OutputSpec("ct_patch", np.dtype(np.float32), (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)),
    )
