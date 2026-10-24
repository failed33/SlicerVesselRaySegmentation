"""Validate the CUDA resource ABI on hosts without a CUDA device."""

import copy

import numpy as np
import pytest
from model_o_inference.cuda_inputs import _parameter_record
from model_o_inference.shaders import Shader, load_shader


def test_cuda_reflection_preserves_packed_onnx_buffers_and_pointer_order():
    shader = load_shader("patch_inputs", "cuda")
    sizes = {
        "config": 32,
        "volume_zyx": 64,
        "ray_geometry": 28,
        "ray_features": 120 * 2 * 4,
        "ray_mask": 120,
        "origin_zyx_norm": 12,
        "direction_zyx_norm": 12,
        "voxel_indices": 120 * 8,
        "ray_hu": 120 * 4,
        "ct_patch": 48**3 * 4,
    }
    buffers = {
        name: (0x1000 + index * 0x100, size) for index, (name, size) in enumerate(sizes.items())
    }
    record = _parameter_record(shader, dict(reversed(list(buffers.items()))))
    assert record.dtype == np.uint64
    assert record.nbytes == 160
    assert list(record[:, 1]) == [1, 16, 1, 240, 30, 1, 1, 120, 120, 48**3]
    assert list(record[:, 0]) == [buffers[p["name"]][0] for p in shader.metadata["parameters"]]
    assert shader.entry_point == "main_0"


def test_changed_cuda_layout_is_rejected_before_launch():
    shader = load_shader("patch_inputs", "cuda")
    metadata = copy.deepcopy(shader.metadata)
    metadata["parameters"][0]["binding"]["size"] = 8
    with pytest.raises(ValueError, match="pointer/count"):
        _parameter_record(
            Shader(shader.code, metadata), {p["name"]: (0x1000, 0) for p in metadata["parameters"]}
        )
