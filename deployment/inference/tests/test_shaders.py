"""Keep generated storage layouts and the semantic ONNX contract aligned."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from model_o_inference.constants import INPUT_NAMES, SHADER_CONSTANTS
from model_o_inference.shader_contract import config_record, output_specs
from model_o_inference.shaders import Shader, load_shader


def test_reflection_binds_by_name_independently_of_mapping_order():
    shader = load_shader("patch_inputs")
    names = [p["name"] for p in shader.metadata["parameters"]]
    buffers = {name: object() for name in reversed(names)}
    bindings = shader.bindings(buffers)
    for parameter in shader.metadata["parameters"]:
        assert bindings[parameter["binding"]["index"]] is buffers[parameter["name"]]
    with pytest.raises(ValueError, match="resources"):
        shader.bindings({name: buffers[name] for name in names[1:]})


@pytest.mark.parametrize("change", ["group", "duplicate", "resource"])
def test_unsupported_reflection_fails_before_gpu_submission(change):
    metadata = copy.deepcopy(load_shader("patch_inputs").metadata)
    if change == "group":
        metadata["parameters"][0]["binding"]["space"] = 1
    elif change == "duplicate":
        metadata["parameters"][0]["binding"]["index"] = 1
    else:
        metadata["parameters"][0]["type"]["baseShape"] = "texture2D"
    with pytest.raises(ValueError):
        Shader("unused", metadata).bindings({p["name"]: object() for p in metadata["parameters"]})


def test_packed_records_and_final_onnx_tensor_types():
    shader = load_shader("patch_inputs")
    parameters = {p["name"]: p for p in shader.metadata["parameters"]}
    for name, words in (("ray_geometry", 7), ("origin_zyx_norm", 3), ("direction_zyx_norm", 3)):
        fields = parameters[name]["type"]["resultType"]["fields"]
        assert len(fields) == words
        assert [f["binding"]["offset"] for f in fields] == list(range(0, words * 4, 4))
        assert all(f["binding"]["size"] == 4 and f["type"]["kind"] == "scalar" for f in fields)
    record = config_record((11, 12, 13), 65, 123.5, (-24, 7, -9))
    assert record.nbytes == 32
    assert [record[name][0] for name in record.dtype.names] == [11, 12, 13, 65, -24, 7, -9, 123.5]
    specs = output_specs(65)
    assert tuple(s.name for s in specs) == INPUT_NAMES
    assert specs[1].dtype == np.dtype(bool) and specs[1].nbytes == 65 * 120
    assert specs[4].dtype == np.dtype(np.int64) and specs[4].nbytes == 65 * 120 * 8
    assert shader.workgroups(65 + 48**3) == (65 + 48**3 + 63) // 64


@pytest.mark.parametrize("target", ["wgsl", "slang", "constants"])
def test_resource_drift_is_rejected_without_a_runtime_compiler(monkeypatch, tmp_path: Path, target):
    import model_o_inference.shaders as shaders

    source = Path(shaders.__file__).with_name("webgpu")
    shutil.copytree(source, tmp_path / "webgpu")
    root = tmp_path / "webgpu"
    monkeypatch.setattr(shaders, "_ROOT", root)
    if target == "constants":
        path = root / "generated/patch_inputs.json"
        metadata = json.loads(path.read_text())
        metadata["build"]["constants"]["RAY_LENGTH"] = SHADER_CONSTANTS["RAY_LENGTH"] + 4
        path.write_text(json.dumps(metadata))
    else:
        path = root / ("generated/patch_inputs.wgsl" if target == "wgsl" else "patch_inputs.slang")
        path.write_text(path.read_text() + "\n// changed\n")
    load_shader.cache_clear()
    try:
        with pytest.raises(ValueError, match="digest mismatch|geometry disagree"):
            load_shader("patch_inputs")
    finally:
        load_shader.cache_clear()
