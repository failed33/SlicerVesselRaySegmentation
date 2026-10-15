"""Validate the selected model contract before GPU allocation."""

import json
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from model_o_inference.constants import FEATURE_CHANNELS
from model_o_inference.graph import _lower_group_norm, prepare_model_o, validate_model_o
from model_o_inference.shader_contract import output_specs
from onnx import TensorProto, helper, numpy_helper


def model():
    types = {"float32": TensorProto.FLOAT, "bool": TensorProto.BOOL, "int64": TensorProto.INT64}
    inputs = []
    for spec in output_specs(5):
        shape: list[int | str] = list(spec.shape)
        if spec.name != "ct_patch":
            shape[0] = "n_rays"
        inputs.append(helper.make_tensor_value_info(spec.name, types[str(spec.dtype)], shape))
    graph = helper.make_graph(
        [helper.make_node("Unsqueeze", ["ct_patch", "axes"], ["mask_logits"])],
        "contract",
        inputs,
        [helper.make_tensor_value_info("mask_logits", TensorProto.FLOAT, [1, 1, 48, 48, 48])],
        [helper.make_tensor("axes", TensorProto.INT64, [1], [0])],
    )
    result = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 24)])
    helper.set_model_props(result, {"feature_channels": json.dumps(list(FEATURE_CHANNELS))})
    return result


def test_selected_contract_prepares_dynamic_rays(tmp_path):
    source = tmp_path / "selected.onnx"
    onnx.save(model(), source)
    prepared = onnx.load_model_from_string(prepare_model_o(str(source), 65))
    assert [d.dim_value for d in prepared.graph.input[0].type.tensor_type.shape.dim] == [65, 120, 2]
    validate_model_o(prepared, 65)


@pytest.mark.parametrize("explicit_policies", [False, True])
def test_historical_and_explicit_supported_policies_execute_identically(
    tmp_path, explicit_policies
):
    candidate = model()
    metadata = {entry.key: entry.value for entry in candidate.metadata_props}
    if explicit_policies:
        metadata.update(
            ray_context_policy="patch_truncated",
            origin_sampling_policy="legacy_serving",
            origin_hu_policy="trilinear",
            source_grid="native",
        )
    helper.set_model_props(candidate, metadata)
    path = tmp_path / "accepted.onnx"
    onnx.save(candidate, path)
    prepared = onnx.load_model_from_string(prepare_model_o(str(path), 5))
    assert {entry.key: entry.value for entry in prepared.metadata_props} == metadata
    session = ort.InferenceSession(prepared.SerializeToString(), providers=["CPUExecutionProvider"])
    inputs = {spec.name: np.zeros(spec.shape, dtype=spec.dtype) for spec in output_specs(5)}
    inputs["ct_patch"][:] = np.arange(48**3).reshape(1, 48, 48, 48)
    actual = cast(np.ndarray, session.run(None, inputs)[0])
    np.testing.assert_array_equal(actual, inputs["ct_patch"][None])


@pytest.mark.parametrize(
    "field,value",
    [
        ("ray_context_policy", "full_physical"),
        ("ray_context_policy", "unknown"),
        ("origin_sampling_policy", "max_spacing"),
        ("origin_sampling_policy", "unknown"),
        ("origin_hu_policy", "nearest"),
        ("source_grid", "resampled"),
    ],
)
def test_shape_valid_incompatible_sampling_policies_fail_before_preparation(tmp_path, field, value):
    candidate = model()
    metadata = {entry.key: entry.value for entry in candidate.metadata_props}
    metadata[field] = value
    helper.set_model_props(candidate, metadata)
    path = tmp_path / "unsupported.onnx"
    onnx.save(candidate, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match=field):
        validate_model_o(candidate, 5)
    with pytest.raises(ValueError, match=field):
        prepare_model_o(str(path), 5)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    ["seven_channels", "channel_order", "missing_channels", "mask_type", "missing_input", "output"],
)
def test_incompatible_exports_fail_explicitly(change):
    candidate = model()
    if change == "seven_channels":
        candidate.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 7
    elif change == "channel_order":
        candidate.metadata_props[0].value = json.dumps(list(reversed(FEATURE_CHANNELS)))
    elif change == "missing_channels":
        del candidate.metadata_props[:]
    elif change == "mask_type":
        candidate.graph.input[1].type.tensor_type.elem_type = TensorProto.UINT32
    elif change == "missing_input":
        del candidate.graph.input[-1]
    else:
        candidate.graph.output[0].name = "probabilities"
    with pytest.raises(ValueError, match="ModelO|ray features"):
        validate_model_o(candidate, 65)


@pytest.mark.parametrize(
    ("shape", "groups", "dtype"),
    [
        ((2, 4), 1, np.float32),
        ((2, 4, 120), 2, np.float32),
        ((1, 8, 48, 48, 48), 4, np.float32),
        ((2, 4, 7, 8), 4, np.float16),
    ],
)
def test_centered_group_norm_preserves_affine_state_and_dtype(shape, groups, dtype):
    rng = np.random.default_rng(14)
    values = (rng.standard_normal(shape) * 0.25 + 4).astype(dtype)
    channels = shape[1]
    scale = np.linspace(0.5, 1.5, channels).astype(dtype)
    bias = np.linspace(-0.25, 0.5, channels).astype(dtype)
    epsilon = 0.002
    tensor_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    dimensions = ["batch", *shape[1:]]
    source = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "GroupNormalization",
                    ["x", "scale", "bias"],
                    ["y"],
                    num_groups=groups,
                    epsilon=epsilon,
                    stash_type=TensorProto.FLOAT,
                )
            ],
            "normalization",
            [helper.make_tensor_value_info("x", tensor_type, dimensions)],
            [helper.make_tensor_value_info("y", tensor_type, dimensions)],
            [numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "bias")],
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 24)],
    )
    initializer_bytes = [value.SerializeToString() for value in source.graph.initializer]
    _lower_group_norm(source, 24)
    onnx.checker.check_model(source)
    assert [value.SerializeToString() for value in source.graph.initializer] == initializer_bytes
    assert not any(
        node.op_type in {"GroupNormalization", "InstanceNormalization"}
        for node in source.graph.node
    )
    assert all(node.domain == "" for node in source.graph.node)
    for node in source.graph.node:
        if node.op_type == "ReduceMean":
            assert next(
                value for value in source.graph.node if value.output[0] == node.input[1]
            ).attribute[0].t.dims == [1]
    session = ort.InferenceSession(source.SerializeToString(), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": values})[0]
    assert isinstance(actual, np.ndarray)
    grouped = values.astype(np.float64).reshape(shape[0], groups, channels // groups, *shape[2:])
    axes = tuple(range(2, grouped.ndim))
    centered = grouped - grouped.mean(axis=axes, keepdims=True)
    normalized = centered / np.sqrt(np.mean(centered**2, axis=axes, keepdims=True) + epsilon)
    affine_shape = (1, channels, *([1] * (len(shape) - 2)))
    expected = (
        normalized.reshape(shape) * scale.reshape(affine_shape) + bias.reshape(affine_shape)
    ).astype(dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.002 if dtype == np.float16 else 0.00001,
        rtol=0.002 if dtype == np.float16 else 0.00001,
    )


@pytest.mark.parametrize(("channels", "groups"), [(0, 2), (5, 2), (4, 0)])
def test_group_norm_rejects_unrepresentable_group_shapes(channels, groups):
    source = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "GroupNormalization", ["x", "scale", "bias"], ["y"], num_groups=groups
                )
            ],
            "invalid",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, 3])],
            [],
        )
    )
    with pytest.raises(ValueError, match="known channels divisible"):
        _lower_group_norm(source, 24)


def test_preparation_is_idempotent_and_keeps_dynamic_ray_inputs(tmp_path):
    candidate = model()
    candidate.graph.node[0].output[0] = "patch"
    candidate.graph.node.append(
        helper.make_node(
            "GroupNormalization",
            ["patch", "scale", "bias"],
            ["mask_logits"],
            num_groups=1,
        )
    )
    candidate.graph.initializer.extend(
        [
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.25]),
            helper.make_tensor("bias", TensorProto.FLOAT, [1], [-0.5]),
            helper.make_tensor("_model_o_group_norm_1/0", TensorProto.FLOAT, [1], [10]),
        ]
    )
    path = tmp_path / "dynamic.onnx"
    onnx.save(candidate, path)
    first = prepare_model_o(str(path), None)
    path.write_bytes(first)
    assert prepare_model_o(str(path), None) == first
    prepared = onnx.load_model_from_string(first)
    assert prepared.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "n_rays"
    assert prepared.metadata_props == candidate.metadata_props
    assert not any(node.op_type == "GroupNormalization" for node in prepared.graph.node)
