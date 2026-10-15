"""Prepare the shipped ModelO graph for the selected GPU path.

Changes are held in memory. The trained artifact is never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import onnx
from onnx import TensorProto, helper

from .constants import FEATURE_CHANNELS, OUTPUT_NAME, PATCH_SIZE
from .shader_contract import output_specs

if TYPE_CHECKING:
    from dawn_ort import DawnHost, GpuSession


def prepare_model_o(path: str, n_rays: int | None) -> bytes:
    """Prepare identical graph bytes for native and browser runtime paths.

    ``None`` retains dynamic ray dimensions for the metadata-fallback probe.
    """
    model = onnx.load(path)
    validate_model_o(model, n_rays)
    model = onnx.version_converter.convert_version(model, 24)
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "n_rays" and n_rays is not None:
                dim.dim_value = n_rays
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    _lower_group_norm(model, 24)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def validate_model_o(model: onnx.ModelProto, n_rays: int | None) -> None:
    """Reject incompatible exports before allocating GPU buffers or a session."""
    inputs = {value.name: value for value in model.graph.input}
    specs = output_specs(n_rays or 1)
    if set(inputs) != {spec.name for spec in specs}:
        raise ValueError("ModelO export does not match the seven required input tensors")
    types = {"float32": TensorProto.FLOAT, "bool": TensorProto.BOOL, "int64": TensorProto.INT64}
    for spec in specs:
        tensor = inputs[spec.name].type.tensor_type
        dimensions = tensor.shape.dim
        if tensor.elem_type != types[str(spec.dtype)] or len(dimensions) != len(spec.shape):
            raise ValueError(f"ModelO {spec.name} type/rank does not match the runtime")
        for axis, (actual, expected) in enumerate(zip(dimensions, spec.shape, strict=True)):
            if axis == 0 and spec.name != "ct_patch" and (actual.dim_param or n_rays is None):
                continue
            if actual.dim_value != expected:
                raise ValueError(
                    f"ModelO {spec.name} axis {axis}: expected {expected}, got {actual.dim_value}. "
                    f"Required ray features: {FEATURE_CHANNELS}"
                )
    outputs = list(model.graph.output)
    if len(outputs) != 1 or outputs[0].name != OUTPUT_NAME:
        raise ValueError("ModelO export must return only mask_logits")
    tensor = outputs[0].type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != [
        1,
        1,
        PATCH_SIZE,
        PATCH_SIZE,
        PATCH_SIZE,
    ]:
        raise ValueError("ModelO mask_logits must be float32 [1,1,48,48,48]")
    import json

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if "feature_channels" not in metadata:
        raise ValueError("ModelO export must declare its ordered feature_channels metadata")
    if json.loads(metadata["feature_channels"]) != list(FEATURE_CHANNELS):
        raise ValueError("ModelO feature channel order does not match normalized HU, position")
    _validate_sampling_metadata(metadata)


def _validate_sampling_metadata(metadata: dict[str, str]) -> None:
    """Missing policy metadata retains the original shipped sampling contract."""
    supported = {
        "ray_context_policy": "patch_truncated",
        "origin_sampling_policy": "legacy_serving",
        "origin_hu_policy": "trilinear",
        "source_grid": "native",
    }
    for field, expected in supported.items():
        actual = metadata.get(field, expected)
        if actual != expected:
            raise ValueError(
                f"ModelO {field}={actual!r} is unsupported; runtime requires {expected!r}"
            )


def create_model_o_session(host: DawnHost, path: str, n_rays: int) -> GpuSession:
    from dawn_ort import GpuSession

    return GpuSession(
        host,
        prepare_model_o(path, n_rays),
        provider_options={
            "preferredLayout": "NCHW",
            "enableInt64": "1",
            # Ray intermediates exceed the bucket cache's 160 MiB limit.
            "storageBufferCacheMode": "simple",
        },
    )


def _axiswise_mean(
    emit: Callable[..., str], constant: Callable[[list[int]], str], values: str, rank: int
) -> str:
    # Small reductions, from the last spatial axis through channels per group,
    # match the centered FP32 checkpoint export without one long accumulation.
    for axis in range(rank - 1, 1, -1):
        values = emit("ReduceMean", values, constant([axis]), keepdims=1)
    return values


def _group_norm_nodes(
    node: onnx.NodeProto, tensor: onnx.TypeProto.Tensor, prefix: str
) -> list[onnx.NodeProto]:
    attributes = {value.name: helper.get_attribute_value(value) for value in node.attribute}
    dimensions = tensor.shape.dim
    channels = dimensions[1].dim_value if len(dimensions) >= 2 else 0
    groups = int(attributes.get("num_groups", 0))
    if groups <= 0 or channels <= 0 or channels % groups:
        raise ValueError("ModelO GroupNormalization requires known channels divisible by groups")
    if attributes.get("stash_type", TensorProto.FLOAT) != TensorProto.FLOAT:
        raise ValueError("ModelO GroupNormalization requires FP32 statistics")
    nodes: list[onnx.NodeProto] = []

    def emit(op: str, *inputs: str, **attrs) -> str:
        output = f"{prefix}{len(nodes)}"
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output

    def constant(values: list[int] | list[float], dtype: int = TensorProto.INT64) -> str:
        return emit("Constant", value=helper.make_tensor("", dtype, [len(values)], values))

    input_shape = emit("Shape", node.input[0])
    batch = emit("Gather", input_shape, constant([0]), axis=0)
    spatial = emit("Gather", input_shape, constant(list(range(2, len(dimensions)))), axis=0)
    grouped_shape = emit("Concat", batch, constant([groups, channels // groups]), spatial, axis=0)
    fp32 = emit("Cast", node.input[0], to=TensorProto.FLOAT)
    grouped = emit("Reshape", fp32, grouped_shape)
    mean = _axiswise_mean(emit, constant, grouped, len(dimensions) + 1)
    centered = emit("Sub", grouped, mean)
    variance = _axiswise_mean(emit, constant, emit("Mul", centered, centered), len(dimensions) + 1)
    epsilon = constant([float(attributes.get("epsilon", 1e-5))], TensorProto.FLOAT)
    deviation = emit("Sqrt", emit("Add", variance, epsilon))
    inverse = emit("Reciprocal", deviation)
    normalized = emit("Reshape", emit("Mul", centered, inverse), input_shape)
    affine_shape = constant([1, channels] + [1] * (len(dimensions) - 2))
    scale = emit("Reshape", emit("Cast", node.input[1], to=TensorProto.FLOAT), affine_shape)
    bias = emit("Reshape", emit("Cast", node.input[2], to=TensorProto.FLOAT), affine_shape)
    affine = emit("Add", emit("Mul", normalized, scale), bias)
    nodes.append(
        helper.make_node("Cast", [affine], node.output, name=prefix + "output", to=tensor.elem_type)
    )
    return nodes


def _lower_group_norm(model: onnx.ModelProto, opset: int) -> None:
    """Use centered, axiswise FP32 statistics with unchanged affine parameters.

    The standard GroupNormalization/InstanceNormalization CPU kernels can lose
    accuracy on long spatial reductions. Native and browser paths use these
    same standard ONNX operations. No training framework is needed at runtime.
    """
    if opset < 21:
        raise ValueError("ModelO GroupNormalization lowering requires per-channel affine opset 21+")
    types = {
        value.name: value.type.tensor_type
        for value in [*model.graph.input, *model.graph.value_info]
    }
    names = {name for node in model.graph.node for name in [*node.input, *node.output]}
    names.update(value.name for value in [*model.graph.input, *model.graph.initializer])
    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "GroupNormalization" or node.domain not in ("", "ai.onnx"):
            nodes.append(node)
            continue
        prefix = f"_model_o_group_norm_{index}/"
        while any(name.startswith(prefix) for name in names):
            prefix = "_" + prefix
        nodes.extend(_group_norm_nodes(node, types[node.input[0]], prefix))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Replaced operator intermediates may have stale inferred dimensions.
    del model.graph.value_info[:]
