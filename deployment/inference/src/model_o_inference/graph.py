"""Prepare the shipped ModelO graph for the selected GPU path.

Changes are held in memory. The trained artifact is never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import onnx
from onnx import TensorProto, helper, inliner

from .constants import FEATURE_CHANNELS, OUTPUT_NAME, PATCH_SIZE
from .shader_contract import output_specs

if TYPE_CHECKING:
    from dawn_ort import DawnHost, GpuSession


def prepare_model_o(path: str, n_rays: int | None) -> bytes:
    """Prepare identical graph bytes for native and browser acceptance tests.

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


def _lower_group_norm(model: onnx.ModelProto, opset: int) -> None:
    # ONNX's version-converting inliner needs explicit types for initializer arguments.
    model.graph.value_info.extend(
        helper.make_tensor_value_info(value.name, value.data_type, value.dims)
        for value in model.graph.initializer
    )
    types = {
        value.name: value.type.SerializeToString()
        for value in [*model.graph.input, *model.graph.value_info]
    }
    schema = onnx.defs.get_schema("GroupNormalization", opset)
    # ONNX exposes this bound method at runtime but omits it from its OpSchema stub.
    context_function = cast(
        Callable[[bytes, list[bytes]], bytes], getattr(schema, "get_context_dependent_function")
    )
    for index, node in enumerate(model.graph.node):
        if node.op_type != "GroupNormalization":
            continue
        function = onnx.FunctionProto.FromString(
            context_function(node.SerializeToString(), [types[name] for name in node.input])
        )
        function.domain, function.name = "vpt.deployment", f"Normalization_{index}"
        model.functions.append(function)
        node.domain, node.op_type = function.domain, function.name
    model.opset_import.append(helper.make_opsetid("vpt.deployment", 1))
    model.CopyFrom(inliner.inline_local_functions(model, convert_version=True))
    # Removed function intermediates may have stale inferred dimensions.
    del model.graph.value_info[:]
