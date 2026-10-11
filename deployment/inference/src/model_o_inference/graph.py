"""Prepare the shipped ModelO graph for the selected GPU path.

Changes are held in memory. The trained artifact is never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import onnx
from dawn_ort import DawnHost, GpuSession
from onnx import helper, inliner


def prepare_model_o(path: str, n_rays: int | None) -> bytes:
    """Prepare identical graph bytes for native and browser acceptance tests.

    ``None`` retains dynamic ray dimensions for the metadata-fallback probe.
    """
    model = onnx.version_converter.convert_version(onnx.load(path), 24)
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "n_rays" and n_rays is not None:
                dim.dim_value = n_rays
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    _lower_group_norm(model, 24)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def create_model_o_session(host: DawnHost, path: str, n_rays: int) -> GpuSession:
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
