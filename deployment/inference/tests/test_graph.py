"""Validate the selected model contract before GPU allocation."""

import json

import onnx
import pytest
from model_o_inference.constants import FEATURE_CHANNELS
from model_o_inference.graph import prepare_model_o, validate_model_o
from model_o_inference.shader_contract import output_specs
from onnx import TensorProto, helper


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
