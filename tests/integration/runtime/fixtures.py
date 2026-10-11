"""Generate small shared ONNX/WGSL fixtures with an independent numerical oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ITERATIONS = 3
INPUT_SHAPE = (1, 2, 2, 2, 2)


def shaders(input_elements: int, output_elements: int, dtype: str) -> tuple[str, str]:
    scalar = "f16" if dtype == "float16" else "f32"
    extension = "enable f16;\n" if scalar == "f16" else ""
    producer = f"""{extension}
@group(0) @binding(0) var<storage, read_write> x: array<{scalar}>;
@compute @workgroup_size(64)
fn produce(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= {input_elements}u) {{ return; }}
    x[i] += {scalar}(f32(i % 7u + 1u) * 0.25);
}}
"""
    consumer = f"""{extension}
@group(0) @binding(0) var<storage, read> y: array<{scalar}>;
@group(0) @binding(1) var<storage, read> x: array<{scalar}>;
@group(0) @binding(2) var<storage, read_write> audit: array<f32>;
@compute @workgroup_size(64)
fn consume(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= {output_elements}u) {{ return; }}
    let step = u32(f32(x[0]) * 4.0) - 1u;
    audit[step * {output_elements}u + i] = f32(y[i]) * 2.0 + f32(i % 3u) * 0.25;
}}
"""
    return producer, consumer


def reference_conv_transpose(data: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Direct scatter convolution, independent of ORT's CPU or GPU implementations."""
    kernel = weights.shape[2:]
    shape = (1, weights.shape[1], *(size * stride for size, stride in zip(data.shape[2:], kernel)))
    output = np.broadcast_to(bias.reshape(1, -1, 1, 1, 1), shape).copy()
    for channel in range(data.shape[1]):
        for position in np.ndindex(data.shape[2:]):
            destination = tuple(slice(p * k, (p + 1) * k) for p, k in zip(position, kernel))
            output[(0, slice(None), *destination)] += (
                data[(0, channel, *position)] * weights[channel]
            )
    return output


def make_case(output: Path, kernel: tuple[int, int, int], dtype: str) -> dict[str, object]:
    np_dtype = np.dtype(dtype)
    name = f"convtranspose3d-{'x'.join(map(str, kernel))}-{dtype}"
    weights = ((np.arange(4 * prod(kernel)).reshape(2, 2, *kernel) % 5 - 2) * 0.25).astype(np_dtype)
    bias = np.array([0.25, -0.5], dtype=np_dtype)
    output_shape = (1, 2, *(size * stride for size, stride in zip(INPUT_SHAPE[2:], kernel)))
    tensor_type = TensorProto.FLOAT16 if dtype == "float16" else TensorProto.FLOAT
    graph = helper.make_graph(
        [
            helper.make_node(
                "ConvTranspose",
                ["x", "w", "b"],
                ["y"],
                strides=kernel,
                kernel_shape=kernel,
                pads=[0] * 6,
                dilations=[1] * 3,
                output_padding=[0] * 3,
                group=1,
                auto_pad="NOTSET",
            )
        ],
        name,
        [helper.make_tensor_value_info("x", tensor_type, INPUT_SHAPE)],
        [helper.make_tensor_value_info("y", tensor_type, output_shape)],
        [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    model_path = output / f"{name}.onnx"
    model_path.write_bytes(model.SerializeToString())
    producer, consumer = shaders(prod(INPUT_SHAPE), prod(output_shape), dtype)
    expected = []
    for step in range(1, ITERATIONS + 1):
        values = (((np.arange(prod(INPUT_SHAPE)) % 7 + 1) * 0.25) * step).astype(np_dtype)
        prediction = reference_conv_transpose(values.reshape(INPUT_SHAPE), weights, bias).reshape(
            -1
        )
        expected.extend(
            (prediction.astype(np.float32) * 2 + np.arange(prediction.size) % 3 * 0.25).tolist()
        )
    return {
        "name": name,
        "dtype": dtype,
        "input_shape": INPUT_SHAPE,
        "output_shape": output_shape,
        "input_elements": prod(INPUT_SHAPE),
        "output_elements": prod(output_shape),
        "model": model_path.name,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "producer": producer,
        "consumer": consumer,
        "iterations": ITERATIONS,
        "expected": expected,
        "atol": 0.001,
        "rtol": 0.001,
    }


def generate(output: Path) -> list[dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        make_case(output, kernel, dtype)
        for kernel in ((1, 2, 2), (2, 2, 2))
        for dtype in ("float32", "float16")
    ]
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "generated", "cases": len(generate(args.output))}))
