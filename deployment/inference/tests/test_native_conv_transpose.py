"""Opt-in native GPU regression for ARG's two ConvTranspose3D kernel forms."""

from __future__ import annotations

import ctypes as ct
import json
import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import pytest

if TYPE_CHECKING:
    from dawn_ort import DawnHost


@pytest.fixture(scope="module")
def native_gpu() -> Iterator[tuple[DawnHost, Any, Path]]:
    assets_value = os.environ.get("DAWN_ORT_NATIVE_ASSETS")
    if assets_value is None:
        pytest.skip("Set DAWN_ORT_NATIVE_ASSETS to run native GPU regressions")
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort
    from dawn_ort import DawnHost
    from dawn_ort.platforms import platform_config

    assets = Path(assets_value).resolve()
    with ExitStack() as resources:
        ort.register_execution_provider_library(
            "native_convtranspose_test",
            str(assets / platform_config(sys.platform).provider_library),
        )
        resources.callback(ort.unregister_execution_provider_library, "native_convtranspose_test")
        host = DawnHost(assets / "webgpu.py")
        resources.callback(host.close)
        info = host.w.WGPUAdapterInfo()
        assert host.w.wgpuAdapterGetInfo(host.adapter, ct.byref(info)) == host.w.WGPUStatus_Success
        try:
            assert info.backendType == getattr(host.w, f"WGPUBackendType_{host.backend}")
            assert info.backendType != host.w.WGPUBackendType_Null
            assert info.adapterType != host.w.WGPUAdapterType_CPU
        finally:
            host.w.wgpuAdapterInfoFreeMembers(info)
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == "WebGpuExecutionProvider"
        ]
        assert devices, "Native WebGPU provider exposed no device"
        assert devices[0].device.type == ort.OrtHardwareDeviceType.GPU
        yield host, devices[0], assets.parent


def conv_transpose_model(
    kernel: tuple[int, int, int],
    dtype: type[np.float32] | type[np.float16],
) -> tuple[bytes, tuple[int, ...], tuple[int, ...]]:
    rng = np.random.default_rng(52688)
    shape = (1, 4, 3, 4, 5)
    weights = rng.normal(0, 0.1, (4, 3, *kernel)).astype(dtype)
    bias = rng.normal(0, 0.1, 3).astype(dtype)
    tensor_type = onnx.TensorProto.FLOAT16 if dtype is np.float16 else onnx.TensorProto.FLOAT
    output_shape = (1, 3, *(size * stride for size, stride in zip(shape[2:], kernel, strict=True)))
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ConvTranspose",
                ["x", "w", "b"],
                ["y"],
                strides=kernel,
                kernel_shape=kernel,
                group=1,
                pads=[0] * 6,
                dilations=[1] * 3,
                output_padding=[0] * 3,
                auto_pad="NOTSET",
            )
        ],
        "native_conv_transpose3d",
        [onnx.helper.make_tensor_value_info("x", tensor_type, shape)],
        [onnx.helper.make_tensor_value_info("y", tensor_type, output_shape)],
        [onnx.numpy_helper.from_array(weights, "w"), onnx.numpy_helper.from_array(bias, "b")],
    )
    model = onnx.helper.make_model(
        graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 18)]
    )
    onnx.checker.check_model(model)
    return model.SerializeToString(), shape, output_shape


def native_provider_options(
    host: DawnHost, device_owner: str, dtype: type[np.float32] | type[np.float16]
) -> dict[str, str]:
    from dawn_ort.host import address

    host.library.dawn_ort_get_native_procs.restype = ct.c_void_p
    table = host.library.dawn_ort_get_native_procs()
    assert table
    options = {
        "dawnProcTable": str(table),
        "preferredLayout": "NCHW",
        "enableGraphCapture": "0",
        "validationMode": "full",
    }
    if device_owner == "host":
        if dtype is np.float16:
            assert host.w.wgpuDeviceHasFeature(host.device, host.w.WGPUFeatureName_ShaderF16)
        options.update(
            deviceId=str(host.context_id),
            webgpuInstance=str(address(host.instance)),
            webgpuDevice=str(address(host.device)),
            preserveDevice="0",
        )
    return options


@pytest.mark.parametrize("kernel", [(1, 2, 2), (2, 2, 2)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("device_owner", ["provider", "host"])
def test_native_conv_transpose3d_matches_cpu(
    native_gpu: tuple[DawnHost, Any, Path],
    kernel: tuple[int, int, int],
    dtype: type[np.float32] | type[np.float16],
    device_owner: str,
) -> None:
    import onnxruntime as ort

    host, device, build = native_gpu
    rng = np.random.default_rng(52688)
    data, shape, output_shape = conv_transpose_model(kernel, dtype)
    cpu = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    logs = build / "logs"
    logs.mkdir(exist_ok=True)
    options.profile_file_prefix = str(
        logs / f"convtranspose-{device_owner}-{dtype.__name__}-{'x'.join(map(str, kernel))}"
    )
    options.add_provider_for_devices([device], native_provider_options(host, device_owner, dtype))
    gpu = ort.InferenceSession(data, sess_options=options, enable_fallback=False)
    with ExitStack() as resources:
        if device_owner == "host":
            # The shared runtime binds caller-owned buffers on the host queue.
            input_buffer = host.buffer(int(np.prod(shape)) * np.dtype(dtype).itemsize)
            resources.callback(input_buffer.close)
            output_buffer = host.buffer(int(np.prod(output_shape)) * np.dtype(dtype).itemsize)
            resources.callback(output_buffer.close)
            binding = gpu.io_binding()
            resources.callback(binding.clear_binding_inputs)
            resources.callback(binding.clear_binding_outputs)
            binding.bind_input("x", "webgpu", 0, dtype, shape, input_buffer.address)
            binding.bind_output("y", "webgpu", 0, dtype, output_shape, output_buffer.address)
        for _ in range(2):
            inputs = {"x": rng.normal(0, 0.1, shape).astype(dtype)}
            expected = cpu.run(None, inputs)[0]
            if device_owner == "host":
                host.write(input_buffer, inputs["x"])
                gpu.run_with_iobinding(binding)
                actual = host.read(output_buffer, dtype).reshape(output_shape)
            else:
                actual = gpu.run(None, inputs)[0]
            host.raise_errors()
            assert isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
            assert actual.dtype == expected.dtype == dtype
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    profile = json.loads(Path(gpu.end_profiling()).read_text())
    kernels = [
        event for event in profile if event.get("args", {}).get("op_name") == "ConvTranspose"
    ]
    assert len(kernels) == 2
    assert all(event["args"]["provider"] == "WebGpuExecutionProvider" for event in kernels)
