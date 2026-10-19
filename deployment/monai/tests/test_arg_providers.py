"""Device-policy checks exercise selection without pretending to run ARG."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import onnxruntime as ort
import pytest
from arg_inference import providers


@pytest.fixture(autouse=True)
def clear_cuda_initialization_cache() -> Iterator[None]:
    providers.initialize_cuda.cache_clear()
    providers._initialize_plugin.cache_clear()
    yield
    providers.initialize_cuda.cache_clear()
    providers._initialize_plugin.cache_clear()


def _device(name: str, device_type: ort.OrtHardwareDeviceType) -> ort.OrtEpDevice:
    return cast(
        ort.OrtEpDevice, SimpleNamespace(ep_name=name, device=SimpleNamespace(type=device_type))
    )


def test_auto_keeps_npu_gpu_cpu_fallback_and_native_device_order() -> None:
    npu = _device("NpuProvider", ort.OrtHardwareDeviceType.NPU)
    preferred_gpu = _device("PreferredGpuProvider", ort.OrtHardwareDeviceType.GPU)
    second_gpu = _device("SecondGpuProvider", ort.OrtHardwareDeviceType.GPU)
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    selected = providers._select_devices([npu, preferred_gpu, second_gpu, cpu], {}, {}, 8)
    assert selected == [npu, preferred_gpu, cpu]


def test_auto_uses_cpu_when_no_accelerator_is_enumerated() -> None:
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    assert providers._select_devices([cpu], {}, {}, 8) == [cpu]


def test_auto_uses_gpu_when_no_npu_is_enumerated() -> None:
    gpu = _device("GpuProvider", ort.OrtHardwareDeviceType.GPU)
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    assert providers._select_devices([gpu, cpu], {}, {}, 8) == [gpu, cpu]


def test_auto_preserves_default_cpu_after_optimized_cpu() -> None:
    optimized = _device("OptimizedCpuProvider", ort.OrtHardwareDeviceType.CPU)
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    assert providers._select_devices([optimized, cpu], {}, {}, 8) == [optimized, cpu]


def test_auto_respects_native_selection_capacity() -> None:
    npu = _device("NpuProvider", ort.OrtHardwareDeviceType.NPU)
    gpu = _device("GpuProvider", ort.OrtHardwareDeviceType.GPU)
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    assert providers._select_devices([npu, gpu, cpu], {}, {}, 2) == [npu, gpu]


def test_explicit_npu_does_not_select_cpu_device_from_the_same_provider() -> None:
    npu = _device("SharedProvider", ort.OrtHardwareDeviceType.NPU)
    optimized_cpu = _device("SharedProvider", ort.OrtHardwareDeviceType.CPU)
    gpu = _device("GpuProvider", ort.OrtHardwareDeviceType.GPU)
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    assert providers._select_devices([npu, gpu, optimized_cpu, cpu], {}, {}, 8, device="npu") == [
        npu,
        cpu,
    ]


def _assigned_session(
    counts: dict[str, int], registered: tuple[str, ...] | None = None
) -> ort.InferenceSession:
    subgraphs = [
        SimpleNamespace(ep_name=name, get_nodes=lambda count=count: [None] * count)
        for name, count in counts.items()
    ]
    return cast(
        ort.InferenceSession,
        SimpleNamespace(
            get_provider_graph_assignment_info=lambda: subgraphs,
            get_providers=lambda: list(registered or counts),
        ),
    )


def test_provider_node_counts_accumulates_subgraphs() -> None:
    subgraphs = [
        SimpleNamespace(ep_name="CUDAExecutionProvider", get_nodes=lambda: [None] * 4),
        SimpleNamespace(ep_name="CPUExecutionProvider", get_nodes=lambda: [None]),
        SimpleNamespace(ep_name="CUDAExecutionProvider", get_nodes=lambda: [None] * 2),
    ]
    session = cast(
        ort.InferenceSession, SimpleNamespace(get_provider_graph_assignment_info=lambda: subgraphs)
    )
    assert providers.provider_node_counts(session) == {
        "CUDAExecutionProvider": 6,
        "CPUExecutionProvider": 1,
    }


@pytest.mark.parametrize("device", ["gpu", "npu"])
def test_explicit_accelerator_request_cannot_silently_select_cpu(
    device: providers.Device, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [cpu])
    with pytest.raises(RuntimeError, match=f"No enumerated ONNX Runtime {device.upper()}"):
        providers.create_session(b"not a model", device=device)


@pytest.mark.parametrize("device", ["gpu", "npu"])
def test_enumerated_accelerator_must_receive_graph_nodes(
    device: providers.Device, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = _device("AcceleratorProvider", providers._DEVICE_TYPES[device])
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [accelerator, cpu])
    session = _assigned_session({"CPUExecutionProvider": 3}, (accelerator.ep_name, cpu.ep_name))
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    with pytest.raises(RuntimeError, match="No graph nodes were assigned"):
        providers.create_session(b"model", device=device)


@pytest.mark.parametrize("device", ["gpu", "npu"])
def test_explicit_accelerator_allows_partial_cpu_assignment(
    device: providers.Device, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = _device("AcceleratorProvider", providers._DEVICE_TYPES[device])
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [accelerator, cpu])
    session = _assigned_session({"AcceleratorProvider": 9, "CPUExecutionProvider": 2})
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    assert providers.create_session(b"model", device=device) is session


def test_empty_assignment_report_does_not_claim_zero_gpu_nodes() -> None:
    session = _assigned_session({}, ("WebGpuExecutionProvider", "CPUExecutionProvider"))
    providers._require_accelerator(session, {"WebGpuExecutionProvider"}, "gpu")
    assert providers.provider_node_counts(session) == {}


def test_missing_accelerator_provider_is_rejected_even_without_assignment_report() -> None:
    session = _assigned_session({}, ("CPUExecutionProvider",))
    with pytest.raises(RuntimeError, match="provider was not initialized"):
        providers._require_accelerator(session, {"WebGpuExecutionProvider"}, "gpu")


def test_cuda_preloads_packaged_libraries_before_device_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    gpu = _device("CUDAExecutionProvider", ort.OrtHardwareDeviceType.GPU)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(
        ort, "get_build_info", lambda: "ORT Build Info: build type=Release, cuda-plugin-ep=1"
    )

    def preload(*, directory: str) -> None:
        assert directory == ""
        calls.append("preload")

    def enumerate_devices() -> list[ort.OrtEpDevice]:
        calls.append("enumerate")
        return [gpu]

    def create(model: bytes, **kwargs: object) -> ort.InferenceSession:
        calls.append("session")
        options = cast(ort.SessionOptions, kwargs["sess_options"])
        assert options.get_session_config_entry("session.record_ep_graph_assignment_info") == "1"
        assert kwargs["enable_fallback"] is False
        return _assigned_session({"CUDAExecutionProvider": 4})

    monkeypatch.setattr(ort, "preload_dlls", preload)
    monkeypatch.setattr(ort, "get_ep_devices", enumerate_devices)
    monkeypatch.setattr(ort, "InferenceSession", create)
    providers.create_session(b"model", device="gpu")
    assert calls == ["preload", "enumerate", "session"]


@pytest.mark.parametrize("device,cuda_present", [("cpu", True), ("npu", True), ("auto", False)])
def test_non_cuda_selection_and_cpu_wheel_do_not_load_cuda_libraries(
    device: providers.Device, cuda_present: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    npu = _device("NpuProvider", ort.OrtHardwareDeviceType.NPU)
    available = ["CPUExecutionProvider", "CUDAExecutionProvider"] if cuda_present else [cpu.ep_name]
    monkeypatch.setattr(ort, "get_available_providers", lambda: available)
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [npu, cpu])
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *args, **kwargs: _assigned_session({"NpuProvider": 1})
    )

    def preload(**kwargs: object) -> None:
        pytest.fail("CPU selection or a CPU wheel must not load CUDA libraries")

    monkeypatch.setattr(ort, "preload_dlls", preload)
    providers.create_session(b"model", device=device)


def test_cuda_wheel_without_build_flag_registers_bundled_plugin_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "onnxruntime"
    capi = package / "capi"
    capi.mkdir(parents=True)
    library = capi / "libonnxruntime_providers_cuda.so"
    library.touch()
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    gpu = _device("CUDAExecutionProvider", ort.OrtHardwareDeviceType.GPU)
    calls: list[str] = []
    monkeypatch.setattr(ort, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(ort, "get_available_providers", lambda: [gpu.ep_name, cpu.ep_name])
    monkeypatch.setattr(ort, "get_build_info", lambda: "ORT Build Info: build type=Release")
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [gpu, cpu] if "register" in calls else [cpu])
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: calls.append("preload"))

    def register(name: str, path: str) -> None:
        assert name == gpu.ep_name
        assert path == str(library)
        calls.append("register")

    monkeypatch.setattr(ort, "register_execution_provider_library", register)
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *args, **kwargs: _assigned_session({gpu.ep_name: 3})
    )
    providers.create_session(b"model", device="gpu")
    providers.create_session(b"model", device="gpu")
    assert calls == ["preload", "register"]


@pytest.mark.parametrize("device", ["auto", "gpu"])
@pytest.mark.parametrize("failure", [RuntimeError, ort.capi.onnxruntime_pybind11_state.Fail])
def test_only_auto_can_fall_back_after_cuda_setup_failure(
    device: providers.Device, failure: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [cpu])

    def initialize() -> None:
        calls.append("initialize")
        raise failure("CUDA driver library is missing")

    def create(*args: object, **kwargs: object) -> ort.InferenceSession:
        calls.append("session")
        return _assigned_session({cpu.ep_name: 2})

    monkeypatch.setattr(providers, "initialize_cuda", initialize)
    monkeypatch.setattr(ort, "InferenceSession", create)
    if device == "auto":
        with pytest.warns(RuntimeWarning, match="accelerator initialization failed"):
            session = providers.create_session(b"model", device=device)
        assert providers.provider_node_counts(session) == {cpu.ep_name: 2}
        assert calls == ["initialize", "session"]
    else:
        with pytest.raises(failure, match="CUDA driver library is missing"):
            providers.create_session(b"model", device=device)
        assert calls == ["initialize"]


@pytest.mark.parametrize("package", providers._PLUGIN_PACKAGES)
def test_installed_plugin_is_registered_before_device_selection(
    package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = _device("CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU)
    gpu = _device("PluginExecutionProvider", ort.OrtHardwareDeviceType.GPU)
    registered: list[tuple[str, str]] = []
    monkeypatch.setattr(providers, "find_spec", lambda name: name == package or None)
    monkeypatch.setattr(ort, "get_available_providers", lambda: [cpu.ep_name])
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [gpu, cpu] if registered else [cpu])
    monkeypatch.setattr(
        providers,
        "import_module",
        lambda name: SimpleNamespace(
            get_ep_name=lambda: gpu.ep_name, get_library_path=lambda: "/runtime/plugin.so"
        ),
    )
    monkeypatch.setattr(
        ort,
        "register_execution_provider_library",
        lambda name, path: registered.append((name, path)),
    )
    options = SimpleNamespace(
        add_session_config_entry=lambda *args: None,
        set_provider_selection_policy_delegate=lambda _delegate: None,
    )
    for _ in range(2):
        assert providers._configure_devices(options, "gpu") == {gpu.ep_name}
    assert registered == [(gpu.ep_name, "/runtime/plugin.so")]


def test_cpu_policy_does_not_import_installed_accelerator_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_discovery() -> list:
        pytest.fail("CPU policy must not discover accelerator dependencies")

    monkeypatch.setattr(providers, "_accelerator_initializers", unexpected_discovery)
    providers._initialize_accelerators("cpu")


def test_native_cpu_session_records_graph_assignment_and_runs() -> None:
    import numpy as np
    import onnx

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["input", "input"], ["output"])],
        "native-assignment",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10
    )
    session = providers.create_session(model.SerializeToString(), device="cpu")
    assert providers.provider_node_counts(session) == {"CPUExecutionProvider": 1}
    value = np.array([1.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(session.run(None, {"input": value})[0], value * 2)


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown ARG device"):
        providers.create_session(b"not a model", device="cuda")  # type: ignore[arg-type]


def test_existing_provider_configuration_is_rejected() -> None:
    options = ort.SessionOptions()
    options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_CPU)
    with pytest.raises(ValueError, match="already configure"):
        providers.create_session(b"not a model", options=options)


def test_model_initialization_error_is_not_retried_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fail(model: bytes | str, **kwargs: object) -> ort.InferenceSession:
        calls.append(kwargs)
        raise ValueError("invalid model")

    monkeypatch.setattr(ort, "InferenceSession", fail)
    with pytest.raises(ValueError, match="invalid model"):
        providers.create_session(b"not a model")
    assert len(calls) == 1
    assert calls[0]["enable_fallback"] is False
    assert cast(ort.SessionOptions, calls[0]["sess_options"]).has_providers()
