"""Dawn host feature negotiation and teardown after validation failures."""

import ctypes as ct
import gc
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from dawn_ort.host import Buffer, DawnHost


class _Limits(ct.Structure):
    _fields_ = [("maxBufferSize", ct.c_uint64)]


class _InstanceDescriptor(ct.Structure):
    _fields_ = [
        ("requiredFeatureCount", ct.c_size_t),
        ("requiredFeatures", ct.POINTER(ct.c_uint)),
    ]


class _AdapterOptions(ct.Structure):
    _fields_ = [("backendType", ct.c_uint)]


_ErrorCallback = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p)


class _ErrorCallbackInfo(ct.Structure):
    _fields_ = [("callback", _ErrorCallback)]


class _DeviceDescriptor(ct.Structure):
    _fields_ = [
        ("requiredLimits", ct.POINTER(_Limits)),
        ("requiredFeatureCount", ct.c_size_t),
        ("requiredFeatures", ct.POINTER(ct.c_uint)),
        ("uncapturedErrorCallbackInfo", _ErrorCallbackInfo),
    ]


@pytest.mark.parametrize("char_type", [ct.c_char, ct.c_ubyte])
@pytest.mark.parametrize("text", ["", "WGSL shader", "vessel \u03b1"])
def test_string_view_uses_generated_char_pointer_and_retains_utf8(
    char_type: Any, text: str
) -> None:
    class StringView(ct.Structure):
        _fields_ = [("data", ct.POINTER(char_type)), ("length", ct.c_size_t)]

    host = object.__new__(DawnHost)
    host.w = ModuleType("generated_webgpu")
    setattr(host.w, "WGPUStringView", StringView)
    view = host._view(text)
    gc.collect()
    encoded = text.encode()
    assert view.length == len(encoded)
    assert ct.string_at(view.data, view.length) == encoded
    assert DawnHost._message(view) == text


@pytest.mark.parametrize("supported", [True, False])
def test_device_requests_shader_f16_only_when_adapter_supports_it(
    monkeypatch: pytest.MonkeyPatch, supported: bool
) -> None:
    bindings = SimpleNamespace(
        WGPUInstanceFeatureName=ct.c_uint,
        WGPUInstanceFeatureName_TimedWaitAny=1,
        WGPUInstanceDescriptor=_InstanceDescriptor,
        WGPUBackendType_Metal=5,
        WGPURequestAdapterOptions=_AdapterOptions,
        WGPULimits=_Limits,
        WGPUStatus_Success=1,
        WGPUFeatureName=ct.c_uint,
        WGPUFeatureName_ShaderF16=42,
        WGPUDeviceDescriptor=_DeviceDescriptor,
        WGPUUncapturedErrorCallback=_ErrorCallback,
        wgpuCreateInstance=Mock(return_value="instance"),
        wgpuAdapterHasFeature=Mock(return_value=supported),
        wgpuInstanceRequestAdapter=Mock(),
        wgpuAdapterRequestDevice=Mock(),
        wgpuDeviceGetQueue=Mock(return_value="queue"),
        **{f"wgpu{kind}Release": Mock() for kind in ("Instance", "Adapter", "Device", "Queue")},
    )
    device_requests = []
    adapter_requests = []

    def get_limits(_adapter: Any, pointer: Any) -> int:
        ct.cast(pointer, ct.POINTER(_Limits)).contents.maxBufferSize = 4_294_967_295
        return bindings.WGPUStatus_Success

    bindings.wgpuAdapterGetLimits = Mock(side_effect=get_limits)

    def request(_host: DawnHost, kind: str, _function: Any, *args: Any) -> str:
        if kind == "RequestAdapter":
            adapter_requests.append(
                ct.cast(args[1], ct.POINTER(_AdapterOptions)).contents.backendType
            )
            return "adapter"
        assert kind == "RequestDevice"
        descriptor = ct.cast(args[1], ct.POINTER(_DeviceDescriptor)).contents
        device_requests.append(
            {
                "features": list(descriptor.requiredFeatures[: descriptor.requiredFeatureCount]),
                "features_pointer_present": bool(descriptor.requiredFeatures),
                "max_buffer_size": descriptor.requiredLimits.contents.maxBufferSize,
                "error_callback_present": bool(descriptor.uncapturedErrorCallbackInfo.callback),
            }
        )
        return "device"

    monkeypatch.setattr("dawn_ort.host.load_bindings", lambda _path: (bindings, object()))
    monkeypatch.setattr(
        "dawn_ort.host.platform_config", lambda _system: SimpleNamespace(backend="Metal")
    )
    monkeypatch.setattr(DawnHost, "_request", request)
    host = DawnHost(Path("unused-generated-bindings.py"))

    bindings.wgpuAdapterHasFeature.assert_called_once_with("adapter", 42)
    assert adapter_requests == [5]
    assert device_requests == [
        {
            "features": [42] if supported else [],
            "features_pointer_present": supported,
            "max_buffer_size": 4_294_967_295,
            "error_callback_present": True,
        }
    ]
    assert host.device == "device" and host.queue == "queue"
    host._resources.close()
    for kind in ("Queue", "Device", "Adapter", "Instance"):
        getattr(bindings, f"wgpu{kind}Release").assert_called_once_with(kind.lower())


@pytest.fixture
def setup():
    host = object.__new__(DawnHost)
    events = []
    host.errors = []
    host.buffers = set()
    host.callbacks = [object()]
    host.instance, host.adapter, host.device, host.queue = (object() for _ in range(4))
    host.w = ModuleType("generated_webgpu")
    setattr(host.w, "wgpuQueueOnSubmittedWorkDone", Mock())
    setattr(host.w, "wgpuBufferRelease", Mock(side_effect=lambda _handle: events.append("buffer")))
    host._request = Mock(side_effect=lambda *_args: events.append("complete"))
    host._resources = ExitStack()
    for kind in ("instance", "adapter", "device", "queue"):
        host._resources.callback(events.append, kind)
    buffer = Buffer(host, object(), 16, 16)
    return SimpleNamespace(host=host, buffer=buffer, events=events)


@pytest.mark.parametrize("error_during_wait", [False, True])
def test_completed_close_releases_resources_before_reporting_validation(setup, error_during_wait):
    host = setup.host
    error = "invalid shader or bind group"
    if error_during_wait:

        def complete(*_args):
            setup.events.append("complete")
            host.errors.append(error)

        host._request.side_effect = complete
    else:
        host.errors.append(error)
    queue = host.queue
    with pytest.raises(RuntimeError, match="Dawn validation: " + error):
        host.close()
    host._request.assert_called_once_with(
        "QueueWorkDone", host.w.wgpuQueueOnSubmittedWorkDone, queue
    )
    assert setup.events == ["complete", "buffer", "queue", "device", "adapter", "instance"]
    assert setup.buffer.closed
    assert not host.buffers and not host.callbacks
    assert host.instance is host.adapter is host.device is host.queue is None
    assert host.errors == [error]
    host.close()
    assert host._request.call_count == 1


def test_close_is_idempotent_without_validation_errors(setup):
    setup.host.close()
    setup.host.close()
    setup.host._request.assert_called_once()
    setup.host.w.wgpuBufferRelease.assert_called_once()
    assert setup.events == ["complete", "buffer", "queue", "device", "adapter", "instance"]


@pytest.mark.parametrize("validation_error", [False, True])
def test_failed_queue_completion_preserves_resources_for_retry(setup, validation_error):
    host = setup.host
    if validation_error:
        host.errors.append("invalid shader")
    handles = host.instance, host.adapter, host.device, host.queue
    callbacks = list(host.callbacks)
    host._request.side_effect = RuntimeError("queue completion failed")
    with pytest.raises(RuntimeError, match="queue completion failed"):
        host.close()
    assert not setup.events
    assert not setup.buffer.closed and host.buffers == {setup.buffer}
    assert (host.instance, host.adapter, host.device, host.queue) == handles
    assert host.callbacks == callbacks
    host.w.wgpuBufferRelease.assert_not_called()

    host._request.side_effect = lambda *_args: setup.events.append("complete")
    if validation_error:
        with pytest.raises(RuntimeError, match="Dawn validation: invalid shader"):
            host.close()
    else:
        host.close()
    assert setup.buffer.closed
    assert setup.events == ["complete", "buffer", "queue", "device", "adapter", "instance"]


def test_sync_keeps_reporting_stored_validation_errors(setup):
    host = setup.host
    host.errors.append("invalid bind group")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Dawn validation: invalid bind group"):
            host.sync()
    assert host._request.call_count == 2
    assert host.errors == ["invalid bind group"]
    assert not setup.buffer.closed
    assert host.queue is not None
