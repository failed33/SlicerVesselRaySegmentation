"""Ownership and failure handling for the Dawn session seam, using fake handles.

Real queue ordering and buffer retention are checked by examples/buffer_pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from dawn_ort import DeviceTensor
from dawn_ort import session as backend
from dawn_ort.host import DawnHost

SPECS = {
    "left": ((2, 5), np.float32),
    "right": ((2, 5), np.float32),
    "sum": ((2, 5), np.float32),
    "difference": ((2, 5), np.float32),
}


@dataclass
class Owner:
    device: object
    handle: int
    allocated_size: int
    references: list[Owner]
    closed: bool = False

    @property
    def address(self) -> int:
        if self.closed:
            raise RuntimeError("Owner released")
        return self.handle

    def retain(self) -> Owner:
        self.address
        value = Owner(self.device, self.handle, self.allocated_size, self.references)
        self.references.append(value)
        return value

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def setup(monkeypatch):
    host = object.__new__(DawnHost)
    host._wait_for_completion = Mock()
    host.errors = []
    native = Mock()
    monkeypatch.setattr(backend, "create_session", lambda *_args, **_kwargs: native)
    submit = Mock()
    monkeypatch.setattr(backend, "run", submit)
    session = backend.GpuSession(host, Path("arithmetic.onnx"))
    host._wait_for_completion.assert_called_once()
    host._wait_for_completion.reset_mock()
    refs = []
    tensors = {}
    for number, (name, (shape, dtype)) in enumerate(SPECS.items(), start=1):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        tensors[name] = DeviceTensor(
            Owner(host, number, (size + 15) // 16 * 16, refs), shape, dtype
        )
    outputs = {name: tensors.pop(name) for name in ("sum", "difference")}
    return SimpleNamespace(
        host=host,
        native=native,
        session=session,
        refs=refs,
        inputs=tensors,
        outputs=outputs,
        submit=submit,
    )


def test_binding_retains_native_owners_and_queues_repeated_inference(setup):
    binding = setup.session.bind(setup.inputs, outputs=setup.outputs)
    for tensor in [*setup.inputs.values(), *setup.outputs.values()]:
        tensor.owner.close()
    result = binding.submit()
    assert result["sum"].owner.address == 3
    assert result["difference"].owner.address == 4
    with pytest.raises(TypeError):
        result["sum"] = result["difference"]
    binding.submit()
    assert setup.submit.call_count == 2
    setup.host._wait_for_completion.assert_not_called()
    assert len(setup.refs) == 4 and all(not owner.closed for owner in setup.refs)
    binding.close()
    setup.host._wait_for_completion.assert_called_once()
    assert all(owner.closed for owner in setup.refs)
    with pytest.raises(RuntimeError, match="binding has been closed"):
        binding.submit()
    binding.close()
    setup.host._wait_for_completion.assert_called_once()
    setup.session.close()
    with pytest.raises(RuntimeError, match="session has been closed"):
        setup.session.bind(setup.inputs, outputs=setup.outputs)


@pytest.mark.parametrize("invalid", ["device", "alias", "closed"])
def test_invalid_native_inputs_are_rejected_before_retaining_or_submitting(setup, invalid):
    tensors = setup.inputs
    value = tensors["left"]
    if invalid == "device":
        value.owner.device = object()
    elif invalid == "alias":
        setup.outputs["sum"].owner.handle = value.owner.handle
    else:
        value.owner.close()
    expected_error: type[Exception] = RuntimeError if invalid == "closed" else ValueError
    with pytest.raises(expected_error):
        setup.session.bind(tensors, outputs=setup.outputs)
    assert not setup.refs
    setup.submit.assert_not_called()
    setup.session.close()


def test_failed_bind_releases_only_the_new_references(setup):
    setup.native.io_binding.return_value.bind_input.side_effect = RuntimeError("bind failed")
    with pytest.raises(RuntimeError, match="bind failed"):
        setup.session.bind(setup.inputs, outputs=setup.outputs)
    assert all(owner.closed for owner in setup.refs)
    assert all(not tensor.owner.closed for tensor in setup.inputs.values())
    setup.session.close()


def test_failed_completion_preserves_references_for_retry(setup):
    binding = setup.session.bind(setup.inputs, outputs=setup.outputs)
    binding.submit()
    setup.host._wait_for_completion.side_effect = RuntimeError("queue completion failed")
    with pytest.raises(RuntimeError, match="queue completion failed"):
        setup.session.close()
    assert all(not owner.closed for owner in setup.refs)
    setup.host._wait_for_completion.side_effect = None
    setup.session.close()
    assert all(owner.closed for owner in setup.refs)
    with pytest.raises(RuntimeError, match="binding has been closed"):
        binding.submit()


def test_rebinding_preserves_the_existing_pipeline(setup):
    assert setup.session.bind(setup.inputs, outputs=setup.outputs) is setup.session
    with pytest.raises(RuntimeError, match="already bound"):
        setup.session.bind(setup.inputs, outputs=setup.outputs)
    assert len(setup.refs) == 4 and all(not owner.closed for owner in setup.refs)
    setup.native.io_binding.assert_called_once()
    setup.session.submit()
    setup.submit.assert_called_once()
    setup.session.close()


def test_binding_clears_before_retained_references_are_released(setup, monkeypatch):
    events = []
    setup.host._wait_for_completion.side_effect = lambda: events.append("complete")
    binding = setup.native.io_binding.return_value
    binding.clear_binding_inputs.side_effect = lambda: events.append("inputs")
    binding.clear_binding_outputs.side_effect = lambda: events.append("outputs")
    original_close = Owner.close

    def close(owner):
        events.append("release")
        original_close(owner)

    monkeypatch.setattr(Owner, "close", close)
    setup.session.bind(setup.inputs, outputs=setup.outputs)
    setup.session.close()
    assert events[0] == "complete"
    assert set(events[1:3]) == {"inputs", "outputs"}
    assert events[3:] == ["release"] * 4


@pytest.mark.parametrize("error_during_wait", [False, True])
def test_completed_close_releases_binding_before_reporting_validation(setup, error_during_wait):
    binding = setup.session.bind(setup.inputs, outputs=setup.outputs)
    binding.submit()
    error = "invalid shader"
    if error_during_wait:
        setup.host._wait_for_completion.side_effect = lambda: setup.host.errors.append(error)
    else:
        setup.host.errors.append(error)
    with pytest.raises(RuntimeError, match="Dawn validation: " + error):
        setup.session.close()
    setup.host._wait_for_completion.assert_called_once()
    native_binding = setup.native.io_binding.return_value
    native_binding.clear_binding_inputs.assert_called_once()
    native_binding.clear_binding_outputs.assert_called_once()
    assert all(owner.closed for owner in setup.refs)
    assert all(not tensor.owner.closed for tensor in setup.inputs.values())
    with pytest.raises(RuntimeError, match="session has been closed"):
        setup.session.native_session
    with pytest.raises(RuntimeError, match="binding has been closed"):
        binding.submit()
    assert setup.host.errors == [error]
    setup.session.close()
    setup.host._wait_for_completion.assert_called_once()


def test_failed_completion_with_validation_error_preserves_binding_for_retry(setup):
    setup.session.bind(setup.inputs, outputs=setup.outputs).submit()
    setup.host.errors.append("invalid shader")
    setup.host._wait_for_completion.side_effect = RuntimeError("queue completion failed")
    with pytest.raises(RuntimeError, match="queue completion failed"):
        setup.session.close()
    native_binding = setup.native.io_binding.return_value
    native_binding.clear_binding_inputs.assert_not_called()
    native_binding.clear_binding_outputs.assert_not_called()
    assert setup.session.native_session is setup.native
    assert all(not owner.closed for owner in setup.refs)
    setup.host._wait_for_completion.side_effect = None
    with pytest.raises(RuntimeError, match="Dawn validation: invalid shader"):
        setup.session.close()
    assert all(owner.closed for owner in setup.refs)
    native_binding.clear_binding_inputs.assert_called_once()
    native_binding.clear_binding_outputs.assert_called_once()


def test_model_and_provider_choices_are_forwarded_without_preparation(monkeypatch):
    create = Mock()
    monkeypatch.setattr(backend, "create_session", create)
    source = b"caller-supplied-model"
    backend.GpuSession(Mock(spec=DawnHost), source, provider_options={"preferredLayout": "NHWC"})
    assert create.call_args.args[1] is source
    assert create.call_args.kwargs["preferredLayout"] == "NHWC"
    assert "enableInt64" not in create.call_args.kwargs


@pytest.mark.parametrize("policy", [None, "0", "1"])
def test_explicit_cpu_fallback_policy_is_preserved(monkeypatch, policy):
    create = Mock()
    monkeypatch.setattr(backend, "create_session", create)
    options = backend.ort.SessionOptions()
    if policy is not None:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", policy)
    backend.GpuSession(Mock(spec=DawnHost), b"model", options=options)
    forwarded = create.call_args.kwargs["options"]
    assert forwarded.get_session_config_entry("session.disable_cpu_ep_fallback") == (
        "1" if policy is None else policy
    )


def test_initialization_completion_precedes_exposing_the_native_session(monkeypatch):
    events = []
    native = Mock()
    host = Mock(spec=DawnHost)
    host.sync.side_effect = lambda: events.append("complete")

    def create(*_args, **_kwargs):
        events.append("initialize")
        return native

    monkeypatch.setattr(backend, "create_session", create)
    session = backend.GpuSession(host, b"model")
    assert session.native_session is native
    assert events == ["initialize", "complete"]


@pytest.mark.parametrize("key", ["deviceId", "preserveDevice", "enableGraphCapture"])
def test_device_and_queue_policy_cannot_be_overridden(key):
    with pytest.raises(ValueError, match="managed by the runtime"):
        backend.GpuSession(object.__new__(DawnHost), b"model", provider_options={key: "1"})


def test_outputs_need_independent_storage_and_a_binding(setup):
    with pytest.raises(ValueError, match="at least one output"):
        setup.session.bind(setup.inputs, outputs={})
    setup.outputs["difference"].owner.handle = setup.outputs["sum"].owner.handle
    with pytest.raises(ValueError, match="must not alias"):
        setup.session.bind(setup.inputs, outputs=setup.outputs)
    assert not setup.refs
