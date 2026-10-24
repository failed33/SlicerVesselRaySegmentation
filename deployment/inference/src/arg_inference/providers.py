"""Choose ARG execution devices through ONNX Runtime's native device APIs.

Only enumerated devices participate. A compiled provider name alone does not
establish that usable hardware exists, or that it supports ARG's 3D graph.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import cache, partial
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import onnxruntime as ort
from cuda_ort import initialize_cuda

Device = Literal["auto", "cpu", "gpu", "npu"]

_DEVICE_TYPES = {
    "npu": ort.OrtHardwareDeviceType.NPU,
    "gpu": ort.OrtHardwareDeviceType.GPU,
    "cpu": ort.OrtHardwareDeviceType.CPU,
}
_PLUGIN_PACKAGES = ("onnxruntime_ep_migraphx", "onnxruntime_ep_webgpu")


def _select_devices(
    ep_devices: Sequence[ort.OrtEpDevice],
    _model_metadata: Mapping[str, str],
    _runtime_metadata: Mapping[str, str],
    max_selections: int,
    *,
    device: Device = "auto",
) -> list[ort.OrtEpDevice]:
    """Select devices while retaining ORT's order within each requested type."""
    # ORT orders the delegate's input by device type, vendor and discrete GPU
    # preference. Its built-in PREFER_NPU policy omits GPU fallback.
    selected: list[ort.OrtEpDevice] = []
    device_types = _DEVICE_TYPES.values() if device == "auto" else [_DEVICE_TYPES[device]]
    for device_type in device_types:
        match = next((ep for ep in ep_devices if ep.device.type == device_type), None)
        if match is not None:
            selected.append(match)
    # Keep the default CPU EP's operator coverage when an optimized CPU EP won.
    default_cpu = next((ep for ep in ep_devices if ep.ep_name == "CPUExecutionProvider"), None)
    if default_cpu is not None and default_cpu not in selected:
        selected.append(default_cpu)
    return selected[:max_selections]


def provider_node_counts(session: ort.InferenceSession) -> dict[str, int]:
    """Count optimized graph nodes assigned to each EP at session creation."""
    counts: dict[str, int] = {}
    for subgraph in session.get_provider_graph_assignment_info():
        counts[subgraph.ep_name] = counts.get(subgraph.ep_name, 0) + len(subgraph.get_nodes())
    return counts


@cache
def _initialize_plugin(package: str) -> None:
    """Register the native library exposed by an installed ORT plugin package."""
    plugin = import_module(package)
    name = plugin.get_ep_name()
    if not any(ep.ep_name == name for ep in ort.get_ep_devices()):
        ort.register_execution_provider_library(name, plugin.get_library_path())


def _accelerator_initializers() -> list[Callable[[], None]]:
    initializers: list[Callable[[], None]] = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        initializers.append(initialize_cuda)
    initializers.extend(
        partial(_initialize_plugin, package)
        for package in _PLUGIN_PACKAGES
        if find_spec(package) is not None
    )
    return initializers


def _initialize_accelerators(device: Device) -> None:
    if device not in {"auto", "gpu"}:
        return
    for initialize in _accelerator_initializers():
        try:
            initialize()
        except Exception as error:
            if device != "auto":
                raise
            # ORT native exceptions and dynamic-library loader errors do not
            # share a Python base class. This boundary covers provider setup
            # only, before model loading; model/run errors are never retried.
            warnings.warn(
                f"ONNX Runtime accelerator initialization failed: {error}", RuntimeWarning
            )


def _configure_devices(options: ort.SessionOptions, device: Device) -> set[str]:
    _initialize_accelerators(device)
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if device == "auto":
        options.set_provider_selection_policy_delegate(_select_devices)
        return set()
    required_providers = {
        ep.ep_name for ep in ort.get_ep_devices() if ep.device.type == _DEVICE_TYPES[device]
    }
    if not required_providers:
        raise RuntimeError(f"No enumerated ONNX Runtime {device.upper()} device is available")
    options.set_provider_selection_policy_delegate(partial(_select_devices, device=device))
    return required_providers


def _require_accelerator(
    session: ort.InferenceSession, required_providers: set[str], device: Device
) -> None:
    if not required_providers.intersection(session.get_providers()):
        raise RuntimeError(
            f"The requested ONNX Runtime {device.upper()} provider was not initialized"
        )
    counts = provider_node_counts(session)
    # ORT 1.30 returns no assignment records for the tested WebGPU plugin even
    # when CPU fallback is disabled and profiling proves GPU kernel execution.
    # An empty report is unavailable evidence, not zero assigned GPU nodes.
    if counts and not any(counts.get(name, 0) for name in required_providers):
        raise RuntimeError(
            f"No graph nodes were assigned to the requested ONNX Runtime {device.upper()} device"
        )


def create_session(
    model_source: bytes | str | Path,
    *,
    device: Device = "auto",
    options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Create a session with NPU > GPU > CPU priority, or a required device type.

    CPU operators remain available for graph partitioning. Explicit GPU/NPU
    requests require a matching device and initialized provider. When assignment
    records are available, they must include accelerator nodes. Initialization
    and run errors propagate without reconstructing the session on CPU.
    """
    if device not in {"auto", *_DEVICE_TYPES}:
        raise ValueError(f"Unknown ARG device {device!r}; use auto, cpu, gpu or npu")
    if options is None:
        options = ort.SessionOptions()
    if options.has_providers():
        raise ValueError("ARG SessionOptions must not already configure execution providers")
    required_providers = _configure_devices(options, device)
    # ORT Python otherwise retries ValueError/RuntimeError during construction,
    # including unrelated model errors, with a different provider list.
    session = ort.InferenceSession(model_source, sess_options=options, enable_fallback=False)
    if device in {"gpu", "npu"}:
        _require_accelerator(session, required_providers, device)
    return session
