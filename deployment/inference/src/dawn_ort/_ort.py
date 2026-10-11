"""External-device ONNX Runtime setup and asynchronous queue submission."""

import ctypes as ct

import onnxruntime as ort

from .host import DawnHost, address


def create_session(
    host: DawnHost,
    model_source: bytes | str,
    *,
    options: ort.SessionOptions | None = None,
    **provider_options: str,
) -> ort.InferenceSession:
    if options is None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    host.library.dawn_ort_get_native_procs.restype = ct.c_void_p
    table = host.library.dawn_ort_get_native_procs()
    if not table:
        raise RuntimeError("Dawn returned a null procedure table")
    devices = [dev for dev in ort.get_ep_devices() if dev.ep_name == "WebGpuExecutionProvider"]
    options.add_provider_for_devices(
        devices[:1],
        {
            "deviceId": str(host.context_id),
            "webgpuInstance": str(address(host.instance)),
            "webgpuDevice": str(address(host.device)),
            "dawnProcTable": str(table),
            "preserveDevice": "0",
            "enableGraphCapture": "0",
            "validationMode": "wgpuOnly",
            **provider_options,
        },
    )
    return ort.InferenceSession(model_source, sess_options=options)


def run(session: ort.InferenceSession, binding: ort.IOBinding) -> None:
    options = ort.RunOptions()
    options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    session.run_with_iobinding(binding, options)
