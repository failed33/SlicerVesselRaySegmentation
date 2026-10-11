"""Named ONNX input/output bindings backed by caller-owned Dawn buffers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType

import onnxruntime as ort

from ._ort import create_session, run
from .host import DawnHost
from .tensors import DeviceTensor


class GpuSession:
    """One ONNX session and reusable binding, with completion before release.

    Submissions and consumers use the host's ordered queue.
    """

    def __init__(
        self,
        host: DawnHost,
        model: bytes | str | Path,
        *,
        options: ort.SessionOptions | None = None,
        provider_options: Mapping[str, str] | None = None,
    ) -> None:
        settings = dict(provider_options or {})
        managed = {
            "deviceId",
            "webgpuInstance",
            "webgpuDevice",
            "dawnProcTable",
            "preserveDevice",
            "enableGraphCapture",
            "validationMode",
        }
        if managed.intersection(settings):
            raise ValueError("Device, lifetime and queue policy are managed by the runtime")
        self.host = host
        if options is None:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # Honor explicit caller policy; GPU-only remains the default.
        try:
            options.get_session_config_entry("session.disable_cpu_ep_fallback")
        except RuntimeError:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self._session: ort.InferenceSession | None = create_session(
            host,
            str(model) if isinstance(model, Path) else model,
            options=options,
            **settings,
        )
        self._binding = None
        self._outputs: Mapping[str, DeviceTensor] = MappingProxyType({})
        self._resources = ExitStack()

    @property
    def native_session(self) -> ort.InferenceSession:
        if self._session is None:
            raise RuntimeError("GPU session has been closed")
        return self._session

    def bind(
        self, inputs: Mapping[str, DeviceTensor], *, outputs: Mapping[str, DeviceTensor]
    ) -> GpuSession:
        session = self.native_session
        if self._binding is not None:
            raise RuntimeError("The session is already bound")
        self._validate(inputs, outputs)
        binding, retained_outputs = session.io_binding(), {}
        with ExitStack() as resources:
            try:
                for is_output, tensors in ((False, inputs), (True, outputs)):
                    method = binding.bind_output if is_output else binding.bind_input
                    for name, value in tensors.items():
                        owner = value.owner.retain()
                        resources.callback(owner.close)
                        method(name, "webgpu", 0, value.dtype, value.shape, owner.address)
                        if is_output:
                            retained_outputs[name] = DeviceTensor(owner, value.shape, value.dtype)
            finally:
                resources.callback(binding.clear_binding_inputs)
                resources.callback(binding.clear_binding_outputs)
            self._resources = resources.pop_all()
        self._binding, self._outputs = binding, MappingProxyType(retained_outputs)
        return self

    def _validate(
        self, inputs: Mapping[str, DeviceTensor], outputs: Mapping[str, DeviceTensor]
    ) -> None:
        if not outputs:
            raise ValueError("Bind at least one output")
        for value in (*inputs.values(), *outputs.values()):
            value.validate(self.host)
        input_handles = {value.owner.address for value in inputs.values()}
        output_handles = [value.owner.address for value in outputs.values()]
        if input_handles.intersection(output_handles) or len(set(output_handles)) != len(
            output_handles
        ):
            raise ValueError("Output buffers must not alias inputs or other outputs")

    def submit(self) -> Mapping[str, DeviceTensor]:
        if self._binding is None:
            raise RuntimeError("GPU binding has been closed or has not been created")
        for output in self._outputs.values():
            output.validate(self.host)
        run(self.native_session, self._binding)
        return self._outputs

    def close(self) -> None:
        if self._session is not None:
            self.host._wait_for_completion()
            self._resources.close()
            self._binding = self._session = None
            self._outputs = MappingProxyType({})
            self.host.raise_errors()
