"""ONNX Runtime CUDA execution with Slang inputs and one shared CUDA stream."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from cuda_ort import initialize_cuda
from cuda_ort.host import CudaHost

from .constants import OUTPUT_NAME, PATCH_SIZE
from .cuda_inputs import CudaPatchInputs
from .graph import prepare_model_o
from .inputs import PackedRayCase
from .stitching import mask_from_logits


@dataclass(frozen=True)
class _PendingPatch:
    logits: ort.OrtValue
    # Retain pageable upload owners until the shared stream completes.
    case: PackedRayCase
    config: np.ndarray


class CudaPatchInference:
    """Own one capacity group's session and sampler; retain uploads until collection."""

    def __init__(
        self, host: CudaHost, model_path: Path, case: PackedRayCase, volume: ort.OrtValue
    ) -> None:
        self.host = host
        initialize_cuda()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry("session.disable_prepacking", "1")
        options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
        self.native_session = ort.InferenceSession(
            prepare_model_o(str(model_path), case.n_rays),
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": "0",
                        "use_tf32": "0",
                        "user_compute_stream": str(int(host.stream)),
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                )
            ],
        )
        self.native_session.disable_fallback()
        if "CUDAExecutionProvider" not in self.native_session.get_providers():
            raise RuntimeError("ModelO CUDA session did not initialize CUDAExecutionProvider")
        self.logits = host.empty((1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), np.dtype(np.float32))
        self._inputs = CudaPatchInputs(host, case, volume)
        try:
            self.binding = self.native_session.io_binding()
            for name, tensor in self._inputs.gpu_inputs.items():
                self.binding.bind_ortvalue_input(name, tensor)
            self.binding.bind_ortvalue_output(OUTPUT_NAME, self.logits)
            self.run_options = ort.RunOptions()
            self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
        except BaseException:
            self._inputs.close()
            raise

    def submit(self, case: PackedRayCase, resources: ExitStack) -> _PendingPatch:
        config = self._inputs.generate(case)
        self.native_session.run_with_iobinding(self.binding, self.run_options)
        snapshot = self.host.empty(tuple(self.logits.shape()), np.dtype(np.float32))
        self.host.copy(snapshot, self.logits, self.logits.tensor_size_in_bytes())
        return _PendingPatch(snapshot, case, config)

    def collect(self, readback: _PendingPatch) -> np.ndarray:
        self.host.synchronize()
        return mask_from_logits(readback.logits.numpy().reshape((PATCH_SIZE,) * 3))

    def close(self) -> None:
        self._inputs.close()
        # Release ORT's arena before the next capacity group is constructed.
        del self.binding, self.native_session, self.logits, self._inputs
