"""Initialize ONNX Runtime's CUDA provider without a training framework."""

from functools import cache
from pathlib import Path


@cache
def initialize_cuda() -> None:
    import onnxruntime as ort

    ort.preload_dlls(directory="")
    # Some 1.30 GPU wheels export the plugin ABI without the build-info flag
    # used by ORT's automatic registration hook.
    if ", cuda-plugin-ep=1" not in ort.get_build_info():
        if any(ep.ep_name == "CUDAExecutionProvider" for ep in ort.get_ep_devices()):
            return
        library = next(
            (Path(ort.__file__).parent / "capi").glob("*onnxruntime_providers_cuda.*"), None
        )
        if library is None:
            raise RuntimeError("The bundled ONNX Runtime CUDA provider library is missing")
        ort.register_execution_provider_library("CUDAExecutionProvider", str(library))
