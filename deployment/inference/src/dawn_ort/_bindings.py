"""Connect generated WebGPU declarations to the matching native Dawn host."""

import ctypes as ct
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .platforms import platform_config


def load_bindings(path: Path) -> tuple[ModuleType, ct.CDLL]:
    spec = importlib.util.spec_from_file_location("generated_webgpu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bindings from {path}")
    bindings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bindings)
    library = ct.CDLL(str(path.resolve().parent / platform_config(sys.platform).dawn_library))
    bind_procedures(bindings, library)
    return bindings, library


def bind_procedures(bindings: ModuleType, library: ct.CDLL) -> None:
    """Bind each generated WGPUProc type through Dawn's procedure resolver."""
    resolve = library.dawn_ort_get_proc_address
    resolve.restype = ct.c_void_p
    resolve.argtypes = [ct.c_char_p]
    for type_name, procedure_type in list(vars(bindings).items()):
        if type_name.startswith("WGPUProc") and type_name != "WGPUProc":
            name = "wgpu" + type_name.removeprefix("WGPUProc")
            pointer = resolve(name.encode("ascii"))
            if not pointer:
                raise RuntimeError("Dawn procedure missing: " + name)
            setattr(bindings, name, procedure_type(pointer))
