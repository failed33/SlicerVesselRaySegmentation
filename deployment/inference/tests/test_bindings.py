"""Exercise the procedure loader as ordinary Python, outside generated text."""

import ctypes as ct
from types import ModuleType
from unittest.mock import Mock

import pytest
from dawn_ort._bindings import bind_procedures


def test_generated_procedure_calls_resolved_native_function():
    procedure_type = ct.CFUNCTYPE(ct.c_uint32, ct.c_uint32)
    callback = procedure_type(lambda value: value + 1)
    declarations = ModuleType("generated_webgpu")
    setattr(declarations, "WGPUProc", ct.c_void_p)
    setattr(declarations, "WGPUProcAddOne", procedure_type)
    resolve = Mock(return_value=ct.cast(callback, ct.c_void_p).value)
    library = Mock(spec=ct.CDLL)
    library.dawn_ort_get_proc_address = resolve

    bind_procedures(declarations, library)

    resolve.assert_called_once_with(b"wgpuAddOne")
    assert declarations.wgpuAddOne(41) == 42
    assert declarations.WGPUProc is ct.c_void_p


def test_missing_native_procedure_fails_at_binding():
    declarations = ModuleType("generated_webgpu")
    setattr(declarations, "WGPUProcRequired", ct.CFUNCTYPE(None))
    library = Mock(spec=ct.CDLL)
    library.dawn_ort_get_proc_address = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="Dawn procedure missing: wgpuRequired"):
        bind_procedures(declarations, library)
