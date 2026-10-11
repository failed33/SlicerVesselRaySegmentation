"""Verify the built host and generate ctypes declarations for its pinned headers."""

import ctypes as ct
import os
import subprocess
import sys
from pathlib import Path

import clang


def main() -> None:
    library_path, include_dir, output_path, dawn_commit, libclang_name = sys.argv[1:]
    library = ct.CDLL(library_path)
    library.dawn_ort_get_proc_version.restype = ct.POINTER(ct.c_uint8)
    actual_commit = bytes(library.dawn_ort_get_proc_version()[:20]).hex()
    if actual_commit != dawn_commit:
        raise RuntimeError(
            f"Dawn procedure-table identity mismatch: expected {dawn_commit}, got {actual_commit}"
        )
    libclang = Path(clang.__file__).parent / "native" / libclang_name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ctypeslib.clang2py",
            str(Path(include_dir) / "dawn/webgpu.h"),
            "-o",
            output_path,
        ],
        env={**os.environ, "CLANG_LIBRARY_PATH": str(libclang)},
        check=True,
    )


if __name__ == "__main__":
    main()
