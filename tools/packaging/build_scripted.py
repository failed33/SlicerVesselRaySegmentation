"""Build and unpack a scripted-only CPack archive for a specific Slicer runtime."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


def build_package(
    output: Path,
    version: str,
    *,
    inference_wheel: Path | None = None,
    desktop_runtime: Path | None = None,
    wasm_runtime: Path | None = None,
) -> Path:
    source = Path(__file__).resolve().parent
    output = output.resolve()
    install = output / "install" / "ext" / "VesselRaySegmentation"
    # Removed manifest entries must disappear from both delivery formats on rebuild.
    for directory in (output / "staged", install, output / "unpacked"):
        if directory.exists():
            shutil.rmtree(directory)
    for command in (
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(output),
            f"-DSLICER_VERSION={version}",
            f"-DCMAKE_INSTALL_PREFIX={install}",
            f"-DVRS_INFERENCE_WHEEL={inference_wheel.resolve() if inference_wheel else ''}",
            f"-DVRS_DESKTOP_RUNTIME_DIR={desktop_runtime.resolve() if desktop_runtime else ''}",
            f"-DVRS_WASM_RUNTIME_DIR={wasm_runtime.resolve() if wasm_runtime else ''}",
        ],
        ["cmake", "--build", str(output)],
        ["cmake", "--install", str(output)],
        ["cpack", "--config", str(output / "CPackConfig.cmake"), "-B", str(output)],
    ):
        subprocess.run(command, check=True)
    archive = output / f"VesselRaySegmentation-scripted-Slicer-{version}.zip"
    with zipfile.ZipFile(archive) as package:
        package.extractall(output / "unpacked")
        for member in package.infolist():
            mode = member.external_attr >> 16
            if mode:
                (output / "unpacked" / member.filename).chmod(mode & 0o777)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slicer-version", required=True)
    parser.add_argument("--inference-wheel", type=Path)
    parser.add_argument("--desktop-runtime", type=Path)
    parser.add_argument("--wasm-runtime", type=Path)
    arguments = parser.parse_args()
    print(
        build_package(
            arguments.output,
            arguments.slicer_version,
            inference_wheel=arguments.inference_wheel,
            desktop_runtime=arguments.desktop_runtime,
            wasm_runtime=arguments.wasm_runtime,
        )
    )
