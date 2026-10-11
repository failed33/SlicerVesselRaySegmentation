"""Build and unpack a scripted-only CPack archive for a specific Slicer runtime."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


def build_package(output: Path, version: str) -> Path:
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
        ],
        ["cmake", "--build", str(output)],
        ["cmake", "--install", str(output)],
        ["cpack", "--config", str(output / "CPackConfig.cmake"), "-B", str(output)],
    ):
        subprocess.run(command, check=True)
    archive = output / f"VesselRaySegmentation-scripted-Slicer-{version}.zip"
    with zipfile.ZipFile(archive) as package:
        package.extractall(output / "unpacked")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slicer-version", required=True)
    arguments = parser.parse_args()
    print(build_package(arguments.output, arguments.slicer_version))
