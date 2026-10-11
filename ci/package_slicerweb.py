"""Use upstream SlicerWeb's extension wheel builder on the CMake install tree."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def package(source: Path, build: Path, version: str) -> None:
    build = build.resolve()
    spec = importlib.util.spec_from_file_location(
        "slicerweb_wheels", source / "scripts/make_wheels.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("SlicerWeb wheel builder is unavailable")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    sources = build / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "VesselRaySegmentation").symlink_to(
        Path(__file__).resolve().parents[1], target_is_directory=True
    )
    builder.write_extension_wheels(
        SimpleNamespace(
            install=str(build / "install"),
            src=str(sources),
            extensions_out=str(build / "wheels"),
        ),
        version,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--slicer-version", required=True)
    arguments = parser.parse_args()
    package(arguments.source, arguments.build, arguments.slicer_version)
