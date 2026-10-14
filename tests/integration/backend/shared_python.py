"""Accept the installed numerical profile without native or serving packages."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import runpy
import sys
import sysconfig
from pathlib import Path

_FORBIDDEN = (
    "onnxruntime",
    "dawn_ort",
    "SimpleITK",
    "pydantic",
    "mlflow",
    "torch",
    "monai",
    "monailabel",
    "onnx",
    "skimage",
)


def check_numerical_profile(application: Path) -> dict[str, object]:
    # dawn_ort ships in the same wheel but must not load in the numerical profile.
    unexpected = [
        name
        for name in _FORBIDDEN
        if name != "dawn_ort" and importlib.util.find_spec(name) is not None
    ]
    if unexpected:
        raise AssertionError(f"Numerical environment contains unrelated dependencies: {unexpected}")
    modules = (
        "arg_inference",
        "arg_inference.plans",
        "model_o_inference.geometry",
        "model_o_inference.preparation",
        "model_o_inference.planning",
        "model_o_inference.stitching",
        "model_o_inference.shader_contract",
        "model_o_inference.shaders",
    )
    installed = Path(sysconfig.get_path("purelib")).resolve()
    locations = {}
    for name in modules:
        module = importlib.import_module(name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(installed):
            raise AssertionError(f"Imported {name} outside the installed wheel: {location}")
        locations[name] = str(location)
    fixture = runpy.run_path(str(application))
    result = fixture["application_summary"]()
    loaded = {name.split(".")[0] for name in sys.modules}
    unexpected = sorted(loaded.intersection(_FORBIDDEN))
    if unexpected:
        raise AssertionError(f"Numerical preparation imported unrelated packages: {unexpected}")
    load_shader = importlib.import_module("model_o_inference.shaders").load_shader
    if not all(load_shader(name).code for name in ("patch_inputs",)):
        raise AssertionError("The installed wheel omitted ModelO WGSL")
    return {
        "status": "passed",
        "python": sys.version.split()[0],
        "profile": "numerical",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vessel-inference", "numpy", "scipy")
        },
        "source_imports": False,
        "module_paths": locations,
        "unrelated_dependencies": [],
        "application": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    application = Path(__file__).resolve().parents[1] / "runtime/application.py"
    report = check_numerical_profile(application)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
