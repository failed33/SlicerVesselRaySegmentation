"""Bundle a private Python installation, locked inference dependencies and native assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PYTHON_VERSION = "3.12.15"
UV_VERSION = "0.12.23"


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def bundle(native: Path, wheel: Path, output: Path, uv: str) -> Path:
    native, wheel, output = native.resolve(), wheel.resolve(), output.resolve()
    source = Path(__file__).resolve().parents[2] / "deployment/inference"
    if run([uv, "--version"]).split()[1] != UV_VERSION:
        raise ValueError(f"Desktop packaging requires uv {UV_VERSION}")
    output.mkdir(parents=True, exist_ok=True)
    requirements = output / "requirements.txt"
    subprocess.run(
        [
            uv,
            "export",
            "--project",
            str(source),
            "--frozen",
            "--extra",
            "arg",
            "--extra",
            "model-o",
            "--no-default-groups",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ],
        check=True,
    )
    downloads = output / "downloads"
    subprocess.run(
        [
            uv,
            "python",
            "install",
            PYTHON_VERSION,
            "--install-dir",
            str(downloads),
            "--no-bin",
            "--no-registry",
        ],
        check=True,
    )
    installations = list(downloads.glob(f"cpython-{PYTHON_VERSION}-*"))
    if len(installations) != 1:
        raise ValueError(f"Expected one managed Python installation, found {installations}")
    staging = output / f"desktop-{sys.platform}-{platform.machine().lower()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    # Materialize internal symlinks so ordinary ZIP extraction preserves executable files.
    shutil.copytree(
        installations[0],
        staging / "python",
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    executable = staging / "python" / ("python.exe" if os.name == "nt" else "bin/python3")
    subprocess.run(
        [
            uv,
            "pip",
            "sync",
            "--python",
            str(executable),
            "--system",
            "--break-system-packages",
            "--require-hashes",
            str(requirements),
        ],
        check=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(executable),
            "--system",
            "--break-system-packages",
            "--no-deps",
            str(wheel),
        ],
        check=True,
    )
    assets = staging / "assets"
    with zipfile.ZipFile(native) as archive:
        archive.extractall(assets)
    native_manifest = json.loads((assets / "manifest.json").read_text())
    for name, digest in native_manifest["files"].items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Native asset hash mismatch: {name}")
    if native_manifest["platform"] != sys.platform:
        raise ValueError("Native assets must target the desktop bundle's platform")
    if native_manifest["architecture"] != platform.machine().lower():
        raise ValueError("Native assets must target the desktop bundle's architecture")
    metadata = json.loads(
        run(
            [
                str(executable),
                "-I",
                "-B",
                "-c",
                """
import importlib.metadata, importlib.util, json, os, pathlib, sys
os.environ.setdefault('ORT_DISABLE_TELEMETRY', '1')
import numpy, onnxruntime
import model_o_inference.preparation, arg_inference.plans
for name in ('torch', 'triton', 'nnunetv2', 'monailabel', 'mlflow', 'kubernetes'):
    assert importlib.util.find_spec(name) is None, name
assert pathlib.Path(onnxruntime.__file__).is_relative_to(pathlib.Path(sys.prefix))
print(json.dumps({'python': sys.version.split()[0], 'onnxruntime': onnxruntime.__version__,
 'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())}))
""",
            ]
        )
    )
    if metadata["onnxruntime"] != native_manifest["onnxruntime"]:
        raise ValueError("Python ORT core and compiled provider versions differ")
    shutil.copyfile(requirements, staging / "requirements.txt")
    manifest = {
        "platform": sys.platform,
        "architecture": platform.machine().lower(),
        "uv": UV_VERSION,
        "python_executable": executable.relative_to(staging).as_posix(),
        "inference_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest(),
        "native_manifest": native_manifest,
        **metadata,
        "files": {
            path.relative_to(staging).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "bundle": str(staging),
                "python": str(executable),
                "packages": len(metadata["packages"]),
            }
        )
    )
    return staging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-archive", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    bundle(args.native_archive, args.wheel, args.output, args.uv)
