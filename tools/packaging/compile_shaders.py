"""Regenerate packaged WGSL and reflection with the pinned Slang compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "deployment/inference/src"
SHADERS = SOURCE / "model_o_inference/webgpu"
VERSION = "2026.12"
ASSETS = {
    ("Darwin", "arm64"): (
        "macos-aarch64",
        "e77cf4168b9c225cfed7fe9054b8360d8b6a30dc8c2dba209c82b9cb1fd69646",
    ),
    ("Linux", "x86_64"): (
        "linux-x86_64",
        "495682d2fd145dcc93cf762ab92e9fec1803f0c578dd0594c5d767954c46fb7b",
    ),
}
FLAGS = ["-target", "wgsl", "-entry", "main", "-stage", "compute", "-fp-mode", "precise"]


def download_compiler(cache: Path) -> Path:
    target, digest = ASSETS[(platform.system(), platform.machine())]
    filename = f"slang-{VERSION}-{target}.zip"
    archive = cache / filename
    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = f"https://github.com/shader-slang/slang/releases/download/v{VERSION}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            with archive.with_suffix(".partial").open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        archive.with_suffix(".partial").replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Slang compiler checksum mismatch: {archive}")
    destination = cache / target
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid compiler archive path: {relative}")
            path = destination.joinpath(*relative.parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(package.read(member))
            path.chmod(stat.S_IMODE(member.external_attr >> 16) or 0o644)
    return destination / "bin/slangc"


def constants() -> dict[str, int]:
    sys.path.insert(0, str(SOURCE))
    from model_o_inference.constants import SHADER_CONSTANTS

    if SHADER_CONSTANTS["RAY_LENGTH"] % 4:
        raise ValueError("Packed mask words must belong to one ray: RAY_LENGTH % 4 != 0")
    return SHADER_CONSTANTS


def compile_shaders(compiler: Path, output: Path) -> None:
    version = subprocess.run(
        [str(compiler), "-version"], check=True, capture_output=True, text=True
    )
    if (version.stdout + version.stderr).strip() != VERSION:
        raise ValueError(f"Expected Slang {VERSION}, got {version.stdout}{version.stderr}")
    values = constants()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("patch_inputs",):
        source = SHADERS / f"{name}.slang"
        code, reflection = output / f"{name}.wgsl", output / f"{name}.json"
        flags = [*FLAGS, *(f"-D{key}={value}" for key, value in values.items())]
        subprocess.run(
            [
                str(compiler),
                source.relative_to(ROOT).as_posix(),
                *flags,
                "-o",
                str(code),
                "-reflection-json",
                str(reflection),
            ],
            cwd=ROOT,
            check=True,
        )
        metadata = json.loads(reflection.read_text())
        metadata["build"] = {
            "compiler": VERSION,
            "flags": flags,
            "constants": values,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "wgsl_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
        }
        reflection.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "build/slang")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    compiler = args.compiler.resolve() if args.compiler else download_compiler(args.cache.resolve())
    with tempfile.TemporaryDirectory() as staging:
        output = Path(staging) if args.check else SHADERS / "generated"
        compile_shaders(compiler, output)
        if args.check:
            for path in output.iterdir():
                packaged = SHADERS / "generated" / path.name
                if not packaged.is_file() or packaged.read_bytes() != path.read_bytes():
                    raise ValueError(f"Generated shader is stale: {packaged}")
    print(json.dumps({"status": "verified" if args.check else "generated", "compiler": VERSION}))


if __name__ == "__main__":
    main()
