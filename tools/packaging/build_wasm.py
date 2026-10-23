"""Compile and package the pinned ONNX Runtime C++ WebGPU WASM component."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = ROOT / "deployment/inference"
ORT_URL = "https://github.com/microsoft/onnxruntime.git"
DAWN_URL = "https://github.com/google/dawn.git"
WASM_NAME = "ort-wasm-simd-threaded.asyncify"
RUNTIME_FILES = ("ort.webgpu.min.mjs", f"{WASM_NAME}.mjs", f"{WASM_NAME}.wasm")
COMPILE_DIRECTORY = "compile-pinned"
DAWN_PATCHES = (
    "dawn_destroy_buffer_on_destructor.patch",
    "dawn_binskim.patch",
    "safari_polyfill.patch",
    "dawn_device_lost_keepalive.patch",
    "dawn_dxc_output_dir.patch",
    "dawn_parallel_build_fix.patch",
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_pins(inference: Path = INFERENCE) -> dict[str, str]:
    text = (inference / "native/pins.cmake").read_text()
    pins = {}
    for key in ("ORT_COMMIT", "DAWN_COMMIT"):
        matches = re.findall(rf"^set\({key} ([0-9a-f]{{40}})\)$", text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Expected one exact {key} assignment in native/pins.cmake")
        pins[key] = matches[0]
    return pins


def patch_paths(inference: Path = INFERENCE) -> tuple[Path, ...]:
    return (
        inference / "native/patches/pr32688-runtime.patch",
        inference / "native/patches/wgsl-optional-variable-prerequisite.patch",
        inference / "wasm/ort-webgpu-esm.patch",
    )


def source_identity(inference: Path = INFERENCE) -> dict[str, object]:
    toolchain = json.loads((inference / "wasm/toolchain.json").read_text())
    return {
        **read_pins(inference),
        **toolchain,
        "patches": [
            {"path": str(path.relative_to(inference)), "sha256": sha256(path)}
            for path in patch_paths(inference)
        ],
    }


def validate_output(output: Path, root: Path = ROOT) -> Path:
    output = output.resolve()
    build = (root / "build").resolve()
    if not output.is_relative_to(build) or output == build:
        raise ValueError(
            "Choose a dedicated output directory beneath the repository's ignored build/"
        )
    return output


def run(command: list[str], cwd: Path, output: Path, env: dict[str, str] | None = None) -> None:
    print(f"Running: {shlex.join(command)}", flush=True)
    with (output / "build.log").open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\nRunning: {shlex.join(command)}\n")
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def prepare_checkout(name: str, url: str, commit: str, output: Path, cache: Path | None) -> Path:
    source = output / "sources" / name
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        if cache is not None:
            run(["git", "clone", "--no-hardlinks", str(cache.resolve()), str(source)], ROOT, output)
            run(["git", "remote", "set-url", "origin", url], source, output)
        else:
            run(["git", "init", str(source)], ROOT, output)
            run(["git", "remote", "add", "origin", url], source, output)
        run(["git", "fetch", "--depth", "1", "origin", commit], source, output)
        run(["git", "checkout", "--detach", commit], source, output)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != commit:
        raise ValueError(f"{name} source identity mismatch: {actual}; use a fresh output directory")
    return source


def ensure_commonjs_boundary(source: Path) -> None:
    # ORT's root has no package.json. Without this boundary, Node can inherit
    # type=module from the enclosing application's unrelated package.json and
    # misinterpret upstream CommonJS generators such as wasm_post_build.js.
    boundary = source / "package.json"
    expected = {"private": True, "type": "commonjs"}
    if boundary.exists() and json.loads(boundary.read_text()) != expected:
        raise ValueError("Unexpected ORT root package.json; use a fresh output directory")
    boundary.write_text(json.dumps(expected, indent=2) + "\n")


def ensure_typescript_boundary(source: Path) -> None:
    # TypeScript otherwise discovers ambient types in every ancestor's
    # node_modules. Keep the upstream build independent of an enclosing app.
    path = source / "js/tsconfig.json"
    config = json.loads(path.read_text())
    options = config["compilerOptions"]
    expected = ["./node_modules/@types"]
    if "typeRoots" in options and options["typeRoots"] != expected:
        raise ValueError("Unexpected ORT base typeRoots; use a fresh output directory")
    options["typeRoots"] = expected
    path.write_text(json.dumps(config, indent=2) + "\n")


def apply_patches(source: Path, output: Path, pins: dict[str, str]) -> None:
    runtime, wgsl, esm = patch_paths()
    run(
        [
            "cmake",
            f"-DGIT_EXECUTABLE={shutil.which('git')}",
            f"-DSOURCE_DIR={source}",
            f"-DEXPECTED_COMMIT={pins['ORT_COMMIT']}",
            f"-DRUNTIME_PATCH={runtime}",
            f"-DWGSL_PATCH={wgsl}",
            f"-DPATCH_STATE_FILE={output / 'patch-state.txt'}",
            "-P",
            str(INFERENCE / "native/apply_patches.cmake"),
        ],
        ROOT,
        output,
    )
    check = subprocess.run(
        ["git", "apply", "--check", str(esm)], cwd=source, capture_output=True, text=True
    )
    if check.returncode == 0:
        run(["git", "apply", str(esm)], source, output)
    else:
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(esm)],
            cwd=source,
            capture_output=True,
            text=True,
        )
        if reverse.returncode:
            raise ValueError(f"Cannot apply pinned ORT ESM build patch: {check.stderr}")


def apply_dawn_patches(source: Path, dawn: Path, output: Path) -> None:
    patches = [source / "cmake/patches/dawn" / name for name in DAWN_PATCHES]
    expected = {path.name: sha256(path) for path in patches}
    state = output / "dawn-patches.json"
    if state.exists():
        if json.loads(state.read_text()) != expected:
            raise ValueError("Upstream Dawn patches changed; use a fresh output directory")
    # Use exactly ORT's patch flags. Several upstream hunks need whitespace
    # tolerance or have metadata accepted by patch but rejected by git apply.
    for path in patches:
        command = ["patch", "--batch", "--binary", "--ignore-whitespace", "-p1", "-i", str(path)]
        forward = subprocess.run(
            [*command, "--forward", "--dry-run"], cwd=dawn, capture_output=True, text=True
        )
        if forward.returncode == 0:
            run([*command, "--forward"], dawn, output)
        else:
            reverse = subprocess.run(
                [*command, "--reverse", "--dry-run"], cwd=dawn, capture_output=True, text=True
            )
            if reverse.returncode:
                raise ValueError(
                    f"Cannot apply upstream Dawn patch {path.name}: {forward.stdout}{forward.stderr}"
                )
    state.write_text(json.dumps(expected, indent=2) + "\n")


def build_command(source: Path, output: Path, jobs: int, emscripten: str) -> list[str]:
    return [
        sys.executable,
        str(source / "tools/ci_build/build.py"),
        "--update",
        "--build",
        "--config",
        "Release",
        "--build_dir",
        str(output / COMPILE_DIRECTORY),
        "--build_wasm",
        "--use_webgpu",
        "--enable_wasm_simd",
        "--enable_wasm_threads",
        "--emsdk_version",
        emscripten,
        "--enable_wasm_api_exception_catching",
        "--target",
        "onnxruntime_webassembly",
        "--skip_tests",
        "--skip_submodule_sync",
        "--parallel",
        str(jobs),
        "--compile_no_warning_as_error",
        "--cmake_generator",
        "Ninja",
        "--cmake_extra_defines",
        f"FETCHCONTENT_SOURCE_DIR_DAWN={output / 'sources/dawn'}",
    ]


def build_javascript(source: Path, output: Path) -> None:
    ensure_typescript_boundary(source)
    env = {
        **os.environ,
        "ELECTRON_SKIP_BINARY_DOWNLOAD": "1",
        "npm_config_cache": os.environ.get("npm_config_cache", str(output / "npm-cache")),
    }
    for directory in ("js", "js/common", "js/web"):
        run(["npm", "ci", "--no-audit", "--no-fund"], source / directory, output, env)
    dist = source / "js/web/dist"
    dist.mkdir(exist_ok=True)
    for suffix in ("mjs", "wasm"):
        shutil.copyfile(
            output / f"{COMPILE_DIRECTORY}/Release/{WASM_NAME}.{suffix}",
            dist / f"{WASM_NAME}.{suffix}",
        )
    run(
        ["npm", "run", "build", "--", "--bundle-mode=perf", "--webgpu-ep"],
        source / "js/web",
        output,
        env,
    )


def copy_licenses(source: Path, output: Path, staging: Path) -> None:
    shutil.copyfile(source / "LICENSE", staging / "LICENSE")
    shutil.copyfile(source / "ThirdPartyNotices.txt", staging / "ThirdPartyNotices.txt")
    for directory, label in (
        (source, "ort"),
        (output / "sources/dawn", "dawn"),
        (output / COMPILE_DIRECTORY / "Release/_deps", "dependencies"),
        (source / "js/node_modules", "npm/build"),
        (source / "js/common/node_modules", "npm/common"),
        (source / "js/web/node_modules", "npm/web"),
        (source / "cmake/external/emsdk/upstream/emscripten/system/lib", "emscripten/system"),
    ):
        for pattern in ("*LICENSE*", "*NOTICE*", "*ThirdPartyNotices*", "*COPYRIGHT*", "*COPYING*"):
            for path in directory.rglob(pattern):
                if not path.is_file() or any(
                    part in {".git", "node_modules", "emsdk"}
                    for part in path.relative_to(directory).parts
                ):
                    continue
                target = staging / "licenses" / label / path.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    emscripten = source / "cmake/external/emsdk/upstream/emscripten"
    for name in ("LICENSE", "AUTHORS"):
        target = staging / "licenses/emscripten" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(emscripten / name, target)
    # The C++ WebGPU build uses the canonical Dawn source override.
    if not (staging / "licenses/dawn/LICENSE").is_file():
        raise ValueError("Missing fetched Dawn license")


def package(source: Path, output: Path, identity: dict[str, object]) -> Path:
    source_dist = source / "js/web/dist"
    for name in RUNTIME_FILES:
        path = source_dist / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing compiled runtime artifact: {path}")
    with (source_dist / f"{WASM_NAME}.wasm").open("rb") as binary:
        if binary.read(8) != b"\0asm\x01\0\0\0":
            raise ValueError("Runtime artifact is not a WebAssembly version 1 binary")
    version = (source / "VERSION_NUMBER").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"Unexpected ORT version: {version}")
    recipe_hash = sha256(Path(__file__))
    patch_set = hashlib.sha256(
        json.dumps({**identity, "build_recipe_sha256": recipe_hash}, sort_keys=True).encode()
    ).hexdigest()
    bundle_id = f"onnxruntime-webgpu-{version}-{str(identity['ORT_COMMIT'])[:8]}-{patch_set[:12]}"
    staging = output / "dist.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name in RUNTIME_FILES:
        shutil.copyfile(source_dist / name, staging / name)
    copy_licenses(source, output, staging)
    emsdk = source / "cmake/external/emsdk"
    manifest = {
        "format_version": 1,
        "bundle_id": bundle_id,
        "runtime_version": version,
        **identity,
        "execution_provider": "WebGPUExecutionProvider",
        "jsep": False,
        "wasm": {"simd": True, "threads_compiled": True, "runtime_threads": 1, "async": "asyncify"},
        "emsdk_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=emsdk, text=True
        ).strip(),
        "tools": {
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "python": sys.version.split()[0],
        },
        "upstream_dependencies_sha256": sha256(source / "cmake/deps.txt"),
        "build_recipe_sha256": recipe_hash,
        "source_commonjs_boundary_sha256": sha256(source / "package.json"),
        "source_typescript_boundary_sha256": sha256(source / "js/tsconfig.json"),
        "source_diffs": {
            name: hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=directory)
            ).hexdigest()
            for name, directory in (("ort", source), ("dawn", output / "sources/dawn"))
        },
        "dawn_upstream_patches": {
            name: sha256(source / "cmake/patches/dawn" / name) for name in DAWN_PATCHES
        },
        "dawn_headers": {
            name: sha256(output / "sources/dawn" / name)
            for name in (
                "src/dawn/dawn.json",
                "include/webgpu/webgpu.h",
                "third_party/emdawnwebgpu/pkg/webgpu/src/webgpu.cpp",
            )
        },
        "dawn_generated_headers": {
            "webgpu/webgpu.h": sha256(
                output
                / COMPILE_DIRECTORY
                / "Release/_deps/dawn-build/gen/src/emdawnwebgpu/include/webgpu/webgpu.h"
            )
        },
        "convtranspose3d_wgsl_sha256": sha256(
            output
            / COMPILE_DIRECTORY
            / "Release/wgsl_generated/wgsl_template_gen/generated/nn/conv_transpose3d.h"
        ),
        "npm_locks": {
            directory: sha256(source / directory / "package-lock.json")
            for directory in ("js", "js/common", "js/web")
        },
        "files": {
            str(path.relative_to(staging)): sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    dist = output / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    staging.rename(dist)
    archive = Path(shutil.make_archive(str(output / bundle_id), "zip", dist))
    (output / f"{bundle_id}.zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/wasm")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="Read-only ORT Git checkout used to seed a fresh source tree",
    )
    parser.add_argument(
        "--dawn-source-cache",
        type=Path,
        help="Read-only Dawn Git checkout used to seed a fresh source tree",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = validate_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    node_version = subprocess.check_output(["node", "--version"], text=True).strip()
    if int(node_version.removeprefix("v").split(".")[0]) != identity["node_major"]:
        raise ValueError(f"Node {identity['node_major']} required, found {node_version}")
    state = output / "source-identity.json"
    if state.exists() and json.loads(state.read_text()) != identity:
        raise ValueError("Source pins, patches or toolchain changed; use a fresh output directory")
    state.write_text(json.dumps(identity, indent=2) + "\n")
    pins = read_pins()
    source = prepare_checkout("ort", ORT_URL, pins["ORT_COMMIT"], output, args.source_cache)
    ensure_commonjs_boundary(source)
    dawn = prepare_checkout("dawn", DAWN_URL, pins["DAWN_COMMIT"], output, args.dawn_source_cache)
    run(
        ["git", "submodule", "update", "--init", "--depth", "1", "cmake/external/emsdk"],
        source,
        output,
    )
    apply_patches(source, output, pins)
    apply_dawn_patches(source, dawn, output)
    run(
        build_command(source, output, args.jobs, str(identity["emscripten_version"])),
        source,
        output,
    )
    build_javascript(source, output)
    archive = package(source, output, identity)
    print(
        json.dumps(
            {"dist": str(output / "dist"), "archive": str(archive), "sha256": sha256(archive)}
        )
    )


if __name__ == "__main__":
    main()
