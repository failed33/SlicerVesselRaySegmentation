"""Check platform selection and build stages with local native inputs."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def native_inputs(tmp_path: Path) -> list[str]:
    native = Path(__file__).resolve().parents[1] / "native"
    match = re.search(r"set\(DAWN_COMMIT (\w+)\)", (native / "pins.cmake").read_text())
    assert match is not None, "Native pins must declare DAWN_COMMIT"
    commit = match[1]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    names = [
        f"dawn-headers-{commit}.tar.gz",
        f"dawn-apple-{commit}.xcframework.tar.gz",
        f"Dawn-{commit}-ubuntu-latest-Release.tar.gz",
        f"Dawn-{commit}-windows-latest-Release.tar.gz",
    ]
    digest = "sha256:" + hashlib.sha256(b"local build-graph fixture").hexdigest()
    (inputs / "release.json").write_text(
        json.dumps({"assets": [{"name": name, "digest": digest} for name in names]})
    )
    for name in [
        "include/dawn/webgpu.h",
        "include/dawn/dawn_proc_table.h",
        "macos-arm64_x86_64/libwebgpu_dawn.a",
        "lib64/libwebgpu_dawn.a",
        "lib/webgpu_dawn.lib",
    ]:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return [
        f"-DFETCHCONTENT_SOURCE_DIR_{name}={inputs}"
        for name in ("DAWN_RELEASE", "DAWN_HEADERS", "DAWN_BINARY")
    ]


@pytest.mark.skipif(shutil.which("cmake") is None, reason="CMake is a native build tool")
@pytest.mark.parametrize(
    ("system", "machine", "supported"),
    [
        ("Darwin", "arm64", True),
        ("Darwin", "x86_64", True),
        ("Linux", "x86_64", True),
        ("Windows", "AMD64", True),
        ("Linux", "aarch64", True),
        ("Linux", "arm64", True),
        ("Windows", "arm64", False),
        ("FreeBSD", "x86_64", False),
    ],
)
def test_pinned_native_targets(system: str, machine: str, supported: bool, tmp_path: Path) -> None:
    pins = Path(__file__).resolve().parents[1] / "native/pins.cmake"
    script = tmp_path / "target.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        f"set(APPLE {'ON' if system == 'Darwin' else 'OFF'})\n"
        f"set(WIN32 {'ON' if system == 'Windows' else 'OFF'})\n"
        f"set(CMAKE_SYSTEM_NAME {system})\n"
        f"set(CMAKE_SYSTEM_PROCESSOR {machine})\n"
        f'include("{pins.as_posix()}")\n'
        'message(STATUS "SOURCE_BUILD=${DAWN_BUILD_FROM_SOURCE}")\n'
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == supported, result.stderr
    if supported:
        from_source = system == "Linux"
        assert f"SOURCE_BUILD={'ON' if from_source else 'OFF'}" in result.stdout


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ninja") is None,
    reason="CMake and Ninja are native build tools",
)
@pytest.mark.parametrize("generator", ["Ninja", "Ninja Multi-Config"])
def test_native_build_stages(generator: str, tmp_path: Path, native_inputs: list[str]) -> None:
    native = Path(__file__).resolve().parents[1] / "native"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(tmp_path),
            "-G",
            generator,
            f"-DPython3_EXECUTABLE={sys.executable}",
            *native_inputs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = subprocess.run(
        [
            "cmake",
            "--build",
            str(tmp_path),
            "--config",
            "Release",
            "--target",
            "ort_provider",
            "--",
            "-n",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for stage in ("configure", "build", "install"):
        assert f"No {stage} step for 'dawn_source'" in plan
    for stage in ("patch", "configure", "build"):
        assert f"Performing {stage} step for 'ort_provider'" in plan
    assert "ort_source" not in plan
    assert "verify_headers" in plan
    build_graph = "\n".join(path.read_text() for path in tmp_path.rglob("*.ninja"))
    state_file = tmp_path / "external/tmp/ort_provider/native-ort-patches.sha256"
    assert f"-DPATCH_STATE_FILE={state_file}" in build_graph
    assert "--compile-no-warning-as-error" in build_graph
    assert "<STAMP_DIR>" not in build_graph
    for name in ("pr32688-runtime.patch", "wgsl-optional-variable-prerequisite.patch"):
        assert str(native / "patches" / name) in build_graph


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ninja") is None,
    reason="CMake and Ninja are native build tools",
)
@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_linux_builds_the_pinned_backend_before_host_and_provider(
    tmp_path: Path, native_inputs: list[str], machine: str
) -> None:
    native = Path(__file__).resolve().parents[1] / "native"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(tmp_path),
            "-G",
            "Ninja",
            "-DCMAKE_SYSTEM_NAME=Linux",
            f"-DCMAKE_SYSTEM_PROCESSOR={machine}",
            "-DCMAKE_CXX_COMPILER_WORKS=ON",
            "-DCMAKE_SKIP_RPATH=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
            *native_inputs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = subprocess.check_output(
        ["cmake", "--build", str(tmp_path), "--target", "bindings", "ort_provider", "--", "-n"],
        text=True,
    )
    assert plan.index("Performing build step for 'dawn_backend'") < plan.index(
        "Performing configure step for 'ort_provider'"
    )
    assert plan.index("Performing build step for 'dawn_backend'") < plan.index(
        "Building CXX object CMakeFiles/webgpu_dawn.dir/dawn_proc_bridge.cc.o"
    )
    graph = (tmp_path / "build.ninja").read_text()
    for option in (
        "-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC",
        "-DDAWN_SUPPORTS_CXX_MODULES=OFF",
        "-DDAWN_FETCH_DEPENDENCIES=ON",
        "-DDAWN_ENABLE_VULKAN=ON",
        "-DDAWN_ENABLE_NULL=OFF",
        "-DDAWN_USE_X11=OFF",
        "-DDAWN_USE_WAYLAND=OFF",
    ):
        assert option in graph
    assert "dawn-backend/lib/libwebgpu_dawn.a" in graph
    assert "C_INCLUDE_PATH=" in graph
    assert "verify_headers" in plan


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ninja") is None,
    reason="CMake and Ninja are native build tools",
)
@pytest.mark.parametrize(
    "patch_name", ["pr32688-runtime.patch", "wgsl-optional-variable-prerequisite.patch"]
)
def test_native_patch_invalidates_cached_build(
    tmp_path: Path, native_inputs: list[str], patch_name: str
) -> None:
    native = tmp_path / "native"
    shutil.copytree(Path(__file__).resolve().parents[1] / "native", native)
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DPython3_EXECUTABLE={sys.executable}",
            *native_inputs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    targets = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "targets", "all"], text=True
    )
    # Preserve the generated dependencies, replacing build commands with no-ops
    # so Ninja can record a completed cache without downloads or compilation.
    graph = build / "build.ninja"
    graph.write_text(
        re.sub(r"^  COMMAND = .+$", "  COMMAND = cmake -E true", graph.read_text(), flags=re.M)
    )
    timestamp = time.time_ns()
    for target in targets.splitlines():
        name, rule = target.rsplit(": ", 1)
        if rule == "CUSTOM_COMMAND":
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            os.utime(path, ns=(timestamp, timestamp))
    subprocess.run(["ninja", "-C", str(build), "ort_provider"], check=True, capture_output=True)
    command = ["ninja", "-C", str(build), "-n", "ort_provider"]
    unchanged = subprocess.check_output(command, text=True)
    assert "no work to do" in unchanged
    patch = native / "patches" / patch_name
    patch.write_bytes(patch.read_bytes() + b"\n")
    changed = subprocess.check_output(command, text=True)
    for stage in ("patch", "configure", "build", "verify_headers"):
        assert f"Performing {stage} step for 'ort_provider'" in changed


@pytest.fixture
def patch_checkout(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    for name in ("operator", "helper", "unrelated"):
        (repo / name).write_text("before\n")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    patches = tmp_path / "patches"
    patches.mkdir()
    for name in ("operator", "helper"):
        (patches / f"{name}.patch").write_text(
            f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n"
            "@@ -1 +1 @@\n-before\n+after\n"
        )
    return repo, patches, commit


def run_patch_step(checkout: tuple[Path, Path, str]) -> subprocess.CompletedProcess[str]:
    repo, patches, commit = checkout
    script = Path(__file__).resolve().parents[1] / "native/apply_patches.cmake"
    return subprocess.run(
        [
            "cmake",
            f"-DGIT_EXECUTABLE={shutil.which('git')}",
            f"-DSOURCE_DIR={repo}",
            f"-DEXPECTED_COMMIT={commit}",
            f"-DRUNTIME_PATCH={patches / 'operator.patch'}",
            f"-DWGSL_PATCH={patches / 'helper.patch'}",
            f"-DPATCH_STATE_FILE={patches.parent / 'applied.sha256'}",
            "-P",
            str(script),
        ],
        capture_output=True,
        text=True,
    )


def test_native_patch_step_is_repeatable(patch_checkout: tuple[Path, Path, str]) -> None:
    repo, _, _ = patch_checkout
    (repo / "unrelated").write_text("local edit\n")
    for _ in range(2):
        result = run_patch_step(patch_checkout)
        assert result.returncode == 0, result.stderr
        assert (repo / "operator").read_text() == "after\n"
        assert (repo / "helper").read_text() == "after\n"
        assert (repo / "unrelated").read_text() == "local edit\n"


@pytest.mark.parametrize("change", ["changed", "invalid", "removed", "source_edit"])
def test_native_patch_step_rejects_stale_cache(
    patch_checkout: tuple[Path, Path, str],
    change: str,
) -> None:
    repo, patches, _ = patch_checkout
    assert run_patch_step(patch_checkout).returncode == 0
    (repo / "unrelated").write_text("local edit\n")
    patch = patches / "operator.patch"
    if change == "changed":
        patch.write_text(patch.read_text().replace("+after", "+replacement"))
    elif change == "invalid":
        patch.write_text("invalid patch\n")
    elif change == "removed":
        patch.unlink()
    else:
        (repo / "operator").write_text("edited after patching\n")
    before = {name: (repo / name).read_bytes() for name in ("operator", "helper", "unrelated")}
    result = run_patch_step(patch_checkout)
    assert result.returncode != 0
    assert {name: (repo / name).read_bytes() for name in before} == before
    if change != "removed":
        assert "fresh native build directory" in " ".join(result.stderr.split())
