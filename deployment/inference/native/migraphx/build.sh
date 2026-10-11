#!/usr/bin/env bash
set -euo pipefail

# Build only the standalone EP against the installed ROCm 7.1.1 packages.
# Usage: bash build.sh OUTPUT_DIRECTORY [ROCM_PREFIX]
arg_migraphx_output="$(realpath -m "${1:?An output directory is required}")"
arg_rocm_prefix="${2:-/opt/rocm}"
arg_recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
arg_plugin_commit=87198f3ac6315d7fe7a0e10454fb29a21eae3021
arg_plugin_sha256=6db5520fddf0b704b74cccbed491df38fabe6ef47048f896db10eb528b5ec200
arg_sdk_sha256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd

if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
    echo 'This pinned ROCm 7.1.1 recipe requires Linux x86_64.' >&2
    exit 1
fi
python3 - "$arg_rocm_prefix" <<'PY'
import re
import sys
from pathlib import Path

prefix = Path(sys.argv[1])
header = (prefix / "include/migraphx/version.h").read_text()
version = tuple(int(re.search(rf"^#define MIGRAPHX_VERSION_{part} (\d+)$", header, re.M)[1])
                for part in ("MAJOR", "MINOR", "PATCH"))
if version != (2, 14, 0):
    raise SystemExit("This compatibility patch requires matching MIGraphX 2.14.0 headers")
if not (prefix / ".info/version").read_text().strip().startswith("7.1.1"):
    raise SystemExit("This recipe requires the ROCm 7.1.1 development packages")
PY

mkdir -p "$arg_migraphx_output"
arg_build_dir="$(mktemp -d)"
trap 'rm -rf "$arg_build_dir"' EXIT
curl --fail --location --retry 3 \
    "https://codeload.github.com/onnxruntime/onnxruntime-ep-amdgpu/tar.gz/$arg_plugin_commit" \
    --output "$arg_build_dir/plugin.tar.gz"
curl --fail --location --retry 3 \
    https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz \
    --output "$arg_build_dir/sdk.tgz"
printf '%s  %s\n' "$arg_plugin_sha256" "$arg_build_dir/plugin.tar.gz" \
    "$arg_sdk_sha256" "$arg_build_dir/sdk.tgz" | sha256sum --check
mkdir "$arg_build_dir/plugin"
tar -xzf "$arg_build_dir/plugin.tar.gz" --strip-components=1 -C "$arg_build_dir/plugin"
tar -xzf "$arg_build_dir/sdk.tgz" -C "$arg_build_dir"
cp "$arg_recipe_dir/rocm711_operators.cc" "$arg_build_dir/plugin/src/migraphx/"
git -C "$arg_build_dir/plugin" apply "$arg_recipe_dir/rocm711.patch"

python3 "$arg_build_dir/plugin/tools/ci_build/build.py" \
    --config Release --update --build --parallel "${ARG_MIGRAPHX_BUILD_JOBS:-2}" \
    --build_dir "$arg_build_dir/build" --cmake_generator Ninja \
    --use_migraphx --migraphx_home "$arg_rocm_prefix" --hip_path "$arg_rocm_prefix" \
    --onnxrt_home "$arg_build_dir/onnxruntime-linux-x64-1.30.0" \
    --build_wheel --no-migraphx_libs_dependency --compile_no_warning_as_error \
    --cmake_extra_defines USE_AMDGPU=OFF USE_DML=OFF USE_HIP=OFF
arg_plugin_wheel="$(find "$arg_build_dir/build/Release/dist" -name 'onnxruntime_ep_migraphx-*.whl' -print -quit)"
test -n "$arg_plugin_wheel"
cp "$arg_plugin_wheel" "$arg_migraphx_output/"
