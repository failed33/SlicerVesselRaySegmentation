#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"

export EM_COMPILER_WRAPPER=ccache
export EM_CACHE="$PWD/build/cache/emscripten"

# Compile patched ORT and its matching WebGPU JavaScript bindings
python tools/packaging/build_wasm.py --output build/wasm --jobs 2

# Build the shared Python wheel
uv build deployment/inference --out-dir build/backend/dist

