#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"

uv sync --project deployment/inference --frozen --group native --group test
deployment/inference/.venv/bin/python -m pytest -q deployment/inference/tests --junitxml=build/backend/inference-tests.xml

# Compile the pinned host bridge and patched provider
cmake -S deployment/inference/native -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/deployment/inference/.venv/bin/python" -DCMAKE_INSTALL_PREFIX="$PWD/build/native/assets" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build/native --parallel 2
cmake --install build/native


export ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER=1
# Bundle and test relocated native libraries
python tools/packaging/bundle_native.py --build build/native --output build/backend/native
deployment/inference/.venv/bin/python tests/integration/backend/native_bundle.py build/backend/native/*.zip --output build/backend/native/report.json


# Bundle the isolated desktop Python runtime
uv build deployment/inference --out-dir build/backend/dist
python tools/packaging/bundle_desktop.py --native-archive build/backend/native/dawn-ort-linux-x86_64.zip \
  --wheel build/backend/dist/vessel_inference-0.1.0-py3-none-any.whl --output build/desktop-runtime
tar -czf build/desktop-runtime.tar.gz -C build/desktop-runtime/desktop-linux-x86_64 .


