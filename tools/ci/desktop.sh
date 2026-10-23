#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"
channel="${1:?Choose stable or nightly}"
mode="${2:-scripted}"
export QTWEBENGINE_DISABLE_SANDBOX=1 LIBGL_ALWAYS_SOFTWARE=1
export PACKAGE_ROOT="$PWD/build/desktop-package/unpacked"
export TEST_REPORT="$PWD/build/desktop-package/desktop.json"
mkdir -p build/slicer
python tools/runtimes/download_slicer.py "$channel" --output build/slicer | tee build/slicer/runtime.log
slicer_version=$(runtime_value build/slicer/runtime.log slicer_version)
slicer_executable=$(runtime_value build/slicer/runtime.log slicer_executable)
package_args=(--slicer-version "$slicer_version" --output build/desktop-package)
if [[ "$mode" == runtime ]]; then
  export ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER=1 VRS_RUNTIME_ALLOW_SOFTWARE=1
  export VRS_RUNTIME_FIXTURES="$PWD/build/runtime-fixtures"
  export VRS_TEST_ENDPOINT_PYTHON="$PWD/build/workflow-endpoint/bin/python"
  UV_PROJECT_ENVIRONMENT="$PWD/build/workflow-endpoint" uv sync --project deployment/monai --frozen --group dev
  mkdir -p build/runtime
  tar -xzf build/desktop-runtime.tar.gz -C build/runtime
  build/runtime/python/bin/python3 tests/integration/runtime/fixtures.py --output build/runtime-fixtures
  package_args+=(--inference-wheel build/backend/dist/vessel_inference-0.1.0-py3-none-any.whl --desktop-runtime build/runtime)
elif [[ "$mode" != scripted ]]; then
  echo "Unknown desktop acceptance mode: $mode" >&2; exit 2
fi
python tools/packaging/build_scripted.py "${package_args[@]}"
xvfb-run -a "$slicer_executable" --no-splash --no-main-window --testing \
  --additional-module-paths "$PACKAGE_ROOT/lib/Slicer-$slicer_version/qt-scripted-modules" \
  --python-script "$PWD/tests/integration/slicer/desktop.py" > build/desktop-package/slicer.log 2>&1
