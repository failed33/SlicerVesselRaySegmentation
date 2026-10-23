#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"
mode="${1:-scripted}"
export SLICERWEB_WHEELS="$PWD/build/web-runtime/wheels"
export SLICERWEB_EXTENSIONS="$PWD/build/web-package/wheels"
mkdir -p build/web-runtime
python tools/runtimes/prepare_slicerweb.py --output build/web-runtime | tee build/web-runtime/runtime.log
web_source=$(runtime_value build/web-runtime/runtime.log source)
slicer_version=$(runtime_value build/web-runtime/runtime.log slicer_version)
package_args=(--slicer-version "$slicer_version" --output build/web-package)
test_args=("$web_source" build/web-package)
if [[ "$mode" == runtime ]]; then
  uv sync --project deployment/inference --frozen --group test
  deployment/inference/.venv/bin/python tests/integration/runtime/fixtures.py --output build/runtime-fixtures
  UV_PROJECT_ENVIRONMENT="$PWD/build/workflow-endpoint" uv sync --project deployment/monai --frozen --group dev
  package_args+=(--inference-wheel build/backend/dist/vessel_inference-0.1.0-py3-none-any.whl --wasm-runtime build/wasm/dist)
  test_args+=(--fixtures build/runtime-fixtures --allow-software --endpoint-python build/workflow-endpoint/bin/python)
elif [[ "$mode" != scripted ]]; then
  echo "Unknown browser acceptance mode: $mode" >&2; exit 2
fi
python tools/packaging/build_scripted.py "${package_args[@]}"
python tools/packaging/build_slicerweb.py --source "$web_source" --slicer-version "$slicer_version" --build build/web-package
(cd "$web_source/web" && npm ci --prefer-offline --no-audit --no-fund)
if ! command -v google-chrome >/dev/null; then
  node "$web_source/web/node_modules/playwright-core/cli.js" install --with-deps chrome
fi
(cd "$web_source/web" && npm run build:app)
node tests/integration/slicer/slicerweb.mjs "${test_args[@]}"
