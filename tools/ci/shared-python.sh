#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"

# Accept shared numerical code without native or server dependencies
uv build deployment/inference --out-dir build/shared-python/dist
uv venv --python "${PYTHON_VERSION:-python}" build/shared-python/env
uv pip install --python build/shared-python/env/bin/python \
  'build/shared-python/dist/vessel_inference-0.1.0-py3-none-any.whl[numerical]' \
  "numpy==${NUMPY_VERSION}" "scipy==${SCIPY_VERSION}"
build/shared-python/env/bin/python -I tests/integration/backend/shared_python.py --output build/shared-python/report.json


