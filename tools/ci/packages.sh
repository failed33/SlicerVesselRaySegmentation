#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"

export UV_PROJECT_ENVIRONMENT="$PWD/build/backend/installed"
export VRS_TEST_ENDPOINT_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
uv sync --project deployment/monai --frozen --no-install-project --no-editable --group dev --extra arg --extra model-o --extra kubernetes

# Build wheels from source distributions
uv build deployment/inference --out-dir build/backend/dist
uv build deployment/monai --out-dir build/backend/dist
uv pip install --python build/backend/installed/bin/python --no-deps --reinstall build/backend/dist/*.whl


# Test packaged runtime asset integrity
build/backend/installed/bin/python -m unittest discover -s tests/integration/runtime -v

# Test standalone backend behavior
(cd deployment/monai
../../build/backend/installed/bin/python -m pytest -q tests --junitxml=../../build/backend/monai-tests.xml
)

# Test workflow transport against native MONAI Label
build/backend/installed/bin/python -m pytest -q VesselRaySegmentation/Testing/Python --junitxml=build/backend/workflow-tests.xml

# Accept installed wheels outside the checkout
build/backend/installed/bin/python -I tests/integration/backend/wheels.py --output build/backend/wheels.json

# Keep Docker bind mounts in the build volume shared with the job-local daemon.
mkdir -p build/container-tmp
export TMPDIR="$PWD/build/container-tmp"
bash tools/ci/containers.sh


# Accept built containers with the synthetic ONNX model
build/backend/installed/bin/python -I tests/integration/backend/wheels.py --server-image vessel-label:ci --worker-image vessel-arg:ci --output build/backend/containers/report.json
docker image inspect vessel-arg:ci vessel-label:ci > build/backend/containers/images.json


