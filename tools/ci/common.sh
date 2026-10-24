#!/usr/bin/env bash
# Shared acceptance commands run from the extension root on either CI platform.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/build/cache/uv}"
export npm_config_cache="${npm_config_cache:-$PWD/build/cache/npm}"
export ORT_DISABLE_TELEMETRY=1 MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
export CCACHE_DIR="${CCACHE_DIR:-$PWD/build/cache/ccache}"
export CCACHE_BASEDIR="$PWD" CCACHE_COMPILERCHECK=content CCACHE_NOHASHDIR=true
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-2G}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

runtime_value() {
  python -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])[sys.argv[2]])' "$1" "$2"
}

software_vulkan() {
  local drivers=(/usr/share/vulkan/icd.d/lvp_icd*.json)
  if [[ "${#drivers[@]}" != 1 || ! -f "${drivers[0]}" ]]; then
    echo "Expected one installed Mesa software Vulkan driver" >&2
    return 1
  fi
  export VK_DRIVER_FILES="${drivers[0]}"
  export ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER=1
}

require_software_adapter() {
  python -c 'import json,sys; report=json.load(open(sys.argv[1])); assert report["adapter"]["software"], report["adapter"]' "$1"
}
