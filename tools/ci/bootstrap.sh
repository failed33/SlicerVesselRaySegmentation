#!/usr/bin/env bash
# Ubuntu 24.04 matches the public hosted acceptance environment.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3.12-venv python-is-python3 \
  cmake make g++ ninja-build patch pkg-config
python -m venv /opt/ci-tools
/opt/ci-tools/bin/pip install uv==0.12.23
case "$1" in
  native)
    apt-get install -y --no-install-recommends ccache libvulkan1 mesa-vulkan-drivers libegl1 libxext6 ;;
  wasm|web)
    apt-get install -y --no-install-recommends ccache xz-utils
    curl --fail --silent --show-error --location https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-x64.tar.xz -o /tmp/node.tar.xz
    echo 'fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6  /tmp/node.tar.xz' | sha256sum --check
    tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1
    rm /tmp/node.tar.xz ;;
  desktop)
    apt-get install -y --no-install-recommends xvfb libglu1-mesa libopengl0 libpulse-mainloop-glib0 \
      libnss3 libasound2t64 qt5dxcb-plugin libsm6 libxcb-cursor0 libvulkan1 mesa-vulkan-drivers libegl1 libxext6 ;;
  packages)
    apt-get install -y --no-install-recommends docker.io docker-buildx libgomp1 ;;
  shared-python|shaders) ;;
  *) echo "Unknown acceptance environment: $1" >&2; exit 2 ;;
esac
