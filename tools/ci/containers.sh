#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=tools/ci/common.sh
source "$(dirname "$0")/common.sh"
user_args=()
if [[ "$(id -u)" != 0 ]]; then
  user_args=(--build-arg "MONAI_UID=$(id -u)" --build-arg "MONAI_GID=$(id -g)")
fi
if [[ -n "${CI_REGISTRY:-}" ]]; then
  printf '%s' "$CI_REGISTRY_PASSWORD" | docker login "$CI_REGISTRY" --username "$CI_REGISTRY_USER" --password-stdin
  mkdir -p build/backend
  # This is the job-local DinD network, including its private-registry DNS.
  config=(--driver docker-container --driver-opt network=host)
  if [[ -f "${SSL_CERT_FILE:-}" ]]; then
    printf '[registry."%s"]\n  ca = ["%s"]\n' "$CI_REGISTRY" "$SSL_CERT_FILE" > build/backend/buildkitd.toml
    config+=(--buildkitd-config build/backend/buildkitd.toml)
  fi
  docker buildx create --name acceptance --use "${config[@]}"
  docker buildx inspect --bootstrap
fi
for target in arg label_arg; do
  image=vessel-arg:ci
  [[ "$target" != label_arg ]] || image=vessel-label:ci
  cache=()
  if [[ -n "${CI_REGISTRY_IMAGE:-}" ]]; then
    cache_ref="$CI_REGISTRY_IMAGE/ci-cache/$target:ci-cache-$CI_COMMIT_REF_SLUG"
    cache=(--cache-from "type=registry,ref=$cache_ref"
      --cache-from "type=registry,ref=$CI_REGISTRY_IMAGE/ci-cache/$target:ci-cache-$CI_DEFAULT_BRANCH"
      --cache-to "type=registry,ref=$cache_ref,mode=max,compression=zstd,ignore-error=true")
  fi
  docker buildx build --load -f deployment/monai/Dockerfile.workers --target "$target" \
    "${user_args[@]}" \
    "${cache[@]}" -t "$image" .
done
