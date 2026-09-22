#!/usr/bin/env bash
# ============================================================================
# build.sh — build the patched vLLM + exllamav3 + vllm-exl3 image
# (docker/Dockerfile) and check it on the GPU.
#
# Run it on the Spark itself: the image is aarch64 and the extensions are
# compiled for sm_121. The first build takes a while (two CUDA extensions);
# rebuilds reuse the layer cache unless a pin in .env changed.
#
# Usage:
#   ./build.sh                 # build IMAGE from .env, then run preflight in it
#   ./build.sh --no-verify     # skip the GPU preflight
#   ./build.sh --no-cache      # rebuild every layer (e.g. VLLM_EXL3_REF=main moved)
#   VLLM_EXL3_REF=main IMAGE=qwen38-exl3:main ./build.sh
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_common.sh"
cd "$RECIPE_DIR"

VERIFY=true
DOCKER_BUILD_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --no-verify) VERIFY=false ;;
        --no-cache)  DOCKER_BUILD_FLAGS+=(--no-cache) ;;
        --pull)      DOCKER_BUILD_FLAGS+=(--pull) ;;
        -h|--help)   sed -n '3,/^# =*$/p' "$0" | sed '$d;s/^# \?//'; exit 0 ;;
        *)           err "unknown option: $arg (try --help)" ;;
    esac
done

load_env
common_defaults
require_docker

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    err "this image targets the DGX Spark (aarch64); this host is $ARCH.
       The exllamav3 aarch64 patch refuses to run elsewhere. Build on the Spark."
fi

# Only pass the pins that are set, so docker/Dockerfile stays the single
# source of defaults.
BUILD_ARGS=()
for v in BASE_IMAGE VLLM_VERSION EXLLAMAV3_REPO EXLLAMAV3_REF VLLM_EXL3_REPO VLLM_EXL3_REF \
         TORCH_CUDA_ARCH_LIST MAX_JOBS; do
    if [[ -n "${!v:-}" ]]; then
        BUILD_ARGS+=(--build-arg "$v=${!v}")
        info "  $v=${!v}"
    fi
done

info "Building $IMAGE ..."
docker build ${DOCKER_BUILD_FLAGS[@]+"${DOCKER_BUILD_FLAGS[@]}"} ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
    -f docker/Dockerfile -t "$IMAGE" .
ok "Built $IMAGE"
docker run --rm --entrypoint cat "$IMAGE" /opt/recipe/BUILD_INFO | sed 's/^/        /'

if $VERIFY; then
    # Imports the compiled extensions, resolves exl3 through vLLM's
    # quantization registry and checks the three vLLM patches. This needs the
    # GPU driver, which is why it cannot run during docker build.
    info "Running preflight in the image (needs the GPU, a few seconds) ..."
    if docker run --rm --gpus all --entrypoint python3 "$IMAGE" /opt/recipe/preflight.py; then
        ok "Preflight passed."
    else
        err "Preflight failed inside $IMAGE; see the FAIL lines above."
    fi
fi

info ""
info "Next: ./download.sh   (fetch + prepare the pack, ~80 GB)"
info "then: ./start.sh"
