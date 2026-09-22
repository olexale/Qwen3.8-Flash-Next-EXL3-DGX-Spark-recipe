#!/usr/bin/env bash
# ============================================================================
# download.sh — fetch turboderp's Qwen3.8-Flash-Next EXL3 pack into MODEL_DIR
# and prepare it for vllm-exl3 (scripts/prepare_pack.sh), both inside the
# image, so the host needs nothing but docker.
#
# The download is resumable: rerun after an interruption. Preparation is
# idempotent and keeps backups (config.json.native, *.index.json.native), so
# rerunning the whole script on a finished pack is safe.
#
# Usage:
#   ./download.sh                    # REVISION into MODEL_DIR from .env, then prepare
#   ./download.sh --prepare-only     # pack already downloaded (e.g. with `hf download`)
#   ./download.sh --skip-prepare     # download only
#   VERIFY=1 ./download.sh --prepare-only   # also run the plugin's GPU verify gates
#   REVISION=4.05bpw_h6_ng6 MODEL_DIR=~/models/Qwen3.8-Flash-Next-exl3-4.05bpw ./download.sh
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_common.sh"
cd "$RECIPE_DIR"

DO_DOWNLOAD=true
DO_PREPARE=true
for arg in "$@"; do
    case "$arg" in
        --prepare-only) DO_DOWNLOAD=false ;;
        --skip-prepare) DO_PREPARE=false ;;
        -h|--help)      sed -n '3,/^# =*$/p' "$0" | sed '$d;s/^# \?//'; exit 0 ;;
        *)              err "unknown option: $arg (try --help)" ;;
    esac
done

load_env
common_defaults
HF_TOKEN="${HF_TOKEN:-}"
VERIFY="${VERIFY:-0}"
require_docker
require_image

mkdir -p "$MODEL_DIR"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"

# Run as the invoking user so the pack stays owned by you, not root.
RUN_AS=(--user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf)

if $DO_DOWNLOAD; then
    info "Downloading $HF_REPO@$REVISION into $MODEL_DIR (~80 GB for 3.05 bpw) ..."
    export HF_TOKEN
    docker run --rm "${RUN_AS[@]}" \
        ${HF_TOKEN:+-e HF_TOKEN} \
        -e REPO="$HF_REPO" -e REVISION="$REVISION" \
        -v "$MODEL_DIR:/model" \
        --entrypoint python3 "$IMAGE" -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["REPO"], revision=os.environ["REVISION"], local_dir="/model")
'
    ok "Download complete."
fi

if $DO_PREPARE; then
    [[ -f "$MODEL_DIR/config.json" ]] || err "$MODEL_DIR/config.json missing; download the pack first."
    info "Preparing the pack for vllm-exl3 (scan, config rewrite, index regeneration) ..."
    GPU_ARGS=()
    [[ "$VERIFY" == "1" ]] && GPU_ARGS=(--gpus all)
    docker run --rm "${RUN_AS[@]}" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
        -e PACK_DIR=/model -e VERIFY="$VERIFY" \
        -v "$MODEL_DIR:/model" \
        --entrypoint bash "$IMAGE" /opt/recipe/prepare_pack.sh
    ok "Pack prepared: $MODEL_DIR"
fi

info ""
info "Next: ./start.sh"
