# shellcheck shell=bash
# Shared by build.sh, download.sh, start.sh and stop.sh. Source, don't run.

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*" >&2; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*" >&2; exit 1; }

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source .env with the precedence environment > .env > built-in default:
# anything already set in the environment survives the source, so
# `PORT=9000 ./start.sh` works without editing .env.
load_env() {
    local f="$RECIPE_DIR/.env" n i
    local names=() vals=()
    if [[ ! -f "$f" ]]; then
        if [[ "${1:-}" == "--required" ]]; then
            err ".env not found. Copy .env.sample to .env and edit it."
        fi
        return 0
    fi
    while IFS= read -r n; do
        if [[ -n "${!n+x}" ]]; then names+=("$n"); vals+=("${!n}"); fi
    done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$f" | sort -u)
    # shellcheck source=/dev/null
    source "$f"
    for (( i = 0; i < ${#names[@]}; i++ )); do
        printf -v "${names[$i]}" '%s' "${vals[$i]}"
    done
}

# Defaults shared by every script. Call after load_env.
common_defaults() {
    IMAGE="${IMAGE:-qwen38-flash-next-exl3-vllm:latest}"
    CONTAINER_NAME="${CONTAINER_NAME:-qwen38-exl3-vllm}"
    HF_REPO="${HF_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
    REVISION="${REVISION:-3.05bpw_h5_ng5}"
    MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
    CACHE_DIR="${CACHE_DIR:-$HOME/.cache/qwen38-exl3-vllm}"
}

require_docker() {
    command -v docker >/dev/null 2>&1 || err "docker not found in PATH."
    docker info >/dev/null 2>&1 || err "cannot talk to the docker daemon (is it running, and are you in the docker group?)."
}

require_image() {
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || err "image $IMAGE not found. Build it first: ./build.sh"
}
