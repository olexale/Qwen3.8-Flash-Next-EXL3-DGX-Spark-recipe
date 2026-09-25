#!/usr/bin/env bash
# ============================================================================
# start_tabby.sh — serve the EXL3 pack through TabbyAPI on the tuned
# exllamav3 fork (docker/tabbyapi), then follow the log until /health answers.
#
# Reads .env like start_vllm.sh. MODEL_DIR, BIND, PORT, SERVED_NAME and
# REQUIRE_IDLE_GPU are shared with it, so clients keep the same address and
# model name whichever server runs (only one fits in memory). Settings only
# this script uses start with TABBY_. Engine settings (context, cache, draft,
# chunk size, vision, sampling) are in docker/tabbyapi/config.yml, baked into
# the image; TABBY_CONFIG mounts a different one without rebuilding.
# Kernel tuning results persist in the Docker volume TABBY_CACHE_VOLUME, so
# only the very first start pays for tuning.
#
# The container restarts by itself after a crash or a reboot
# (TABBY_RESTART=unless-stopped) until ./stop_tabby.sh stops it.
#
# Usage:
#   ./start_tabby.sh                  # profile from .env
#   ./start_tabby.sh --build          # build the image first (docker/tabbyapi)
#   ./start_tabby.sh --no-launch      # print the docker command, don't start
#   PORT=5000 ./start_tabby.sh        # one-off override of any setting
#   EXL3_DRAFT_CONFIDENCE=0.5 ./start_tabby.sh
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_common.sh"
cd "$RECIPE_DIR"

DO_LAUNCH=true
DO_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=false ;;
        --build)     DO_BUILD=true ;;
        -h|--help)   sed -n '3,/^# =*$/p' "$0" | sed '$d;s/^# \?//'; exit 0 ;;
        *)           err "unknown option: $arg (try --help)" ;;
    esac
done

load_env
common_defaults
BIND="${BIND:-0.0.0.0}"
PORT="${PORT:-18300}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
TABBY_IMAGE="${TABBY_IMAGE:-qwen38-exl3-tabby:latest}"
TABBY_CONTAINER="${TABBY_CONTAINER:-qwen38-tabby}"
TABBY_CPUSET="${TABBY_CPUSET:-5-9,15-19}"
TABBY_RESTART="${TABBY_RESTART:-unless-stopped}"
TABBY_CONFIG="${TABBY_CONFIG:-}"
TABBY_CACHE_VOLUME="${TABBY_CACHE_VOLUME-qwen38-tabby-cache}"
TABBY_READY_TIMEOUT_S="${TABBY_READY_TIMEOUT_S:-600}"
TABBY_EXTRA_DOCKER_ARGS="${TABBY_EXTRA_DOCKER_ARGS:-}"

require_docker

if $DO_BUILD; then
    info "Building $TABBY_IMAGE from docker/tabbyapi ..."
    docker build -t "$TABBY_IMAGE" docker/tabbyapi
fi
docker image inspect "$TABBY_IMAGE" >/dev/null 2>&1 \
    || err "image $TABBY_IMAGE not found. Build it: ./start_tabby.sh --build"

# ---------------------------------------------------------------------------
# Pre-flight on the host
# ---------------------------------------------------------------------------
[[ -d "$MODEL_DIR" ]] || err "MODEL_DIR=$MODEL_DIR does not exist. Download the pack first (docker/tabbyapi/README.md)."
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
[[ -f "$MODEL_DIR/config.json" ]] || err "$MODEL_DIR/config.json missing."
[[ "$SERVED_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || err "SERVED_NAME=$SERVED_NAME: use letters, digits, '.', '_' and '-' only (it becomes a folder name)."
if [[ -n "$TABBY_CONFIG" ]]; then
    [[ -f "$TABBY_CONFIG" ]] || err "TABBY_CONFIG=$TABBY_CONFIG does not exist."
    TABBY_CONFIG="$(cd "$(dirname "$TABBY_CONFIG")" && pwd)/$(basename "$TABBY_CONFIG")"
fi

if $DO_LAUNCH && docker ps -q -f "name=^${TABBY_CONTAINER}\$" | grep -q .; then
    err "$TABBY_CONTAINER is already running. ./stop_tabby.sh first."
fi

if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    TENANTS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
              --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
    if [[ -n "$TENANTS" ]]; then
        echo "$TENANTS"
        err "GPU is in use by the processes above (./stop_vllm.sh?). Stop them, or set REQUIRE_IDLE_GPU=false."
    fi
fi

if [[ -n "${API_KEY:-}" ]]; then
    warn "API_KEY is set, but TabbyAPI runs with auth off (disable_auth in docker/tabbyapi/config.yml); it is ignored."
fi
if [[ "$BIND" != "127.0.0.1" && "$BIND" != "::1" && "$BIND" != "localhost" ]]; then
    warn "BIND=$BIND: the API has no key and is open to the network."
fi

# ---------------------------------------------------------------------------
# docker run
# ---------------------------------------------------------------------------
MOUNT="/models/$SERVED_NAME"
DOCKER_ARGS=(
    run -d --name "$TABBY_CONTAINER"
    --gpus all
    --restart "$TABBY_RESTART"
    --security-opt no-new-privileges
    --log-opt max-size=50m --log-opt max-file=3
    -p "$BIND:$PORT:5000"
    -v "$MODEL_DIR:$MOUNT:ro"
)
[[ -n "$TABBY_CPUSET" ]] && DOCKER_ARGS+=(--cpuset-cpus "$TABBY_CPUSET")
# A pack prepared for vLLM (prepare_pack.sh) has a rewritten config.json and
# index; serve exllamav3 the originals it keeps as *.native, as
# scripts/exl3_native/make_native_view.sh does. The pack itself is unchanged.
for f in config.json model.safetensors.index.json; do
    if [[ -f "$MODEL_DIR/$f.native" ]]; then
        DOCKER_ARGS+=(-v "$MODEL_DIR/$f.native:$MOUNT/$f:ro")
    fi
done
[[ -n "$TABBY_CONFIG" ]] && DOCKER_ARGS+=(-v "$TABBY_CONFIG:/app/config.yml:ro")
# Kernel tuning results (exllamav3 GEMM autotune, Triton) kept across restarts
[[ -n "$TABBY_CACHE_VOLUME" ]] && DOCKER_ARGS+=(-v "$TABBY_CACHE_VOLUME:/home/tabby/.cache")
# GB10 engine knobs: forwarded only when set, so the image's defaults apply otherwise.
for v in EXL3_DRAFT_CONFIDENCE EXL3_GR_INT8 EXL3_MOE_COOP_WIDE EXL3_INT8_GEMV \
         EXL3_MTP_HEAD_N EXL3_NGRAM_STREAM EXL3_MOE_FUSED_UNIFORM EXL3_GR_COLLAPSE EXL3_QSA_STAGE \
         EXL3_GDN_NOCOPY TABBY_ENCODE_CACHE EXL3_PLD EXL3_PLD_START EXL3_PLD_MAX EXL3_PLD_MIN_MATCH \
         EXL3_MOE_BSZN_MAX EXL3_PREFIX_DIAG \
         EXL3_HIST_STASH; do
    if [[ -n "${!v:-}" ]]; then DOCKER_ARGS+=(-e "$v=${!v}"); fi
done
read -ra _extra_docker <<<"$TABBY_EXTRA_DOCKER_ARGS"
DOCKER_ARGS+=(${_extra_docker[@]+"${_extra_docker[@]}"} "$TABBY_IMAGE" --model-name "$SERVED_NAME")

info "Image:   $TABBY_IMAGE"
info "Model:   $MODEL_DIR (served as $SERVED_NAME)"
info "Serving: http://$BIND:$PORT/v1  (container $TABBY_CONTAINER, restart=$TABBY_RESTART)"
if ! $DO_LAUNCH; then
    printf 'docker'; printf ' %q' "${DOCKER_ARGS[@]}"; echo
    exit 0
fi

docker rm -f "$TABBY_CONTAINER" >/dev/null 2>&1 || true
docker "${DOCKER_ARGS[@]}" >/dev/null
ok "Container $TABBY_CONTAINER started."
info "Loading (about 1 minute). Following logs until ready..."

# ---------------------------------------------------------------------------
# Wait for /health
# ---------------------------------------------------------------------------
HEALTH_HOST="$BIND"
[[ "$BIND" == "0.0.0.0" || "$BIND" == "::" ]] && HEALTH_HOST=127.0.0.1
[[ "$HEALTH_HOST" == *:* ]] && HEALTH_HOST="[$HEALTH_HOST]"

docker logs -f "$TABBY_CONTAINER" &
LOGPID=$!
trap 'kill $LOGPID 2>/dev/null || true' EXIT
START=$(date +%s)
while true; do
    sleep 5
    ELAPSED=$(( $(date +%s) - START ))
    # With a restart policy a failed load shows up as restarts, not an exit
    RESTARTS=$(docker inspect "$TABBY_CONTAINER" --format '{{.RestartCount}}' 2>/dev/null || echo gone)
    if [[ "$RESTARTS" == "gone" ]] || [[ "$RESTARTS" -gt 0 ]] \
       || ! docker ps -q -f "name=^${TABBY_CONTAINER}\$" | grep -q .; then
        kill $LOGPID 2>/dev/null || true
        echo
        if docker inspect "$TABBY_CONTAINER" --format '{{.State.OOMKilled}}' 2>/dev/null | grep -q true; then
            warn "The container was OOM-killed."
        fi
        err "TabbyAPI failed to start (restarts: $RESTARTS). Docker keeps retrying until you run ./stop_tabby.sh.
       Full log: docker logs $TABBY_CONTAINER. See docker/tabbyapi/README.md \"Troubleshooting\"."
    fi
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://$HEALTH_HOST:$PORT/health" 2>/dev/null || true)
    if [[ "$CODE" == "200" ]]; then
        kill $LOGPID 2>/dev/null || true
        echo
        ok "TabbyAPI ready on port $PORT after ${ELAPSED}s."
        info "Test:  curl http://$HEALTH_HOST:$PORT/v1/models"
        info "Logs:  docker logs -f $TABBY_CONTAINER"
        info "Stop:  ./stop_tabby.sh"
        exit 0
    fi
    if (( ELAPSED > TABBY_READY_TIMEOUT_S )); then
        kill $LOGPID 2>/dev/null || true
        echo
        err "Not ready after ${ELAPSED}s (TABBY_READY_TIMEOUT_S=$TABBY_READY_TIMEOUT_S). The container is still running;
       inspect with: docker logs $TABBY_CONTAINER   stop with: ./stop_tabby.sh"
    fi
done
