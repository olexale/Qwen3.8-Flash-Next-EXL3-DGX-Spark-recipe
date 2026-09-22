#!/usr/bin/env bash
# ============================================================================
# start.sh — serve the prepared EXL3 pack from the patched image on one
# DGX Spark, then follow the log until /health answers.
#
# The container runs scripts/serve_one_spark_qwen.sh (the image entrypoint)
# with the serving knobs from .env passed through as environment variables,
# so the defaults and their reasoning live in one place: that script.
#
# Usage:
#   ./start.sh                              # profile from .env
#   ./start.sh --no-launch                  # print the docker command, don't start
#   SPEC_CONFIG=none ./start.sh             # no MTP draft (prompts past ~163k tokens)
#   NGRAM_TABLE=disk MODEL_DIR=~/models/Qwen3.8-Flash-Next-exl3-4.05bpw ./start.sh
#   EXTRA_VLLM_ARGS="--max-num-batched-tokens 4096" ./start.sh
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_common.sh"
cd "$RECIPE_DIR"

DO_LAUNCH=true
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=false ;;
        -h|--help)   sed -n '3,/^# =*$/p' "$0" | sed '$d;s/^# \?//'; exit 0 ;;
        *)           err "unknown option: $arg (try --help)" ;;
    esac
done

load_env --required
common_defaults
BIND="${BIND:-127.0.0.1}"
PORT="${PORT:-8899}"
API_KEY="${API_KEY:-}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

require_docker
require_image

# ---------------------------------------------------------------------------
# Pre-flight on the host: fail in seconds, not after a ten-minute load.
# ---------------------------------------------------------------------------
[[ -d "$MODEL_DIR" ]] || err "MODEL_DIR=$MODEL_DIR does not exist. Run ./download.sh"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
[[ -f "$MODEL_DIR/config.json" ]] || err "$MODEL_DIR/config.json missing. Run ./download.sh"
if ! grep -q '"quant_method"[[:space:]]*:[[:space:]]*"exl3"' "$MODEL_DIR/config.json" \
   || ! grep -q 'ngram_embedding' "$MODEL_DIR/model.safetensors.index.json" 2>/dev/null; then
    err "$MODEL_DIR is downloaded but not prepared for vllm-exl3. Run ./download.sh --prepare-only"
fi

if [[ -n "${GPU_MEM_UTIL:-}" ]] && awk -v u="$GPU_MEM_UTIL" 'BEGIN{exit !(u > 0.85)}'; then
    err "GPU_MEM_UTIL=$GPU_MEM_UTIL is above the measured-safe ceiling (0.85) for this box."
fi

if docker ps -q -f "name=^${CONTAINER_NAME}\$" | grep -q .; then
    err "$CONTAINER_NAME is already running. ./stop.sh first."
fi

if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    TENANTS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
              --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
    if [[ -n "$TENANTS" ]]; then
        echo "$TENANTS"
        err "GPU is in use by the processes above. Stop them, or set REQUIRE_IDLE_GPU=false."
    fi
fi

if [[ "$BIND" != "127.0.0.1" && "$BIND" != "::1" && "$BIND" != "localhost" && -z "$API_KEY" ]]; then
    warn "BIND=$BIND is not loopback and API_KEY is empty: the API is open to the network."
fi

# ---------------------------------------------------------------------------
# docker run
# ---------------------------------------------------------------------------
mkdir -p "$CACHE_DIR"
DOCKER_ARGS=(
    run -d --name "$CONTAINER_NAME"
    --gpus all --network host --ipc host
    --ulimit memlock=-1 --ulimit stack=67108864
    --log-opt max-size=50m --log-opt max-file=3
    -v "$MODEL_DIR:/model"
    -v "$CACHE_DIR:/root/.cache"
    -e MODEL_DIR=/model
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e TRITON_CACHE_DIR=/root/.cache/triton
    -e HOST="$BIND" -e PORT="$PORT"
)
# Serving knobs: forwarded only when set, so the serve script's defaults apply
# otherwise.
for v in SERVED_NAME MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS SPEC_CONFIG MAMBA_SSM_DTYPE \
         NGRAM_TABLE VLLM_EXL3_NGRAM_KERNEL TOOL_CALL_PARSER PROFILER_DIR \
         PYTORCH_CUDA_ALLOC_CONF; do
    if [[ -n "${!v:-}" ]]; then DOCKER_ARGS+=(-e "$v=${!v}"); fi
done
# The key is passed by name, so its value never appears in the command line.
if [[ -n "$API_KEY" ]]; then
    export VLLM_API_KEY="$API_KEY"
    DOCKER_ARGS+=(-e VLLM_API_KEY)
fi
read -ra _extra_docker <<<"$EXTRA_DOCKER_ARGS"
read -ra _extra_vllm <<<"$EXTRA_VLLM_ARGS"
DOCKER_ARGS+=(${_extra_docker[@]+"${_extra_docker[@]}"} "$IMAGE" ${_extra_vllm[@]+"${_extra_vllm[@]}"})

info "Image:   $IMAGE"
info "Model:   $MODEL_DIR"
info "Serving: http://$BIND:$PORT/v1  (container $CONTAINER_NAME)"
if ! $DO_LAUNCH; then
    printf 'docker'; printf ' %q' "${DOCKER_ARGS[@]}"; echo
    exit 0
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker "${DOCKER_ARGS[@]}" >/dev/null
ok "Container $CONTAINER_NAME started."
info "Loading (~9.5 min from cold NVMe, ~2.5 min from page cache). Following logs until ready..."

# ---------------------------------------------------------------------------
# Wait for /health
# ---------------------------------------------------------------------------
HEALTH_HOST="$BIND"
[[ "$BIND" == "0.0.0.0" || "$BIND" == "::" ]] && HEALTH_HOST=127.0.0.1
[[ "$HEALTH_HOST" == *:* ]] && HEALTH_HOST="[$HEALTH_HOST]"

docker logs -f "$CONTAINER_NAME" &
LOGPID=$!
trap 'kill $LOGPID 2>/dev/null || true' EXIT
START=$(date +%s)
while true; do
    sleep 10
    ELAPSED=$(( $(date +%s) - START ))
    if ! docker ps -q -f "name=^${CONTAINER_NAME}\$" | grep -q .; then
        kill $LOGPID 2>/dev/null || true
        echo
        if docker inspect "$CONTAINER_NAME" --format '{{.State.OOMKilled}}' 2>/dev/null | grep -q true; then
            warn "The container was OOM-killed."
        fi
        err "Container exited after ${ELAPSED}s. See README \"Troubleshooting\"; full log: docker logs $CONTAINER_NAME"
    fi
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://$HEALTH_HOST:$PORT/health" 2>/dev/null || true)
    if [[ "$CODE" == "200" ]]; then
        kill $LOGPID 2>/dev/null || true
        echo
        ok "vLLM ready on port $PORT after ${ELAPSED}s."
        # KV pool as vLLM sized it. "Maximum concurrency" is how many requests
        # of MAX_MODEL_LEN fit at once; what to read when tuning GPU_MEM_UTIL.
        docker logs "$CONTAINER_NAME" 2>&1 \
            | grep -iE "GPU KV cache size|Available KV cache|Maximum concurrency" | tail -3 || true
        info "Test:  curl http://$HEALTH_HOST:$PORT/v1/models${API_KEY:+ -H 'Authorization: Bearer \$API_KEY'}"
        info "Stop:  ./stop.sh"
        exit 0
    fi
    if (( ELAPSED > READY_TIMEOUT_S )); then
        kill $LOGPID 2>/dev/null || true
        echo
        err "Not ready after ${ELAPSED}s (READY_TIMEOUT_S=$READY_TIMEOUT_S). The container is still running;
       inspect with: docker logs $CONTAINER_NAME   stop with: ./stop.sh"
    fi
done
