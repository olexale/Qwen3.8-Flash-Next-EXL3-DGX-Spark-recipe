#!/usr/bin/env bash
# ============================================================================
# stop_vllm.sh — stop and remove the serving container, keeping its log.
#
# Graceful by default: vLLM gets SIGTERM and STOP_TIMEOUT seconds (default 30)
# to release its shared-memory segments, which matter because the container
# runs with --ipc host. The log is saved to logs/ before the container is
# removed.
#
# Usage:
#   ./stop_vllm.sh            # SIGTERM, then SIGKILL after STOP_TIMEOUT
#   ./stop_vllm.sh --force    # SIGKILL now
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_common.sh"
cd "$RECIPE_DIR"

FORCE=false
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=true ;;
        -h|--help)  sed -n '3,/^# =*$/p' "$0" | sed '$d;s/^# \?//'; exit 0 ;;
        *)          err "unknown option: $arg (try --help)" ;;
    esac
done

load_env
common_defaults
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || err "STOP_TIMEOUT must be a non-negative integer (got: '$STOP_TIMEOUT')"
require_docker

if [[ -z "$(docker ps -aq -f "name=^${CONTAINER_NAME}\$")" ]]; then
    info "$CONTAINER_NAME is not running."
    exit 0
fi

mkdir -p logs
LOG="logs/${CONTAINER_NAME}-$(date '+%Y%m%dT%H%M%S').log"
docker logs "$CONTAINER_NAME" >"$LOG" 2>&1 || true
info "Log saved to $LOG"

if ! $FORCE; then
    info "Stopping $CONTAINER_NAME (SIGTERM, up to ${STOP_TIMEOUT}s) ..."
    docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null
fi
docker rm -f "$CONTAINER_NAME" >/dev/null
ok "Stopped $CONTAINER_NAME."
