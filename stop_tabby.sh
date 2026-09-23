#!/usr/bin/env bash
# ============================================================================
# stop_tabby.sh — stop and remove the TabbyAPI container, keeping its log.
#
# Stopping it this way also ends its automatic restarts. TabbyAPI gets
# SIGTERM and STOP_TIMEOUT seconds (default 30) to unload the model; the log
# is saved to logs/ before the container is removed.
#
# Usage:
#   ./stop_tabby.sh            # SIGTERM, then SIGKILL after STOP_TIMEOUT
#   ./stop_tabby.sh --force    # SIGKILL now
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
TABBY_CONTAINER="${TABBY_CONTAINER:-qwen38-tabby}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || err "STOP_TIMEOUT must be a non-negative integer (got: '$STOP_TIMEOUT')"
require_docker

if [[ -z "$(docker ps -aq -f "name=^${TABBY_CONTAINER}\$")" ]]; then
    info "$TABBY_CONTAINER is not running."
    exit 0
fi

mkdir -p logs
LOG="logs/${TABBY_CONTAINER}-$(date '+%Y%m%dT%H%M%S').log"
docker logs "$TABBY_CONTAINER" >"$LOG" 2>&1 || true
info "Log saved to $LOG"

if ! $FORCE; then
    info "Stopping $TABBY_CONTAINER (SIGTERM, up to ${STOP_TIMEOUT}s) ..."
    docker stop -t "$STOP_TIMEOUT" "$TABBY_CONTAINER" >/dev/null
fi
docker rm -f "$TABBY_CONTAINER" >/dev/null
ok "Stopped $TABBY_CONTAINER."
