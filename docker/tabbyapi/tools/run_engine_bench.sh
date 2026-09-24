#!/usr/bin/env bash
# Run engine_bench.py in the qwen38-exl3-tabby image, on the Spark.
# TabbyAPI must be stopped first (./stop_tabby.sh): two copies of the model do
# not fit in memory. Extra arguments go to `docker run` (e.g. -e CHUNK=4096).
#
#   docker/tabbyapi/tools/run_engine_bench.sh baseline
#   docker/tabbyapi/tools/run_engine_bench.sh chunk4k -e CHUNK=4096
#   docker/tabbyapi/tools/run_engine_bench.sh prof -e CHUNK=8192 -e PROFILE=600 -e SIZES=
#
#   SCRIPT=moe_trace.py docker/tabbyapi/tools/run_engine_bench.sh trace
#
# SCRIPT picks another script from this directory (default engine_bench.py).
# Output: logs/bench_<name>.log in the repo root.
set -euo pipefail
name=${1:?usage: run_engine_bench.sh <name> [docker run args...]}; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
MOUNT=/models/qwen3.8-flash-next
ARGS=(-v "$MODEL_DIR:$MOUNT:ro")
for f in config.json model.safetensors.index.json; do
    if [[ -f "$MODEL_DIR/$f.native" ]]; then ARGS+=(-v "$MODEL_DIR/$f.native:$MOUNT/$f:ro"); fi
done
mkdir -p "$ROOT/logs"
bash "$ROOT/scripts/exl3_native/tuning/drop-model-cache.sh" "$MODEL_DIR" >/dev/null
# Share the server's kernel tuning cache (start_tabby.sh), so runs skip the tuning
[[ -n "${TABBY_CACHE_VOLUME-qwen38-tabby-cache}" ]] && ARGS+=(-v "${TABBY_CACHE_VOLUME-qwen38-tabby-cache}:/home/tabby/.cache")
docker run --rm --gpus all --cpuset-cpus 5-9,15-19 "${ARGS[@]}" \
    -v "$HERE/${SCRIPT:-engine_bench.py}:/tmp/bench.py:ro" \
    --entrypoint python3 "$@" qwen38-exl3-tabby:latest /tmp/bench.py \
    > "$ROOT/logs/bench_$name.log" 2>&1 || true
grep -E "CONFIG|loaded|RESULT|MEM|PROFILE|FLAGS|TIER|TIME|SCOPE|LAYER|DONE|Error|error" "$ROOT/logs/bench_$name.log" | tail -${TAIL:-20}
echo "full log: logs/bench_$name.log"
