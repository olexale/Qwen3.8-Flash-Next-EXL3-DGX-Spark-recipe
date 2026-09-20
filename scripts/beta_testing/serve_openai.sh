#!/usr/bin/env bash
# OpenAI /v1 over the native GB10 launcher (same env as tuning/run-qwen38-exl3.sh).
# Default port 8899, same as scripts/serve_one_spark_qwen.sh.
# One Generator job at a time — this is chat.py's engine, not vLLM C4.
#
# Measured 2026-09-20 on one GB10, vcruz305/exllamav3 785f206:
#   greedy 400-token code job: 79.5 wall tok/s, 83.8 engine, 74% draft accept
#   (matches the README chat.py 79 / 73%). Nous Hermes tool loop at ~80k
#   prompt survived; the vLLM overlay on this pack died on the 2nd generate
#   (CUBLAS mm 10240x336).
set -euo pipefail
export PATH="${EXL3_ROOT:-$HOME/exllamav3}/.venv/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
export EXL3_INT8_GEMV="${EXL3_INT8_GEMV:-0}"
export EXL3_MOE_COOP_WIDE="${EXL3_MOE_COOP_WIDE:-1}"
export EXL3_GR_INT8="${EXL3_GR_INT8:-1}"
export EXL3_MTP_HEAD_N="${EXL3_MTP_HEAD_N:-65536}"
export EXL3_NGRAM_STREAM="${EXL3_NGRAM_STREAM:-0}"
BIGCORES="${BIGCORES:-5-9,15-19}"
CS="${CS:-262144}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8899}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next-EXL3}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXL3_ROOT="${EXL3_ROOT:-$HOME/exllamav3}"
for cand in \
  "${MODEL_DIR:-}" \
  "$HOME/models/Qwen3.8-Flash-Next-EXL3-native" \
  "$HOME/models/Qwen3.8-Flash-Next-EXL3"
do
  [[ -n "$cand" && -f "$cand/config.json" ]] && { MODEL="$cand"; break; }
done
MODEL="${MODEL:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
[[ -f "$MODEL/config.json" ]] || { echo "missing $MODEL/config.json" >&2; exit 1; }
# prepare_pack.sh rewrites config for vLLM; native wants the original view.
if grep -q 'derived_from_headers' "$MODEL/config.json" 2>/dev/null && [[ ! -e "$MODEL/config.json.native" ]]; then
  echo "this looks like a vLLM-prepared pack without a native view." >&2
  echo "run: bash scripts/exl3_native/make_native_view.sh $MODEL \$HOME/models/Qwen3.8-Flash-Next-EXL3-native" >&2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
DROP="$HERE/tuning/drop-model-cache.sh"
[[ -x "$DROP" ]] && "$DROP" >/dev/null 2>&1 || true
export EXL3_ROOT CS SERVED_NAME
cd "$EXL3_ROOT"
exec taskset -c "$BIGCORES" python "$HERE/serve_openai.py" \
  -m "$MODEL" -mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs "$CS" -topk 1 \
  --host "$HOST" --port "$PORT" "$@"
