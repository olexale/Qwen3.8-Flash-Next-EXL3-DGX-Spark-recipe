#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next EXL3 (3.05bpw) via ExLlamaV3 on the DGX Spark (GB10, aarch64).
# Usage: ~/run-qwen38-exl3.sh [extra chat.py args...]     e.g. -prompt "hello"   (none = interactive)
#        CS=32768 ~/run-qwen38-exl3.sh                    smaller cache for a quick session
#
# Configuration for GB10. Measured 2026-09-17, chat.py cold, greedy, 400 new tokens:
#   code 79 t/s  /  DevOps explainer 62 t/s  /  prose 53 t/s      (no draft: 33)
#   at 240k tokens of context: 72 t/s with 8-bit KV vs 65 fp16; prefill ~1,150 t/s flat to 480k
# What each part is worth is in the recipe README:
#   github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe  (native engine section)
set -euo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1
# int8-activation GEMV is slower than fp16 on GB10; force the wide 128-col MoE coop tile
export EXL3_INT8_GEMV=${EXL3_INT8_GEMV:-0} EXL3_MOE_COOP_WIDE=${EXL3_MOE_COOP_WIDE:-1}
# hyperconnection mixers stored int8 (they ship fp16 in the pack): -1.6 GB, half the mixer bytes/round.
# Default on in the fork since 523ecd3; set explicitly so an older checkout behaves the same.
export EXL3_GR_INT8=${EXL3_GR_INT8:-1}
# draft chain argmaxes over a 64K-column slice of lm_head (verify uses the full head)
export EXL3_MTP_HEAD_N=${EXL3_MTP_HEAD_N:-65536}
export EXL3_NGRAM_STREAM=${EXL3_NGRAM_STREAM:-0}
# 10 Cortex-X925 big cores; the launch thread on an A725 little core costs ~2 t/s
BIGCORES=5-9,15-19
MODEL="$HOME/models/Qwen3.8-Flash-Next-EXL3"
# Context. 262144 is the model's trained window and the real ceiling: needle retrieval fails past it
# (300k, 480k) at both KV precisions, and host memory would be the next wall at ~480k (118 of 121 GiB).
# KV is 24 KB/token fp16 on this geometry (12 of 48 layers full attention, 2 KV heads), 12 KB at 8-bit.
CS=${CS:-262144}
cd "$HOME/exllamav3"
# page cache is GPU-allocatable memory on GB10; drop the model's pages or autosplit refuses to load
"$HOME/drop-model-cache.sh" >/dev/null 2>&1 || true
# -mtp: the model's own MTP head as the drafter. -ndt 5 -dds -dc 0.6: up to 5 drafts, stop early
# when the running confidence falls below 0.6 (prose collapses past draft position 1; code does not).
# -cq 8,8: 8-bit KV. The full-attention layers stream the whole KV every step (5.9 GB/step at 240k
# fp16), so halving it is faster at every depth (+5% at 4k, +11% at 240k) with acceptance unchanged.
exec taskset -c $BIGCORES python examples/chat.py -m "$MODEL" -mode qwen35 -mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs "$CS" -tps "$@"
