# syntax=docker/dockerfile:1
# ============================================================================
# TabbyAPI (OpenAI-compatible, vision) on the tuned exllamav3 fork, for
# Qwen3.8-Flash-Next EXL3 on one DGX Spark (GB10, sm_121, aarch64).
#
# TabbyAPI's published images are amd64-only and its cu12/cu13 extras only
# resolve torch/exllamav3 on x86_64, so on the Spark everything is built here:
#   1. torch 2.13.0 from PyPI (the aarch64 wheel is a CUDA 13 build)
#   2. exllamav3 from vcruz305/exllamav3 329e051 (upstream + aarch64 guards +
#      the GB10 decode changes), extension compiled ahead of time for sm_121
#   3. TabbyAPI base dependencies, plus a one-line patch that passes the
#      fork's draft_confidence (chat.py's -dc) through to the Generator
#   4. a patch that keeps exllamav3's recurrent-state checkpoints on the
#      device (patch_exllamav3_checkpoints.py), and one that re-enables the
#      fused MoE prefill kernel (patch_exllamav3_fused_moe.py)
#
# Build on the Spark (the build needs no GPU), from the repo root:
#   ./start_tabby.sh --build      (or: docker build -t qwen38-exl3-tabby docker/tabbyapi)
# ============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.13.0
ARG EXLLAMAV3_REPO=https://github.com/vcruz305/exllamav3
ARG EXLLAMAV3_REF=329e051385505b6ba981138d86a90bffe032c831
ARG TABBYAPI_REPO=https://github.com/theroyallab/tabbyAPI
# Last commit before TabbyAPI required exllamav3 >= 1.5.1 (f07131c, which only
# adds optional warmup). The fork reports 1.5.0 and predates 1.5.1.
ARG TABBYAPI_REF=2186cdb6c4660c9464ddd7b27c47b8b63c942def
# GB10 is sm_121.
ARG TORCH_CUDA_ARCH_LIST=12.1
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# python3-dev: Python.h for the extension build and for Triton's runtime JIT.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        git curl ca-certificates build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# ---------------------------------------------------------------------------
# 1. torch. Fail the build rather than the model load if it is not CUDA 13.
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel ninja \
 && pip install "torch==${TORCH_VERSION}" \
 && python3 -c 'import torch, sys; c = torch.version.cuda or ""; print("torch", torch.__version__, "CUDA", c); sys.exit(0 if c.startswith("13.") else "expected a CUDA 13 torch build, got CUDA " + repr(c))'

# ---------------------------------------------------------------------------
# 2. exllamav3 fork, extension precompiled. The fork already carries the
#    aarch64 guards. Two source patches before the build: a fused kernel for the
#    hyper-connection stream collapse in prefill (patch_exllamav3_gr_collapse.py,
#    bit-identical to the torch code it replaces; EXL3_GR_COLLAPSE=0 turns it
#    off), and room for 16-row batches in the fused decode MoE kernels
#    (patch_exllamav3_bszn16.py; EXL3_MOE_BSZN_MAX=16 in the image, 8 restores
#    the old limit bit-identically). --no-build-isolation so setup.py sees the torch above; without it the
#    extension is silently skipped.
# ---------------------------------------------------------------------------
COPY patch_exllamav3_gr_collapse.py patch_exllamav3_bszn16.py /tmp/
RUN git clone "${EXLLAMAV3_REPO}" /opt/src/exllamav3 \
 && git -C /opt/src/exllamav3 checkout --quiet "${EXLLAMAV3_REF}" \
 && python3 /tmp/patch_exllamav3_gr_collapse.py /opt/src/exllamav3 \
 && python3 /tmp/patch_exllamav3_bszn16.py /opt/src/exllamav3 \
 && cd /opt/src/exllamav3 \
 && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        pip install --no-build-isolation . \
 && python3 -c 'import importlib.util as u, sys; sys.exit(u.find_spec("exllamav3_ext") is None and "exllamav3_ext was not built")' \
 && python3 -c 'import torch; import importlib.metadata as m; print("torch still", torch.__version__, "exllamav3", m.version("exllamav3"))'

# ---------------------------------------------------------------------------
# 3. TabbyAPI. Base dependencies only: the cu12/cu13 extras would try to
#    install x86 torch/exllamav3 wheels. torch is pinned by constraint so the
#    resolver cannot swap it. uvloop is listed for x86_64 only, but main.py
#    imports it on every non-Windows host; it has aarch64 wheels. The --help
#    run catches startup import errors at build time, without a GPU or model.
# ---------------------------------------------------------------------------
RUN git clone "${TABBYAPI_REPO}" /app \
 && git -C /app checkout --quiet "${TABBYAPI_REF}" \
 && pip freeze | grep -E '^torch(==| @)' > /tmp/torch-constraint.txt \
 && pip install -c /tmp/torch-constraint.txt /app uvloop \
 && rm /tmp/torch-constraint.txt \
 && cd /app && python3 main.py --help > /dev/null

# TabbyAPI exposes dynamic_draft (chat.py's -dds) but not the confidence
# target (-dc); the fork's Generator defaults it to 0.4, the launcher uses
# 0.6. Pass it from EXL3_DRAFT_CONFIDENCE. Fails the build if the anchor moved.
RUN f=/app/backends/exllamav3/model.py \
 && grep -q '^                dynamic_draft_tokens=self.dynamic_draft,$' "$f" \
 && sed -i 's/^                dynamic_draft_tokens=self.dynamic_draft,$/&\n                draft_confidence=float(__import__("os").environ.get("EXL3_DRAFT_CONFIDENCE", "0.6")),/' "$f" \
 && grep -q 'EXL3_DRAFT_CONFIDENCE' "$f"

# Follow-up turns tokenize only the part of the prompt after the shared
# conversation prefix (~0.16 s saved per turn at 115k tokens; token ids
# identical). TABBY_ENCODE_CACHE=0 turns it off.
COPY patch_tabbyapi_encode_cache.py /tmp/
RUN python3 /tmp/patch_tabbyapi_encode_cache.py /app && rm /tmp/patch_tabbyapi_encode_cache.py

# Tool-call arguments as the model wrote them: strip only the template's newline
# on each side of a value and convert types only where the request's tool schema
# asks for it (the stock parser stripped all whitespace and json.loads'ed every
# value: JSON files arrived as objects, files lost their final newline, edits
# lost first-line indentation). TABBY_TOOLCALL_ARGS=0 restores the old parsing.
COPY patch_tabbyapi_toolcall_args.py /tmp/
RUN python3 /tmp/patch_tabbyapi_toolcall_args.py /app && rm /tmp/patch_tabbyapi_toolcall_args.py

# Quoted tool markers stay text: <tool_call> opens a call only when <function=
# follows and it is outside a fenced code block, a stray </tool_call> is text,
# requests without tools never enter tool mode, finish_reason "tool_calls" only
# when a call parsed. Plus reasoning_effort high/max -> xhigh, minimal -> low
# (the template 400s on them). TABBY_TOOL_MARKERS=0 / TABBY_EFFORT_ALIAS=0 turn
# them off.
COPY patch_tabbyapi_tool_markers.py /tmp/
RUN python3 /tmp/patch_tabbyapi_tool_markers.py /app && rm /tmp/patch_tabbyapi_tool_markers.py

# Recurrent-state checkpoints stay on the device instead of blocking copies to
# pageable host memory; see the script. Fails the build if an anchor moved.
COPY patch_exllamav3_checkpoints.py /tmp/
RUN python3 /tmp/patch_exllamav3_checkpoints.py && rm /tmp/patch_exllamav3_checkpoints.py

# The fused multi-expert MoE kernel, which the fork switches off for packs like
# this one (all experts share one K); see the script. Short prompts prefill
# ~2x faster and three concurrent sessions decode ~2x faster.
# EXL3_MOE_FUSED_UNIFORM=0 turns it off again.
COPY patch_exllamav3_fused_moe.py /tmp/
RUN python3 /tmp/patch_exllamav3_fused_moe.py && rm /tmp/patch_exllamav3_fused_moe.py

# Sparse (QSA) attention in prefill dequantizes the 8-bit K/V once per layer
# instead of once per query row; bit-identical, the attention kernel ~2x faster.
# EXL3_QSA_STAGE=0 turns it off again.
COPY patch_exllamav3_qsa_stage.py /tmp/
RUN python3 /tmp/patch_exllamav3_qsa_stage.py && rm /tmp/patch_exllamav3_qsa_stage.py

# GatedDeltaNet chunked prefill without the four full-tensor copies around the
# FLA kernels; bit-identical. Must run after patch_exllamav3_checkpoints.py.
# EXL3_GDN_NOCOPY=0 turns it off again.
COPY patch_exllamav3_gdn_nocopy.py /tmp/
RUN python3 /tmp/patch_exllamav3_gdn_nocopy.py && rm /tmp/patch_exllamav3_gdn_nocopy.py

# Prompt-lookup drafting alongside the MTP head: when the output repeats text in
# the context (edit tool calls, file rewrites), draft the continuation of the
# match instead of running the MTP chain. Verification is unchanged, so outputs
# are unchanged. On (EXL3_PLD=1, drafts of up to 11 tokens, EXL3_PLD_MAX); see
# the script. EXL3_PLD=0 turns it off.
COPY patch_exllamav3_pld.py /tmp/
RUN python3 /tmp/patch_exllamav3_pld.py && rm /tmp/patch_exllamav3_pld.py

# Prefix-reuse diagnostic: one log line per request on how much of the cached
# prefix was resumed and why the rest was prefilled again (lengths only, no
# text). Log-only, on (EXL3_PREFIX_DIAG=1, since 2026-09-26: it also reports whether anchor
# checkpoints get reused); 2 adds divergence details, 0 turns it off.
COPY patch_exllamav3_prefix_diag.py /tmp/
RUN python3 /tmp/patch_exllamav3_prefix_diag.py && rm /tmp/patch_exllamav3_prefix_diag.py

# Recurrent checkpoints at every page boundary of the output, copied from the
# speculative-decoding rollback history, so a follow-up turn resumes at the end
# of the previous answer instead of prefilling it again. Same states the fork's
# own checkpointing stores; EXL3_HIST_STASH=0 turns it off.
COPY patch_exllamav3_hist_stash.py /tmp/
RUN python3 /tmp/patch_exllamav3_hist_stash.py && rm /tmp/patch_exllamav3_hist_stash.py

# Anchor checkpoints: also stash the recurrent state at the page boundary before the
# latest user message (EXL3_CONV_CKPT=1) and, with =2, at the end of a long prefix
# shared with a recent different prompt (a new session with the same system prompt),
# in a separate LRU of EXL3_CONV_CKPT_MAX entries. Moves prefill split points only
# (as prefill chunking does); on (=2, approved 2026-09-26), EXL3_CONV_CKPT=0 turns it off.
COPY patch_exllamav3_conv_ckpt.py /tmp/
RUN python3 /tmp/patch_exllamav3_conv_ckpt.py && rm /tmp/patch_exllamav3_conv_ckpt.py

# With MTP, the last full page of a k*256+1-token prompt kept a placeholder hash, so
# its K/V and the checkpoint prefill stashes after it were never found again (a
# follow-up prefilled from 0). EXL3_LASTPAGE=1 gives the page its content hash when
# prefill completes it (bookkeeping only; approved 2026-09-26). =2 also lets an exact
# re-send resume there (not approved yet, see the script); 0 = the fork's behaviour.
COPY patch_exllamav3_lastpage.py /tmp/
RUN python3 /tmp/patch_exllamav3_lastpage.py && rm /tmp/patch_exllamav3_lastpage.py

# One [decode-stats] line per request: tokens and time of the thinking phase, the answer
# text and the tool calls, with draft counters per phase (Plan D, D0; tools/decode_report.py).
# Log-only, on; EXL3_DECODE_STATS=0 turns it off.
COPY patch_exllamav3_decode_stats.py /tmp/
RUN python3 /tmp/patch_exllamav3_decode_stats.py && rm /tmp/patch_exllamav3_decode_stats.py

# Fallback sampling for requests that set none: the model's generation_config.json
COPY sampler_overrides/qwen38.yml /app/sampler_overrides/qwen38.yml
COPY config.yml /app/config.yml
COPY --chmod=755 entrypoint.sh /app/entrypoint.sh

# ---------------------------------------------------------------------------
# 4. The GB10 launcher's knobs (scripts/exl3_native/tuning/run-qwen38-exl3.sh).
#    Any of them can be overridden with docker run -e. One departure from the
#    launcher: EXL3_NGRAM_STREAM=1 streams the n-gram table from NVMe. The
#    launcher's 0 loads all ~30 GiB of it into RAM, which measured no faster.
# ---------------------------------------------------------------------------
ENV EXL3_INT8_GEMV=0 \
    EXL3_MOE_COOP_WIDE=1 \
    EXL3_GR_INT8=1 \
    EXL3_MTP_HEAD_N=65536 \
    EXL3_NGRAM_STREAM=1 \
    EXL3_DRAFT_CONFIDENCE=0.6 \
    EXL3_MOE_FUSED_UNIFORM=1 \
    EXL3_QSA_STAGE=1 \
    EXL3_GDN_NOCOPY=1 \
    TABBY_ENCODE_CACHE=1 \
    EXL3_PLD=1 \
    EXL3_PLD_MAX=11 \
    EXL3_MOE_BSZN_MAX=16 \
    EXL3_HIST_STASH=1 \
    EXL3_CONV_CKPT=2 \
    EXL3_LASTPAGE=1 \
    EXL3_PREFIX_DIAG=1 \
    EXL3_DECODE_STATS=1

RUN { echo "torch=$(python3 -c 'import importlib.metadata as m; print(m.version("torch"))')"; \
      echo "exllamav3=${EXLLAMAV3_REF} $(git -C /opt/src/exllamav3 rev-parse HEAD)"; \
      echo "tabbyapi=${TABBYAPI_REF} $(git -C /app rev-parse HEAD)"; \
      echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"; \
    } > /app/BUILD_INFO \
 && cat /app/BUILD_INFO

# Unprivileged runtime user. It owns /app because TabbyAPI writes
# api_tokens.yml there when auth is enabled in config.yml. It also owns
# ~/.cache, where start_tabby.sh mounts a volume (below).
RUN useradd --create-home --uid 10001 tabby \
 && mkdir -p /home/tabby/.cache \
 && chown -R tabby:tabby /app /home/tabby/.cache
USER tabby

# Kernel tuning results survive restarts: exllamav3's GEMM autotuner and
# Triton's (the GatedDeltaNet kernels) write them to ~/.cache, which
# start_tabby.sh keeps in a Docker volume. Without it every new container
# tunes again: about 20 s on the first request and about 1 s on the first
# prompt of each new shape. The exllamav3 file is named after the engine
# commit, so a rebuild on another commit starts a fresh one.
ENV EXLLAMAV3_TUNE_CACHE=/home/tabby/.cache/exllamav3-autotune-${EXLLAMAV3_REF}.bin \
    TRITON_CACHE_DIR=/home/tabby/.cache/triton \
    TRITON_CACHE_AUTOTUNING=1

WORKDIR /app
EXPOSE 5000
# entrypoint.sh drops the model's page cache, then runs main.py with the
# container's arguments (e.g. --model-name).
ENTRYPOINT ["/app/entrypoint.sh"]
