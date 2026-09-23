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
#      device (patch_exllamav3_checkpoints.py)
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
#    aarch64 guards, so no patch step. --no-build-isolation so setup.py sees
#    the torch above; without it the extension is silently skipped.
# ---------------------------------------------------------------------------
RUN git clone "${EXLLAMAV3_REPO}" /opt/src/exllamav3 \
 && git -C /opt/src/exllamav3 checkout --quiet "${EXLLAMAV3_REF}" \
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

# Recurrent-state checkpoints stay on the device instead of blocking copies to
# pageable host memory; see the script. Fails the build if an anchor moved.
COPY patch_exllamav3_checkpoints.py /tmp/
RUN python3 /tmp/patch_exllamav3_checkpoints.py && rm /tmp/patch_exllamav3_checkpoints.py

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
    EXL3_DRAFT_CONFIDENCE=0.6

RUN { echo "torch=$(python3 -c 'import importlib.metadata as m; print(m.version("torch"))')"; \
      echo "exllamav3=${EXLLAMAV3_REF} $(git -C /opt/src/exllamav3 rev-parse HEAD)"; \
      echo "tabbyapi=${TABBYAPI_REF} $(git -C /app rev-parse HEAD)"; \
      echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"; \
    } > /app/BUILD_INFO \
 && cat /app/BUILD_INFO

# Unprivileged runtime user. It owns /app because TabbyAPI writes
# api_tokens.yml there when auth is enabled in config.yml.
RUN useradd --create-home --uid 10001 tabby \
 && chown -R tabby:tabby /app
USER tabby

WORKDIR /app
EXPOSE 5000
# entrypoint.sh drops the model's page cache, then runs main.py with the
# container's arguments (e.g. --model-name).
ENTRYPOINT ["/app/entrypoint.sh"]
