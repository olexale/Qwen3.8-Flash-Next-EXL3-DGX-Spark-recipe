# syntax=docker/dockerfile:1
# ============================================================================
# vLLM + exllamav3 + vllm-exl3, patched for Qwen3.8-Flash-Next EXL3 on one
# DGX Spark (GB10, sm_121, aarch64). Build it with ./build.sh, which reads the
# pins below from .env.
#
# Everything the "Quick start: vLLM path" section of the README does by hand
# happens here, once, at build time:
#   1. vLLM from PyPI (prebuilt aarch64 wheel, torch 2.13.0+cu130)
#   2. exllamav3 from source with the plugin's aarch64 patch, extension
#      compiled ahead of time for sm_121 (no JIT on first start)
#   3. vllm-exl3 from source, its vllm_exl3_c extension compiled against (2)
#   4. the three Qwen4Exp quant-config patches applied to vLLM in place
#
# The build needs no GPU; TORCH_CUDA_ARCH_LIST says what to compile for.
# ============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG VLLM_VERSION=0.29.0
ARG EXLLAMAV3_REPO=https://github.com/turboderp-org/exllamav3
ARG EXLLAMAV3_REF=v1.4.7
ARG VLLM_EXL3_REPO=https://github.com/vcruz305/vllm-exl3
ARG VLLM_EXL3_REF=94c29ba25a66a7583bfa49546c783d3e70859cc6
# GB10 is sm_121. It also runs sm_120 binaries, but compile for what it is.
ARG TORCH_CUDA_ARCH_LIST=12.1
# nvcc jobs for the two extension builds. Each can take a few GiB.
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        git curl ca-certificates build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's system Python (3.12) is externally managed; use a venv.
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# ---------------------------------------------------------------------------
# 1. vLLM
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
 && pip install "vllm==${VLLM_VERSION}" \
 && python3 -c 'import torch, sys; c = torch.version.cuda or ""; print("torch", torch.__version__, "CUDA", c); sys.exit(0 if c.startswith("13.") else "expected a CUDA 13 torch build, got CUDA " + repr(c))'

# ---------------------------------------------------------------------------
# 2. Sources. Full clones so EXLLAMAV3_REF / VLLM_EXL3_REF can be a tag, a
#    branch, or a (short) commit. Kept in the image: the pack-prep tools and
#    the vLLM patch scripts live in the plugin repo, not in its wheel.
# ---------------------------------------------------------------------------
RUN git clone "${EXLLAMAV3_REPO}" /opt/src/exllamav3 \
 && git -C /opt/src/exllamav3 checkout --quiet "${EXLLAMAV3_REF}" \
 && git clone "${VLLM_EXL3_REPO}" /opt/src/vllm-exl3 \
 && git -C /opt/src/vllm-exl3 checkout --quiet "${VLLM_EXL3_REF}"

# ---------------------------------------------------------------------------
# 3. exllamav3, aarch64-patched, extension precompiled.
#    --no-build-isolation so setup.py sees the torch installed above; without
#    torch it silently skips the extension and the plugin cannot import it.
# ---------------------------------------------------------------------------
RUN python3 /opt/src/vllm-exl3/tools/patch_exllamav3_aarch64.py \
        /opt/src/exllamav3/exllamav3/exllamav3_ext \
 && cd /opt/src/exllamav3 \
 && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        pip install --no-build-isolation . \
 && python3 -c 'import importlib.util as u, sys; sys.exit(u.find_spec("exllamav3_ext") is None and "exllamav3_ext was not built")'

# ---------------------------------------------------------------------------
# 4. vllm-exl3. Its own CUDA extension includes exllamav3_ext's headers, which
#    the installed exllamav3 wheel does not carry, so point it at the source.
#    Same silent-skip-without-torch rule as above.
# ---------------------------------------------------------------------------
RUN EXL3_EXT_INCLUDE=/opt/src/exllamav3/exllamav3/exllamav3_ext \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        pip install --no-build-isolation /opt/src/vllm-exl3 \
 && python3 -c 'import importlib.util as u, sys; sys.exit(u.find_spec("vllm_exl3_c") is None and "vllm_exl3_c was not built")'

# ---------------------------------------------------------------------------
# 5. Patch vLLM in place. Each script is idempotent, keeps a .orig backup, and
#    exits non-zero if its anchor is missing, which fails the build here
#    instead of failing a ten-minute model load with a misleading OOM later.
# ---------------------------------------------------------------------------
RUN VLLM_DIR="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("vllm").origin))')" \
 && for p in patch_vllm_qwen4_ple.py patch_vllm_mtp_lmhead.py patch_vllm_vision_split.py; do \
        python3 "/opt/src/vllm-exl3/tools/patch_vllm_qwen4_exp/$p" "$VLLM_DIR" || exit 1; \
    done

# ---------------------------------------------------------------------------
# 6. Recipe scripts and build provenance.
# ---------------------------------------------------------------------------
COPY scripts/serve_one_spark_qwen.sh scripts/prepare_pack.sh scripts/preflight.py /opt/recipe/
RUN chmod +x /opt/recipe/*.sh \
 && { echo "vllm=$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')"; \
      echo "torch=$(python3 -c 'import importlib.metadata as m; print(m.version("torch"))')"; \
      echo "exllamav3=${EXLLAMAV3_REF} $(git -C /opt/src/exllamav3 rev-parse HEAD)"; \
      echo "vllm-exl3=${VLLM_EXL3_REF} $(git -C /opt/src/vllm-exl3 rev-parse HEAD)"; \
      echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"; \
    } > /opt/recipe/BUILD_INFO \
 && cat /opt/recipe/BUILD_INFO

ENV PLUGIN_REPO=/opt/src/vllm-exl3 \
    MODEL_DIR=/model \
    HF_HOME=/root/.cache/huggingface

WORKDIR /opt/recipe
ENTRYPOINT ["/opt/recipe/serve_one_spark_qwen.sh"]
