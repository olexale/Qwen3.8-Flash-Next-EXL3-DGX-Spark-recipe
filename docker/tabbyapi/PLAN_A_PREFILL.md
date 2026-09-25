# Track A: faster prefill on the Spark (TabbyAPI, exllamav3 fork)

Written 2026-09-24 for the session that picks this up. Read the whole file
before starting; then read `OPTIMIZATION_PLAN.md` ("Results (2026-09-24)") for
the history this builds on. Everything you need is here, in
`docker/tabbyapi/`, and on the Spark.

## Goal

Make **prompt processing (prefill)** as fast as it can go on the current
hardware settings, without changing what the model outputs.

| | Now (2026-09-24) | Target | Stretch |
|---|---:|---:|---:|
| Cold long prompt (115k tokens, through the API) | 1,228–1,233 tok/s | ≥ 1,800 tok/s | ~2,200+ |
| Cold 20k prompt (engine benchmark, `run_engine_bench.sh`, `-e CHUNK=8192`) | 1,184 tok/s (16.9 s) | ≥ 1,700 tok/s | |
| Cold ~600-token prompt (API) | 1.11 s | ≤ 0.9 s | |
| Follow-up turn, 25k history + ~850 new (API) | 1.52 s | ≤ 1.2 s | |
| Decode, one session (API, 400 tokens, default sampling) | 54.0 tok/s (median of 10) | not lower | |

**Why we believe more is possible:** the vLLM container the owner used before
(`qwen38-flash`, stopped, same EXL3 pack, same machine, same GPU clock limit)
prefilled at **≥ 2,900 tok/s**:
`logs/qwen38-exl3-vllm-20260922T204815.log` on the Spark, line at
`09-22 20:43:48`: "Avg prompt throughput: 2933.5 tokens/s … Prefix cache hit
rate: 0.0%" (a 10 s logging window). Its decode was slower (26–28 tok/s),
which is why the owner moved to TabbyAPI. So vLLM is the reference for
prefill only.

## Constraints (the owner's; do not bend them)

- **Model behaviour must not change.** Prefer changes that are *bit-identical*
  to the code they replace (the `gr_collapse` kernel is the model:
  `patch_exllamav3_gr_collapse.py`, checked with `tools/gr_collapse_ab.py`).
  A change that only moves floating-point rounding is acceptable if it passes
  the gates below (the fused-MoE patch is the precedent). Nothing that
  changes sampling, quantization, KV precision or the draft settings.
- **GPU clocks stay as they are.** The owner runs `sudo nvidia-smi -lgc
  0,1600` (graphics clock capped at ~1,580–1,600 MHz; default 2,418, max
  3,003) for power. Never change clocks, never suggest it again as the plan.
  Prefill is compute-bound, so this cap costs prefill, and that is accepted.
- **Do not run the vLLM container or image for benchmarks.** It takes the
  whole RAM and loads slowly. Reading files out of the image is fine (no model
  load): `docker create` + `docker cp`, or `docker run --rm --entrypoint cat
  <image> <file>`. Never touch the `qwen38-flash` container. If, after all
  of this plan, a single profiled vLLM run would clearly help, ask the owner
  first and do it in one load.
- **No GPU rental, no retraining.**
- **TabbyAPI (`qwen38-tabby`) may be stopped and restarted**, but the owner
  uses it: announce each downtime window in the chat before stopping, keep
  windows short (5–15 min), restart right after, and leave it running on the
  best configuration when you finish. Two copies of the model do not fit, so
  every engine benchmark needs TabbyAPI stopped.
- **No warm-up request** added to startup.
- Ship engine changes as **patch scripts in `docker/tabbyapi/`**, applied at
  image build time, failing the build if an anchor is missing, each behind an
  environment variable (default on once approved) so it can be turned off
  without a rebuild. Source-level patches (CUDA) run before the extension
  compiles (see how `patch_exllamav3_gr_collapse.py` is wired into the
  Dockerfile); Python-only patches run on the installed package (like
  `patch_exllamav3_fused_moe.py`). Forward new `EXL3_*` variables in
  `start_tabby.sh` and document them in `README.md`.
- **Small commits to `main`, pushed** (the owner approved pushing). Never
  commit `.env`. Commit message trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## When to stop and ask the owner

- Before shipping anything that is not bit-identical (show the gates' numbers).
- Before starting a multi-day kernel rewrite (e.g. a new MoE prefill kernel).
- Before any vLLM run.
- When the targets are met, or when the remaining ideas are exhausted: report
  with numbers and what was tried.

## Setup

| | |
|---|---|
| Machine | NVIDIA DGX Spark, GB10 (sm_121, 48 SMs), 128 GB unified memory, aarch64 |
| SSH | `ssh gx10-b2fe.local` (user `ole`) |
| Repo on the Spark | `~/dev/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe` (clone of `olexale/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe`); locally `/Users/ole/dev/test/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe`. Commit locally, push, then `git pull --ff-only` on the Spark |
| Model | `~/models/Qwen3.8-Flash-Next-EXL3` (EXL3 3.05 bpw, `config.json.native` / `model.safetensors.index.json.native` are the originals; the scripts mount those) |
| Engine | [vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051) + three patches in the image: `patch_exllamav3_checkpoints.py`, `patch_exllamav3_fused_moe.py` (`EXL3_MOE_FUSED_UNIFORM=1`), `patch_exllamav3_gr_collapse.py` (`EXL3_GR_COLLAPSE=1`) |
| Server | TabbyAPI `2186cdb`, image `qwen38-exl3-tabby:latest`, container `qwen38-tabby`, port 18300. Older images kept for rollback: `:pre-20260924` (before all of this), `:pre-grcollapse` |
| Start / stop | `./start_tabby.sh [--build]`, `./stop_tabby.sh` in the repo root on the Spark; `docker logs qwen38-tabby` has per-request prefill/TTFT/acceptance lines |
| Config | `docker/tabbyapi/config.yml`: 262,144 max_seq_len, `cache_size: 786432`, 8-bit KV, MTP draft 5 + dynamic, `chunk_size: 8192`, vision on, `sysmem_recurrent_cache: 8192` |
| Kernel tuning cache | Docker volume `qwen38-tabby-cache` at `/home/tabby/.cache` (exllamav3 GEMM autotune + Triton); `run_engine_bench.sh` mounts it too |

Model shape: 48 layers (36 GatedDeltaNet linear attention, 12 full attention
with an indexer), MoE with 512 experts, top-10, expert intermediate 640,
hidden 2560, one shared expert, 4 hyper-connection residual streams
(`hc_count: 4`, `GatedResidual` mixers, rank 320) mixed twice per block, one
MTP layer, a per-layer n-gram embedding table streamed from NVMe.

## Tools (`docker/tabbyapi/tools/`)

- `run_engine_bench.sh <name> [docker args]` — runs a script in the image
  with TabbyAPI stopped; `SCRIPT=<file>` picks the script (default
  `engine_bench.py`), output in `logs/bench_<name>.log`. Pass `-e CHUNK=8192`
  to match the deployed config. `-v host:container:ro` mounts dev builds.
- `engine_bench.py` — TTFT for cold prompts (`SIZES`), a 20k conversation and
  a follow-up turn, built the way TabbyAPI builds the generator.
- `module_times.py` — **start here.** One real prefill of `TOKENS` with a
  device sync around every module forward: exclusive time per module
  (`TransformerBlock@N` = the block's own code outside its submodules).
- `moe_trace.py` — which MoE tier handles how many experts; one MoE layer
  alone on real hidden states, with parity between two paths.
- `gr_collapse_ab.py` — the pattern for proving a kernel bit-identical:
  capture real inputs during a prefill, run old and new paths, count differing
  elements. Copy it for new kernels.
- `greedy_ab.py` — greedy 400-token outputs of two variants + acceptance
  (`COMPARE=a,b`); mount a writable dir at `/out` (`~/scratch/out`, mode 777).
- `api_bench.py`, `concurrent_decode.py`, `three_sessions.py` — through the
  running API (cold 24.5k, follow-up, decode; decode with N sessions; three
  ~115k conversations + memory). `~/scratch/cold600.py` on the Spark: five
  cold ~600–1,100-token prompts.
- `scripts/exl3_native/tuning/ctxfill.py` — the 128k needle test. It expects
  the model at `~/models/...`; run it in the image with the model mounted at
  `/home/tabby/models/Qwen3.8-Flash-Next-EXL3` (see the command in "Gates").

### Dev loop for engine changes (no image rebuild per try)

1. Dev source tree: make a fresh one from the current image, so it has the
   shipped source patch:
   ```bash
   rm -rf ~/scratch/exl3dev && CID=$(docker create qwen38-exl3-tabby:latest) \
     && docker cp $CID:/opt/src/exllamav3 ~/scratch/exl3dev && docker rm $CID
   python3 ~/dev/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/docker/tabbyapi/patch_exllamav3_fused_moe.py ~/scratch/exl3dev/exllamav3
   ```
   (`~/scratch/exl3src` is the pristine `329e051` source; the old
   `~/scratch/exl3dev` carries dropped experiments — do not reuse it.)
2. Build the extension in place, on the small cores so TabbyAPI keeps
   serving (first build ~45 min, later builds recompile only what changed,
   ~3 min plus a ~1 min link):
   ```bash
   docker run --rm --cpuset-cpus 0-4,10-14 --user $(id -u):$(id -g) -e HOME=/tmp \
     -e MAX_JOBS=6 -e TORCH_CUDA_ARCH_LIST=12.1 -v $HOME/scratch/exl3dev:/src -w /src \
     --entrypoint python3 qwen38-exl3-tabby:latest setup.py build_ext --inplace
   ```
3. Test by mounting the built `.so` and any changed Python files over the
   installed ones:
   ```bash
   SP=/opt/venv/lib/python3.12/site-packages; D=$HOME/scratch/exl3dev
   DEV="-v $D/exllamav3_ext.cpython-312-aarch64-linux-gnu.so:$SP/exllamav3_ext.cpython-312-aarch64-linux-gnu.so:ro \
        -v $D/exllamav3/modules/<file>.py:$SP/exllamav3/modules/<file>.py:ro"
   SCRIPT=module_times.py docker/tabbyapi/tools/run_engine_bench.sh <name> $DEV
   ```
   Note: the installed package also carries `patch_exllamav3_checkpoints.py`
   (gated_delta_net.py, ple.py). If you mount one of those files from the dev
   tree, apply that patch to the dev tree first.
4. To ship: write the change as a patch script, wire it into the Dockerfile,
   build into a new tag (`docker build -t qwen38-exl3-tabby:<tag>
   docker/tabbyapi`, ~40 min when the extension recompiles; TabbyAPI keeps
   serving), keep the old `:latest` under a rollback tag, retag, restart.
   `docker build` needs BuildKit (the default); the legacy builder fails on
   `COPY --chmod`.

## What is known

### Where an 8,192-token prompt's time went (before the collapse kernel)

`module_times.py`, exclusive time, 8.4 s total (now ~6.7 s after
`gr_collapse`; **re-measure first**):

| Part | Time | Share |
|---|---:|---:|
| MoE (`BlockSparseMLP`, 144 calls = 48 layers × 3 forwards) | 3.1 s | 37% |
| Transformer blocks' own code (hyper-connection mixing; 48 × 46 ms) | 2.2 s | 26% — collapse kernel took ~1.3 s of it |
| Attention (12 layers, incl. indexer) | 1.1 s | 13% |
| GatedDeltaNet core (36 layers) | 0.7 s | 8% |
| Dense projections (`in_proj_qkv` 0.27, `in_proj_z` 0.17, `out_proj` 0.16, `q_proj` 0.10, `o_proj` 0.05 s) | ~0.8 s | 10% |
| N-gram embedding, norms, rest | ~0.3 s | 4% |

The generator prefills a prompt in `chunk_size` pieces, plus one extra forward
up to the last page boundary (recurrent checkpoint) and the last token. It
does **not** split at every 2,048-token checkpoint (checked).

### The MoE at long prompts

- At 8,192 rows one MoE layer takes ~56 ms on the GPU; the host returns in
  ~3 ms, so it is GPU-bound. Floors: ~4 ms (weight bytes) and ~8 ms (fp16
  tensor math). About 105 experts per layer exceed the fused tier's 256 rows
  and go through the batched-reconstruct tier (dequantize to fp16 + GEMM).
- The fused kernel `exl3_moe` runs 512-thread blocks, 128 registers per
  thread; the 64-row tile already spills (256 B stack).

### vLLM's approach (read from the image, not run)

- Image `qwen38-flash-next-exl3-vllm:latest`, built by `docker/Dockerfile`
  in this repo: vLLM **0.29.0** from PyPI (aarch64 wheel), exllamav3
  **v1.4.7** (turboderp upstream, *not* the fork), vllm-exl3 `94c29ba`, plus
  the Qwen4Exp patches to vLLM. **The image holds the exact sources**; copy
  them out once at the start of A1 (no model load, nothing runs):
  ```bash
  mkdir -p ~/scratch/vllm_ref && CID=$(docker create qwen38-flash-next-exl3-vllm:latest)
  docker cp $CID:/opt/src/exllamav3 ~/scratch/vllm_ref/exllamav3-1.4.7      # C++/CUDA + Python, aarch64-patched
  docker cp $CID:/opt/src/vllm-exl3 ~/scratch/vllm_ref/vllm-exl3            # plugin incl. vllm_exl3_c CUDA sources, tools/
  docker cp $CID:/opt/venv/lib/python3.12/site-packages/vllm ~/scratch/vllm_ref/vllm   # Python only; kernels are compiled
  docker rm $CID
  ```
  vLLM's own CUDA kernel sources are not in the image (wheel); if one matters,
  clone `vllm-project/vllm` at tag `v0.29.0`.
- The EXL3 plugin (`vllm_exl3` 0.4.2) is at
  `vllm/../vllm_exl3/exl3.py` (site-packages):
  `apply_exl3_fused_moe` uses exllamav3's own `exl3_moe` (from **v1.4.7**,
  not the fork) for experts with ≤ 128 rows
  (`TEMP_ROWS_FUSED`) and `apply_exl3_batched_fat` for bigger ones
  (`ext.reconstruct` to fp16 + `ext.hgemm`, per expert, persistent scratch;
  an `exl3_fat_gemm` native kernel for K=4 mcg only, so not for this pack).
- vLLM's model code for this architecture:
  `/opt/venv/lib/python3.12/site-packages/vllm/models/qwen4_exp/nvidia/`
  (`model.py`, `ple_layer.py`, `qsa.py`, `low_latency_gemm.py`, `mtp.py`;
  `.orig` files are pre-patch copies).
- Its launch settings (from the saved log): `--max-num-seqs 3`, MTP with 3
  speculative tokens, prefix caching, piecewise CUDA graphs with **inductor
  compilation** of everything outside the listed "splitting ops"
  (attention, GDN core, PLE n-gram ops, QSA, indexer…). That is, the glue
  between those ops (hyper-connection mixing, norms, activations, residuals)
  is compiled/fused, where exllamav3 runs it eagerly op by op.

### Tried, do not redo

- Knobs with the fused MoE on (cold 40k tok/s, base 1,026–1,047):
  `EXL3_MOE_TILE_N=128` 1,026; `EXL3_MOE_FUSED_DET=0` 999; `EXL3_MOE_MTILE=0`
  949; fused rows 128 1,008 / 512 1,033; `EXL3_MOE_COOP_WIDE=0` same;
  `VISION=0` same; `EXL3_NGRAM_STREAM=0` same (also at 128k); chunk 4096
  worse; chunk 16384 +1.7% for +1.6 GiB (80.7 GiB system-wide, over the
  owner's ~80 GB budget) — re-test only after other gains.
- A 128-row tile instance for the fused MoE kernel: bit-identical, but
  0.65–0.84× (register spills, 696 B stack). Dropped.
- SMs per expert group (`MOE_SMS_PER_EXPERT` made runtime, 4–16): best (4)
  +3–4% on the MoE layer, ~1% end to end. Dropped.
- Everything under "Settled" and "Tried" in `OPTIMIZATION_PLAN.md`.

## Tasks

Order: A0 → A1 → A2, then A3 items by measured size. Commit and report after
each shipped item.

### A0. Confirm the vLLM reference (1 h, no downtime)

Read how vLLM computes "Avg prompt throughput" in this image
(`vllm/v1/metrics/loggers.py` or similar): does it count prefix-cache hits,
is the interval exactly the logging period? Check the other windows in the
saved vLLM logs (`logs/qwen38-exl3-vllm-*.log`) for request sizes (KV cache
usage %, prompt tokens) so the ≥ 2,900 tok/s figure is solid. If it turns
out lower, update the targets above and tell the owner.

### A1. Read vLLM's implementation of this model (half a day, no downtime)

Copy the sources out of the image first (commands under "vLLM's approach").
Compare with exllamav3 (`modules/hyperconnections.py`,
`modules/transformer.py`, `modules/gated_delta_net.py`, `modules/attn.py` or
equivalent, `modules/block_sparse_mlp.py`, `generator/job.py` prefill):

- hyper-connection mix/apply: which ops, fused or compiled, dtypes;
- GatedDeltaNet prefill kernels (FLA chunk kernels? fused norm/packing?);
- full-attention prefill kernel and the indexer/QSA path;
- how dense EXL3 projections run at large M (exl3 GEMM, or reconstruct +
  cuBLAS/hgemm);
- MoE: stock `exl3_moe` launch parameters vs the fork's tier plan;
- prefill chunk size (`max_num_batched_tokens` default in that vLLM).

Write the findings into this file under a new "Findings" section, with the
concrete differences ranked by likely time.

### A2. Re-profile the current image (one ~15 min window)

`module_times.py` at `TOKENS=8192` and at `TOKENS=32768` (attention grows
with depth). For the top 3 modules, a `torch.profiler` kernel table scoped to
that module (see `moe_trace.py` SCOPE code) so you know which kernels and
torch ops dominate.

### A3. Candidates (pick by A1/A2 measurements)

a. **Rest of the GatedResidual prefill path** (`GatedResidual._mix` for R >
   32 in `modules/hyperconnections.py`): `ext.rms_norm` → `matmul(proj_h)`
   → `silu(dm[:, :rank] / H)` and `2 * sigmoid(dm[:, rank:] / H)` →
   `matmul(up_h)` → `gr_collapse`. Fuse the elementwise pieces; check the two
   GEMM shapes (R × 10240 × (320+4), R × 320 × 10240) use tensor cores
   efficiently. Plus `hc_apply` and the leftover ops in
   `TransformerBlock.forward` (`y.half()`, `to2`, residual scalars).
b. **Attention prefill** (1.1 s / 8k before; grows with context).
c. **GatedDeltaNet prefill** (0.7 s / 8k).
d. **Dense projections at large M**: exl3 GEMM vs reconstruct + fp16 GEMM;
   exllamav3 may already switch by M; measure both for the actual shapes.
e. **MoE kernel version**: A/B the `exl3_moe` that vLLM used (exllamav3
   v1.4.7, `~/scratch/vllm_ref/exllamav3-1.4.7`, already aarch64-patched;
   the patch tool is `~/scratch/vllm_ref/vllm-exl3/tools/patch_exllamav3_aarch64.py`)
   against the fork's on a captured layer, as an extension-only dev build in
   the tabby image (its torch). Also vLLM's split (≤ 128 rows fused, above:
   reconstruct + hgemm) against the fork's (≤ 256 fused, above: batched
   reconstruct).
f. **Chunk size 16384** again, once memory headroom allows.
g. **Client-side overhead on long follow-ups** (not prefill, but TTFT): the
   client sees ~0.4 s more than TabbyAPI's own "first token" on 115k
   conversations. Find where it goes (chat template rendering, tokenizing the
   whole history each turn, HTTP) in TabbyAPI (`/app` in the image) and cut
   it if it is cheap (e.g. cache tokenization of an unchanged prefix).
h. A **new MoE prefill kernel** (e.g. 256-thread blocks with a bigger
   register budget, or decode-once-into-shared-memory feeding tensor cores).
   Largest ceiling, weeks of work, ~25% odds: **ask the owner before
   starting.**

## Gates for every change

1. **Parity on real inputs**: capture the module's real inputs during a
   prefill (pattern: `gr_collapse_ab.py`), compare old vs new output. Report
   differing elements and max relative difference. Bit-identical = done.
2. If not bit-identical (ask the owner before shipping):
   - `greedy_ab.py`: old image twice and new image twice
     (`-e TAG=old1` … and `COMPARE=old1,old2`, `new1,new2`, `old1,new1`,
     `old2,new2`); new-vs-old divergence must be no earlier than old-vs-old /
     new-vs-new, acceptance within the same spread. On 2026-09-24 the old
     image diverged from itself at tokens 82 (DevOps) and 148 (prose); the
     new image from itself at 44 / 108 / 22.
   - 128k needle, twice (different seeds):
     ```bash
     MD=$HOME/models/Qwen3.8-Flash-Next-EXL3; H=/home/tabby/models/Qwen3.8-Flash-Next-EXL3
     docker run --rm --gpus all --cpuset-cpus 5-9,15-19 -v $MD:$H:ro \
       -v $MD/config.json.native:$H/config.json:ro \
       -v $MD/model.safetensors.index.json.native:$H/model.safetensors.index.json:ro \
       -v qwen38-tabby-cache:/home/tabby/.cache \
       -v $PWD/scripts/exl3_native/tuning/ctxfill.py:/tmp/ctxfill.py:ro \
       -e K=128 -e CQ=8,8 -e SEED=0 --entrypoint python3 qwen38-exl3-tabby:<tag> /tmp/ctxfill.py
     ```
3. **Speed**: `run_engine_bench.sh` with `-e CHUNK=8192` before/after; then
   through the API after deploying (`api_bench.py`, `~/scratch/cold600.py`,
   `three_sessions.py` for 115k prefill and memory).
4. **No regressions**: decode `N=1 REPS=10 concurrent_decode.py` (median
   ~54 tok/s), `N=3 REPS=5` (~54 tok/s aggregate); memory under
   `three_sessions.py` ≤ ~80 GiB system-wide.

## Reporting

After each shipped change: update `README.md` "What to expect" and the
`EXL3_*` table, add a dated section to `OPTIMIZATION_PLAN.md` "Results"
(numbers, what was tried and did not help), commit, push, pull on the Spark.
Raw logs stay in `logs/` on the Spark (gitignored).
