# TabbyAPI on the Spark: optimization plan

Written 2026-09-23 for whoever picks this up. Everything you need to start is
in this file and in `docker/tabbyapi/`. Read the whole file before starting;
the constraints section is short and matters.

## Goal

The owner uses this server mostly for coding through the **pi** agent, with
long contexts and up to three sessions at once. Compared with the vLLM
container they ran before, **time to first token (TTFT) is the pain point**,
especially on short follow-up turns. Decode speed is acceptable.

| | Now | Target |
|---|---:|---:|
| Follow-up turn: cached history + ~600 new tokens | 3.3–3.6 s | ≤ 1.5 s |
| Cold 600-token prompt | 3.9 s | ≤ 1.5 s |
| Cold prefill, long prompts | ~840 tok/s | ≥ 1,100 tok/s |
| Sessions whose full 262,144-token context stays cached | 1 | 3 |
| Decode | ~56 tok/s (code, default sampling) | not lower |

## Constraints

- **Model behaviour must not change.** No changes to sampling (the `qwen38`
  preset stays as it is) and no changes that alter outputs beyond
  floating-point noise. Speculative-decoding settings are fine, see T2.
- **Do not touch the other containers on the Spark**, in particular the vLLM
  one (`qwen38-flash`, stopped). Only `qwen38-tabby` is yours.
- **TabbyAPI may be stopped and restarted**, but the owner uses it: announce
  downtime, and leave it running on the best configuration when you finish.
  Two copies of the model do not fit in memory, so every engine benchmark
  needs TabbyAPI stopped.
- **The first request after a start may stay slow** (~8 s of one-time kernel
  setup). The owner does not want a warm-up request added.
- Ship engine changes as patch files in `docker/tabbyapi/`, applied at image
  build time and failing the build if their anchor is missing. Follow
  `patch_exllamav3_checkpoints.py`. Keep each one behind an environment
  variable where that is practical, so it can be turned off without a rebuild.
- Commit to `main` in small commits (the owner's preference). Never commit
  `.env`: it holds a Hugging Face token.

## Setup

| | |
|---|---|
| Machine | NVIDIA DGX Spark: GB10, 128 GB unified memory (CPU and GPU share it), aarch64, 10 Cortex-X925 + 10 A725 cores |
| SSH | `ssh gx10-b2fe.local` (user `ole`) |
| Repo on the Spark | `~/dev/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe` (clone of `olexale/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe`) |
| Model | `~/models/Qwen3.8-Flash-Next-EXL3`, turboderp's EXL3 pack, revision `3.05bpw_h5_ng5`. Prepared for vLLM too: `config.json.native` and `model.safetensors.index.json.native` are the originals, and `start_tabby.sh` mounts those |
| Engine | [vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051), built into the image |
| Server | TabbyAPI `2186cdb`, image `qwen38-exl3-tabby:latest`, container `qwen38-tabby`, port 18300, model name `qwen3.8-flash-next`, no API key |
| Start / stop | `./start_tabby.sh [--build]`, `./stop_tabby.sh`, in the repo root; they read `.env` |
| Engine config | `docker/tabbyapi/config.yml`: 262,144 context, 8-bit KV, MTP draft 5 tokens with dynamic drafting, `chunk_size: 8192`, vision on |

Model shape that matters here: 48 layers (36 GatedDeltaNet linear attention,
12 full attention), MoE with 512 experts, top-10 routing, expert intermediate
size 640, hidden size 2560, one shared expert, one MTP layer, and a per-layer
n-gram embedding table read from NVMe.

## Tools

In `docker/tabbyapi/tools/`, used for every number in this file:

- **`run_engine_bench.sh <name> [docker run args]`** runs `engine_bench.py`
  inside the image, without TabbyAPI in the loop. It builds the engine the way
  TabbyAPI does and reports TTFT for a 1k warm-up, cold prompts
  (`SIZES`, default 600, 3k, 12k, 20k, 40k), and a follow-up turn (20k cached
  + ~600 new). `-e PROFILE=600` adds a torch profile of one 600-token
  prefill. `-e CHUNK=`, `-e MBS=`, `-e VISION=0` and any `EXL3_*` variable
  change one thing at a time. **Note:** its defaults are TabbyAPI's
  (`CHUNK=2048`), not the shipped config; pass `-e CHUNK=8192` to measure
  what is deployed. Takes about 5 minutes per run, most of it model load.
- **`api_bench.py`** runs on the Spark's host against the running TabbyAPI:
  TTFT and decode rate through the real API. Cross-check its numbers with
  TabbyAPI's own per-request log lines (`docker logs qwen38-tabby`), which
  report prompt tokens, % cached, prefill tok/s, TTFT and draft acceptance.

Both scripts were adapted from the ones that produced the numbers below; run
each once to confirm it still works before relying on it.

## What is known (2026-09-23)

### Measurements

Engine benchmark, TTFT in seconds (`run_engine_bench.sh`):

| Configuration | cold 600 | cold 3k | cold 12k | cold 20k | follow-up 20k + 600 |
|---|---:|---:|---:|---:|---:|
| A: TabbyAPI defaults (chunk 2048, n-gram table in RAM) | 4.0 | 9.2 | 27.7 | 42.6 | 4.7 |
| n-gram table streamed from NVMe | 5.1 | 9.8 | 28.0 | 42.8 | 4.5 |
| chunk 4096 | 5.0 | 7.5 | 21.2 | 31.7 | 4.4 |
| chunk 8192 | 4.9 | 7.4 | 17.6 | 26.7 | 4.4 |
| J: chunk 8192 + checkpoints on device (**shipped**) | 3.9 | 6.3 | 16.1 | 24.4 | 3.3 |

Chunk 16384 was a wash (20k: 23.6 s; 12k: 19.4 s; 40k: 46.8 s vs 45.6 s at
8192) and chunk 32768 does not load ("Insufficient VRAM"). No effect: CPU
pinning on or off, passing batch and chunk sizes to `model.load`.

Through the API after the shipped changes: cold 24.5k-token prompt prefills
at 843 tok/s (29 s TTFT); follow-up turn with 853 new tokens 3.6 s; 400-token
code answer at 56 tok/s with 66% draft acceptance; ~65 GB memory in use.

### Profile of a 600-token prefill (configuration J)

3.8 s TTFT. Recurrent-state checkpoint copies accounted for 0.9 s before the
checkpoint patch. The rest is dominated by the MoE layers doing per-expert
work:

| Operation | Calls |
|---|---:|
| `cudaLaunchKernel` | ~85,000 |
| `exl3_gemv_kernel` (all variants) | ~52,000 |
| `aten::index_select` | ~28,000 |
| `aten::index_add_`, `aten::mul_` | ~22,800 each (≈ 48 layers × ~475 experts) |
| `cudaGraphLaunch` | ~22,900 |

A 600-token prompt makes ~6,000 token-to-expert assignments, so nearly every
one of the 512 experts in every layer is used, with ~12 tokens each. The floor
for that is reading all routed expert weights once: about 45 GB at 3 bits,
roughly 0.2 s at this machine's memory bandwidth. Today it takes ~2.5 s.

### Why the per-expert path runs: unknown

`exllamav3/modules/block_sparse_mlp.py` in the fork has a fused multi-expert
path (`fused_mode_buffers`, enabled by `support_fused`), which should take
experts with up to 128–256 rows in one launch. Its conditions look satisfied
for this model:

- gate, up and down experts all use codebook `mul1`, 3-bit (read from the
  pack's tensor headers; an earlier guess that the codebooks differ was wrong)
- activation `silu`, no expert biases, intermediate size 640 needs no padding

Yet the profile looks per-expert. Either one of those conditions is false at
runtime, the ops come from somewhere else, or the fused path hands most
experts to the per-expert loop. **Finding out is the first real task (T3).**

### A lead: the same engine measured faster before

The repo's README reports, on this machine:

- stock exllamav3 1.5.0 (`scripts/exl3_native/bench_native.py`: chunk 2048,
  batch size 1): cold prefill **1,129 tok/s** at 24k
- this fork (`scripts/exl3_native/tuning/ctxfill.py`: chunk 4096): **~900
  tok/s at 4k, ~1,150 at 128k–240k**

Configuration A above, the same fork at chunk 2048, gets ~470 tok/s. Things
that differ from those harnesses and have not been tested:

- batch size: `Generator` default 256 and `Cache` default 16 there, 4 here
  (TabbyAPI's default for recurrent models)
- the vision tower is loaded here
- the draft cache's `max_history`
- the GB10 knobs (`EXL3_MOE_COOP_WIDE=1`, `EXL3_GR_INT8=1`,
  `EXL3_INT8_GEMV=0`, `EXL3_MTP_HEAD_N`): tuned for decode, never measured on
  prefill
- stock 1.5.0 kernels vs the fork

### Settled, do not redo

- `EXL3_NGRAM_STREAM=1` (table on NVMe) costs nothing measurable and saves
  ~30 GiB.
- Recurrent-state checkpoints on the device (`patch_exllamav3_checkpoints.py`):
  GB10's memory is unified, so the stock `.cpu()` copies bought nothing.
- The README's "Levers that are closed" list for the native engine (deeper
  drafts, CUDA-graphing the decode step, mixer kernel rewrites).

## Tasks

Order: T1 and T2 are independent and small. T3 comes before T4 and ends in a
decision the owner makes. T5 is independent of T3/T4 but touches the same
generator code as T2; do T2 first.

### T1: room for three full contexts

Change `docker/tabbyapi/config.yml`:

- `cache_size: 786432` (3 × 262,144). Keep `max_seq_len: 262144`, the
  model's trained window. KV at 8-bit is ~12 KB/token, so this adds ~6 GiB.
- Check `max_batch_size` stays ≥ 3 (TabbyAPI defaults to 4 for this model).
- Check whether TabbyAPI's `sysmem_recurrent_cache` (4 GiB, the budget for
  recurrent-state checkpoints, now held on the device) is enough for three
  long sessions. Find the size of one checkpoint and how many a session keeps
  (`exllamav3/cache/recurrent.py`, `recurrent_checkpoint()` in
  `generator/generator.py`); raise it if three sessions would evict each
  other's checkpoints.

Done when: three conversations of ~100k tokens each, interleaved, each get a
follow-up turn that TabbyAPI logs as ≥ 95% cached; memory in use under load
stays under ~80 GB; `README.md` "What to expect" is updated.

### T2: re-tune speculative decoding for sampled output

`draft_num_tokens: 5` and `EXL3_DRAFT_CONFIDENCE=0.6` were tuned with greedy
decoding. The owner's traffic is sampled (temperature 1.0, top_k 20,
top_p 0.95 by default), where acceptance is lower and the best values likely
differ.

These settings do not change outputs: the fork samples each position from the
target model and accepts a drafted token only if it matches the sample
(`generator.py`, around line 1258), so the output distribution is the same as
without drafting.

- Sweep `draft_num_tokens` 3–7 × `EXL3_DRAFT_CONFIDENCE` 0.4–0.8 with
  `dynamic_draft` on. Use `TABBY_CONFIG` (a copy of `config.yml`) and `.env`
  so no rebuild is needed.
- Workload: code, prose, and pi-style tool calls; at least 10 samples per
  cell, since sampled acceptance is noisy. Report the median and range.
- Adopt a change only if it is ≥ 5% faster with non-overlapping ranges, and
  not slower on any prompt class.

### T3: find why short prefills are slow (investigation only)

1. **Which MoE path runs.** Load the model in the image and print, for each
   MoE layer, `is_quantized`, `uniform_expert_q`, `support_quant_paths`,
   `support_fused`, `fused_mode_buffers is not None`, `fused_rows`, and
   whether `bc` is set. Then trace one 600-token forward through
   `BlockSparseMLP.forward`: which tier handles how many experts.
2. **Attribute the profile.** Rerun the 600-token profile with ops tagged by
   module (`torch.profiler` with `with_modules=True`, or `record_function`
   around the MoE forward, the GatedDeltaNet layers, the n-gram embedding and
   attention). Confirm how much of the ~23k `index_add_` calls and ~52k GEMVs
   are MoE and how much is something else.
3. **Reconcile with the README's faster numbers.** Starting from A, change one
   thing at a time toward `ctxfill.py`'s setup: batch sizes, `VISION=0`,
   each GB10 knob off, the draft cache's `max_history`. Then run
   `bench_native.py` on stock 1.5.0 at the same prompt sizes, in a container
   built like the image but with `exllamav3==1.5.0` (see
   `scripts/exl3_native/setup_exllamav3_150.sh` for the aarch64 fix-ups).
4. **Size the prize.** Benchmark one MoE layer alone on real hidden states at
   16, 64, 600 and 2,048 tokens and compare with its weight-reading floor.

Deliverable: a short write-up of the cause, with numbers, and one of:

- (a) a configuration or condition keeps the fused path off: propose the fix
- (b) the fused kernel runs but is slow at ~12 rows per expert: propose tuning
- (c) prefill really has no grouped path for this shape: propose T4 with an
  effort estimate

**Stop there and report to the owner before starting T4.**

### T4: make short prefills fast (after the owner's go)

Scope depends on T3. For (c), the plan is to adapt the fork's existing fused
multi-expert kernel (or the mixed-K "unified" MoE kernel) so all small experts
of a layer run in one launch, rather than writing one from scratch.

Correctness gates, all required:

- per-layer parity with the current path on real hidden states, within the
  tolerance the fork's own parity scripts use
  (`scripts/exl3_native/tuning/gr_parity.py` shows the method)
- greedy 400-token outputs on the code, DevOps and prose prompts: identical,
  or diverging only late with unchanged draft acceptance (±1 point); the int8
  mixer change in the README is the precedent
- the needle test at 128k (`ctxfill.py`) still finds the code

Performance targets: the table at the top. Decode must not get slower.

### T5: prompt-lookup drafting alongside MTP (experiment)

Idea: when the recent output matches text already in the context (pi's edit
tool calls repeat existing code verbatim; so do file rewrites), propose the
continuation of that match as the draft, 10–20 tokens where the MTP head
proposes up to 5. Otherwise draft with MTP as today. Accepted tokens are
checked the same way as now, so outputs do not change.

What to know first:

- The fork refuses n-gram and MTP drafting together (`Generator` asserts
  `not ngram_match_min` when a draft model is set). The hybrid has to fit into
  the optimized MTP loop (batched verify, device-resident draft chain, dynamic
  draft length) in `generator/generator.py`.
- Checking a long draft costs more on this model: 37 ms per round at 2 draft
  tokens, 52 at 5, 86 at 9 (README), because each extra token pulls in more
  experts. So draft long only on long matches, and cap the length.
- The GatedDeltaNet layers keep per-token state history for rollback, sized
  by the longest draft (`max_history`) and the batch size. Estimate the memory
  at the chosen cap; `max_batch_size: 3` may be needed.
- Measuring needs a recorded pi editing session to replay. TabbyAPI does not
  log prompts (`log_prompt: false`). **Ask the owner** how to get one; it is
  their data.

Ship behind `EXL3_PLD=1`, off by default. Keep it only if edit-heavy turns get
clearly faster and other turns are not slower (≤ 2%). If it fails, remove the
patch; the report of what was tried goes in `README.md` next to the other
negative results.

## Reporting

After each task, update `docker/tabbyapi/README.md` "What to expect" with the
new numbers and the date, and note what was tried and did not help. Keep raw
logs in `logs/` on the Spark (gitignored).

## Results (2026-09-24)

Raw logs are in `logs/` on the Spark (`bench_*.log`).

### T3: why short prefills are slow — answer (a), a condition keeps the fused path off

**Cause.** The fork's commit `785f206` ("per-K-group fused MoE dispatch for
mixed-K packs") moved the block that sets `support_fused` in
`BlockSparseMLP.load_local()` into the branch for *mixed-K* packs. This pack
is uniform (every expert 3-bit `mul1`), so `support_fused` stays False,
`fused_mode_buffers` is never built, and prefill falls through to the
per-expert loop. The earlier guesses were right that the conditions hold; the
code just never evaluates them for a uniform pack.

**Which path runs** (`tools/moe_trace.py`, one cold 600-token prefill, 48 MoE
layers):

| | shipped | patched |
|---|---:|---:|
| `support_fused` / fused buffers | False / none | True / 256 rows |
| per-expert graph launches (`run_single_expert`) | 23,687 experts | 0 |
| batched reconstruct | 2,036 experts in 371 groups | 5 experts |
| fused `exl3_moe` launches | 0 | 265 (25,711 experts) |
| `index_add_` / `index_select` / `mul_` in MoE scope | 25,103 / 29,363 / 25,103 | 0 / 12 / 0 |
| MoE share of prefill (profiler, GPU) | 2.95 s of 5.99 s | 0.73 s of 2.28 s |

Every one of the ~23k `index_add_` calls and ~52k GEMVs in the old profile was
MoE. GatedDeltaNet, attention and the n-gram layer are small (<0.1 s each).

**One MoE layer alone** (layer 24, real hidden states, `moe_trace.py`):

| rows | experts touched | weight-read floor | per-expert path | fused path | parity (max rel) |
|---:|---:|---:|---:|---:|---:|
| 16 | 53 | 0.4 ms | 4.0 ms | 1.4 ms | 5.5e-5 |
| 64 | 148 | 1.2 ms | 11.6 ms | 3.6 ms | 5.6e-5 |
| 600 | 316 | 2.5 ms | 31.5 ms | 9.6 ms | 9.0e-5 |
| 2,048 | 445 | 3.6 ms | 53.1 ms | 19.9 ms | 9.3e-5 |

**The fix** is `patch_exllamav3_fused_moe.py`: it moves the block back where
upstream has it. `EXL3_MOE_FUSED_UNIFORM=0` restores the fork's behaviour. It
is in the image and on by default since T4 (below).

Engine benchmark (`run_engine_bench.sh`, chunk 8192, TTFT in s; "repeat" is a
second prompt of the same size, i.e. without one-time tuning):

| | cold 600 | 600 repeat | cold 3k | cold 12k | cold 20k | cold 40k | follow-up 20k + 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| J (shipped) | 3.92 | 2.77–2.96 | 6.19 | 16.7 | 24.8 | 46.5 (861 tok/s) | 3.30 |
| J + fused patch | 2.38 | 1.18–1.23 | 4.02 | 12.3 | 19.9 | 38.4 (1,043 tok/s) | 1.35 |

**Correctness gates** (T4's list, run now so the decision has them):

- MoE layer parity on real hidden states: max relative difference ≤ 9.3e-5
  (fp16 noise), table above.
- Greedy 400 tokens, code / DevOps / prose (`tools/greedy_ab.py`): two runs
  of the *unpatched* engine already diverge from each other at tokens 44 / 108
  / 34 (dynamic drafting and batched verify are not bit-reproducible). Patched
  vs unpatched diverges at 44 / 108 / 122, no earlier than that noise;
  acceptance 74 / 63 / 60% vs 76 / 67 / 61% (the unpatched pair: 76 / 67 / 61%
  vs 74 / 68 / 52%).
- Needle at 128k (`ctxfill.py`, 8-bit KV): found, patched.
- Decode: single-session decode (verify batch ≤ 8) goes through `run_bszN`
  either way, so it is untouched. With three sessions decoding at once the
  verify batch exceeds 8 and would move from the per-expert loop to the fused
  kernel; not measured yet.

### The one-time slowness after a start: kernel tuning that was thrown away

The ~20 s first request and the extra ~1.2 s on the first prompt of a new size
are exllamav3's GEMM autotuner (and Triton's) running. Both keep results on
disk under `~/.cache`, which died with every container. `start_tabby.sh` now
keeps it in the `qwen38-tabby-cache` volume. Second start: first request
19.4 → 1.6 s, first 600-token prompt 2.4 → 1.2 s. No warm-up request.

### T3 step 3: reconciling with the README's faster numbers

- The README's fork numbers (~900 tok/s at 4k, ~1,150 at 128k, `ctxfill.py`)
  are from 2026-09-17, on 523ecd3, before the regressing commit. The patched
  329e051 matches 523ecd3 exactly when both run in the image: `ctxfill.py`
  K=128 prefills at 922–926 tok/s patched and 915 tok/s on an image built from
  523ecd3. The remaining gap to 1,150 is environmental (that host venv no
  longer exists) and could not be reproduced.
- `bench_native.py`'s 1,129 tok/s used a prompt of one sentence repeated,
  which routes to few experts; not comparable with varied prompts.
- One change at a time from the shipped config, fused path on, cold 20k / 40k
  tok/s: chunk 8192 1,032 / 1,047; **chunk 16384 1,062 / 1,070**; chunk 4096
  990 / 983; `EXL3_MOE_COOP_WIDE=0` 1,035 / 1,046; `VISION=0` 1,040 / 1,051;
  fused rows 128 1,000 / 1,008; fused rows 512 1,037 / 1,033;
  `EXL3_NGRAM_STREAM=0` no change (also at 128k). Batch size is not a lever
  for prefill of one prompt. Stock 1.5.0 was not built: its aarch64 patch
  tool (`vllm-exl3/tools/patch_exllamav3_aarch64.py`) is not on the Spark.

**Size of what is left.** With the fused path, the MoE layer is still 3–5x its
weight-read floor (600 rows: 9.6 ms vs 2.5 ms; 2,048 rows: 19.9 vs 3.6 ms), so
48 layers spend ~0.46 s of the 1.14 s 600-token prefill in MoE. Reaching the
≥1,100 tok/s long-prompt target needs work on the fused kernel itself
(option (b): its tiles at ~12 rows per expert), not configuration.

### T1: three full contexts

`cache_size: 786432`, `sysmem_recurrent_cache: 8192` (one checkpoint is ~112
MiB; a ~115k prompt leaves ~11, a 262k one ~15). `max_batch_size` stays at
TabbyAPI's default of 4. `tools/three_sessions.py` through the API: three
~115k-token conversations, then two interleaved rounds of follow-ups — all six
follow-ups 99% cached, 3.7–4.6 s TTFT (one at 9.0 s while an image build ran
beside it). Memory: the server holds 71.5 GiB of device allocations plus
4.2 GiB of host memory (~81 GB) after the test; the system-wide low point of
MemAvailable was 34 GiB of 122.

### T2: speculative decoding for sampled output — no change

`tools/draft_sweep.py`: `draft_num_tokens` 3–7 x `EXL3_DRAFT_CONFIDENCE`
0.4–0.8, dynamic drafting, the qwen38 preset's sampling (temperature 1.0,
top_k 20, top_p 0.95), 320 tokens, 10 samples per cell and prompt class.
Decode tok/s, median [range], shipped cell (5, 0.6) first:

| | code | prose | tool call (pi-style edit) |
|---|---:|---:|---:|
| **5 / 0.6 (shipped)** | 70.0 [62.9–73.8] | 47.5 [44.4–53.2] | 81.4 [79.2–85.5] |
| 7 / 0.4 | 72.0 [64.8–76.9] | 46.1 [45.5–50.4] | 85.4 [83.1–87.2] |
| 7 / 0.5 | 70.3 [63.5–75.5] | 46.3 [43.8–48.4] | 88.1 [73.1–92.2] |
| 5 / 0.4 | 74.1 [56.1–78.9] | 47.4 [42.4–50.2] | 82.1 [78.2–85.0] |
| 6 / 0.7 | 69.8 [65.3–72.8] | 46.5 [45.1–49.1] | 85.3 [76.2–88.2] |
| 3 / 0.5 | 68.3 [62.4–71.6] | 46.9 [45.3–49.5] | 73.7 [64.9–76.8] |

No cell is ≥ 5% faster with non-overlapping ranges on any class, and the ones
that gain on tool calls lose on code or prose, so the setting stays. Prose is
flat at 45–48 tok/s in every cell; tool calls gain most from longer drafts
(acceptance ~85–90%), which is T5's case. Full table:
`logs/bench_draft_sweep.log` on the Spark.

### Status after this round

| | Before | Deployed now | With the fused patch on (awaiting approval) | Target |
|---|---:|---:|---:|---:|
| Follow-up: cached history + ~600 new | 3.3–3.6 s | 3.5 s (API, 25k) | 1.35 s (engine, 20k) | ≤ 1.5 s |
| Cold 600-token prompt | 3.9 s | ~2.8 s after the first of its size (tuning now cached) | 1.2 s (engine) | ≤ 1.5 s |
| Cold prefill, long prompts | ~840 tok/s | ~840 tok/s | 1,047 tok/s at 40k (1,070 with chunk 16384) | ≥ 1,100 |
| Full-context sessions cached | 1 | 3 | 3 | 3 |
| Decode, 400-token code answer | ~56 tok/s | 45–59, median ~53 (6 samples, sampled output) | unchanged path | not lower |
| First request after a start | ~8 s | 1.7 s | | |

### T4: fused MoE kernel shipped (2026-09-24, approved)

`EXL3_MOE_FUSED_UNIFORM=1` is now the image default. Gates: the ones under T3
above, plus decode through the API with the fix off / on
(`tools/concurrent_decode.py`, 400 tokens, default sampling, 4 rounds):

| | off | on |
|---|---:|---:|
| one session | 53.9 [50.0–56.6] tok/s | 54.2 [47.9–55.9] tok/s |
| three at once, aggregate | 25.7 [24.0–26.3] tok/s | **54.9 [52.2–55.6] tok/s** |

(With three sessions the verify batch exceeds the 8-row decode kernel and fell
through to the per-expert loop, ~9 tok/s per session.)

Through the API with the fix on: cold ~600-token prompt 1.27 s (server 1.07 s),
~700 tokens 1.46 s; follow-up on 25k + 853 new 1.59 s; 115k cold prompts
1,035–1,041 tok/s; follow-ups on 115k conversations 1.3–1.7 s server-side,
all 99% cached; memory under load 79.1 GiB system-wide (72 GiB device + 4 GiB
host for the server). Chunk 16384: 1,050–1,059 tok/s at 115k (+1.7%) for
+1.6 GiB (80.7 GiB system-wide); not adopted.

Against the targets: follow-up ≤ 1.5 s — met at ~20k on the server side, 1.6 s at
the client with 850 new tokens; cold 600 ≤ 1.5 s — met; three cached sessions —
met; decode — not lower (and 2.1x with three sessions); long prompts ≥ 1,100
tok/s — **not met**, ~1,040. What is left there is the fused kernel itself.

**Long prompts: what is left.** Knobs of the fused kernel on cold 40k (tok/s):
default 1,026; `EXL3_MOE_TILE_N=128` 1,026; `EXL3_MOE_FUSED_DET=0` (atomic
accumulation) 999; `EXL3_MOE_MTILE=0` 949. None helps. One 8,192-token chunk
takes 7.95 s; one MoE layer at 8,192 real rows returns to the host in ~3 ms
and finishes on the GPU in ~56 ms, so the MoE is GPU-bound, not host-bound:
48 layers x 56 ms = 2.7 s (~34% of the chunk). At 8,192 rows ~105 experts per
layer exceed the fused tier's 256 rows and take the batched-reconstruct tier
(5,038 experts in 451 groups per chunk). The layer's floors are ~4 ms (weights)
and ~8 ms (fp16 tensor math), so the kernel runs at ~7x its compute floor,
mostly the in-kernel trellis decode. +6% to reach 1,100 tok/s means making
the MoE ~20% faster (or the rest of the chunk ~10% faster): CUDA work in
`exl3_moe_kernel.cuh` / the batched-reconstruct tier, estimated at days, with
the per-layer parity and greedy gates above. Not started; the owner's call.

### Kernel work for long prompts (2026-09-24, approved)

**Tried on the fused MoE kernel, no gain:**

- A 128-row GEMM tile (a new `exl3_moe` instance with two shared-memory
  stages, for experts with more than 64 rows), to decode each expert's weights
  half as often. Bit-identical output, but 0.65–0.84x the speed at 512–8,192
  rows: at the 128-register budget of the kernel's 512-thread blocks it spills
  (696 B of stack vs 256 B for the 64-row tile). Not shipped; the premise that
  weight decode dominates was wrong.
- SMs per expert group (`MOE_SMS_PER_EXPERT`, 8 in the fork) made a run-time
  setting and swept 4 / 5 / 6 / 8 / 12 / 16: 4 is best by 3–4% on the MoE layer,
  ~1% end to end. Not shipped.

**Where an 8,192-token prompt's time actually goes** (`tools/module_times.py`,
device sync around every module, exclusive time, 8.4 s total): MoE 3.1 s,
the transformer blocks' own code 2.2 s (48 x 46 ms), attention 1.1 s,
GatedDeltaNet 0.7 s, dense projections ~0.8 s. The blocks' own time is the
hyper-connection mixer (`GatedResidual.mix`, twice per block): for more than 32
rows the fork finishes it in torch,

    mixed = (sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(-2).half()

which builds four fp32 temporaries of R x 4 x 2560 (~335 MB each at 8,192 rows).

**Shipped: `patch_exllamav3_gr_collapse.py`**, a CUDA kernel (`ext.gr_collapse`)
that does it in one pass over fp16 inputs. It is **bit-identical** to the torch
expression: same summation order (torch's mean over 4 is sequential), same
sigmoid (`1 / (1 + expf(-x))` with libdevice's `expf`, called by name because
the extension is built with `--use_fast_math`), IEEE multiply / add / divide.
On 40 real calls (107–8,192 rows): 0 of 590M output elements differ; 2.9x
faster (532 → 184 ms total). `EXL3_GR_COLLAPSE=0` restores the torch path.
Engine benchmark, chunk 8192, fused MoE on (the first variant of the kernel,
same speed as the shipped one):

| | cold 600 | cold 3k | cold 12k | cold 20k | cold 40k | follow-up 20k + 600 |
|---|---:|---:|---:|---:|---:|---:|
| without | 1.20 s | 3.63 s | 12.3 s (976 tok/s) | 19.9 s (1,009) | 38.3 s (1,046) | 1.36 s |
| with `gr_collapse` | 1.12 s | 3.22 s | 10.6 s (1,139) | 16.9 s (1,184) | 32.3 s (**1,238**) | 1.26 s |

Through the API after deploying (2026-09-24): cold 24.5k prompt 20.8 s (24.2
before, ~1,180 tok/s); three cold ~115k prompts 1,228–1,233 tok/s (1,035–1,041
before); follow-ups on them 1.15–1.57 s server-side, 99% cached; cold 573-token
prompt 1.11 s; follow-up 25k + 853 new 1.52 s; decode 54.0 tok/s one session
(10 runs, 49.7–60.6), 53.9 three at once; memory 79.0 GiB system-wide. The
collapse kernel only runs above 32 rows, so decode (≤ 18 rows here) cannot
touch it.

### Where the targets stand (2026-09-24)

| | Before | Now (API) | Target |
|---|---:|---:|---:|
| Follow-up: cached history + ~600–900 new | 3.3–3.6 s | 1.5 s at 25k (1.2–1.6 s server-side at 115k) | ≤ 1.5 s |
| Cold 600-token prompt | 3.9 s | 1.1 s | ≤ 1.5 s |
| Cold prefill, long prompts | ~840 tok/s | ~1,230 tok/s | ≥ 1,100 |
| Full-context sessions cached | 1 | 3 | 3 |
| Decode | ~56 tok/s (one sample) | 54 tok/s (median of 10), 54 with three sessions (was 26) | not lower |

Left open: T5 (prompt-lookup drafting) needs a recorded pi session.

### Sparse attention: 8-bit K/V staged once per layer (2026-09-25, bit-identical)

Plan: `PLAN_A_PREFILL.md` (A0–A2 findings there). With the 8-bit cache, the QSA
gather kernel (`_qsa_sparse_split_kernel`, 17% of a 32k prefill) dequantized
every K/V tile it gathered, so each cached token once per query row selecting
it (~2,048 per row). `patch_exllamav3_qsa_stage.py` adds a staging kernel that
dequantizes the sequence's positions once per layer (same `_qc_load_v`
expression, rotated domain) and runs the gather on fp16. Only for one sequence
and when rows x index width ≥ 4 x positions (so never decode or MTP verify).

- Kernel, captured prefill inputs (`tools/qsa_capture.py`, `tools/qsa_ab.py`):
  204 → 100 ms over 4 calls, 0 of 122,873,856 output elements differ.
- End to end (`tools/prefill_parity.py`, off/on in one process): logits of the
  first 4 new tokens bit-identical at 3k, 9k, 20k.
- Engine benchmark, chunk 8192: cold 3k 3.22 → 3.03 s, 12k 10.6 → 9.73 s,
  20k 16.9 → 15.6 s (1,281 tok/s), 40k 32.3 → 29.7 s (**1,349 tok/s**),
  follow-up 20k + 600 1.26 → 1.23 s. Cold 600 unchanged (dense attention).
- API: cold 24.5k 20.8 → 19.2 s; three cold ~115k prompts **1,329–1,331 tok/s**
  (1,228–1,233); follow-ups on them 1.56–1.94 s at the client; cold 573 tokens
  1.13 s; follow-up 25k + ~680 new 1.48 s; decode one session 54.4 tok/s
  median of 10 (48.8–57.7), three sessions ~54 aggregate; memory 79.5 GiB
  system-wide (79.0 before; staging holds n_tok x 2 KiB per layer transiently).

Tried on the same kernel, no gain on sm_121: vLLM 0.30.0's prefill launch
(1 warp) 0.59x, `BLOCK_N=64` 0.87x (also not bit-identical), 2 warps / 3 stages
1.12x but not bit-identical; 8 warps 0.39x. Rollback image:
`qwen38-exl3-tabby:pre-qsastage`.

### GatedDeltaNet prefill without copies (2026-09-25, bit-identical)

`patch_exllamav3_gdn_nocopy.py`: the chunked prefill path copied the same data
four times (fp16 → bf16 transpose for the conv, FLA's `input_guard` on the q/k/v
views, `torch.cat` of one output); ~0.29 s of kernel time per 8k chunk. The
conv kernel now reads the fp16 projection in place (bf16 rounding on load) and
writes q/k/v contiguous. `prefill_parity.py TOGGLE=EXL3_GDN_NOCOPY`: logits of 4
new tokens bit-identical at 200 / 3k / 9k / 20k. Engine benchmark, alternating
off/on twice in one window (chunk 8192): 3k 3.04/3.04 vs 3.01/3.06 s, 20k
16.37/16.23 vs 16.13/16.03 s, 40k 30.51/31.94 vs 30.19/30.53 s: ~1–2%, less than
the profile's 4% (the copies partly overlap). API after deploying: cold 24.5k
19.3 s, follow-up 25k 1.51 s, cold 573 tokens 1.12 s, decode 52.5 and 54.9 tok/s
(two runs of 10), three sessions 53.2. Rollback: `:pre-gdnnocopy`.

Note for engine A/Bs: an in-process off/on comparison where "off" always runs a
new prompt size first overstates the gain (one-time kernel tuning lands on
"off"); alternate whole runs instead.

### Time to first token outside the engine (2026-09-25, A3g)

Client TTFT minus TabbyAPI's "first token" (which starts at job enqueue): ~0.18 s
on a 90-token prompt and 0.27–0.31 s on 115k follow-ups. Where it goes:

- ~0.18 s, any prompt: the model's first tokens are reasoning markup the
  reasoning parser consumes, so the first *visible* delta arrives 3–4 decode
  steps (~45 ms each) after the server's first token (SSE trace). That is model
  output, not overhead; a client measuring any chunk would see it earlier.
- Chat template rendering: ~1 ms at 115k tokens. Not a cost.
- Tokenizing the whole conversation every turn: ~0.16 s at 115k (0.03 s at
  25k). **Shipped `patch_tabbyapi_encode_cache.py`** (`TABBY_ENCODE_CACHE=1`):
  reuse the ids of a shared prefix up to its last `<|im_start|>`, encode only
  the rest. `tools/encode_cache_test.py`: 96 growing conversations (20k–200k
  characters, code/CJK/emoji/markup-like text), 0 mismatches, tokenization
  4.03 → 0.75 s. Through the API in `verify` mode: 0 mismatches. The ~0.13 s it
  saves at 115k is within run-to-run noise of the follow-up TTFT (client mean
  1.68 s over six follow-ups vs 1.76 s before; server 1.14–1.63 s). Memory 78.1
  GiB system-wide. Rollback: `:pre-enccache`.
