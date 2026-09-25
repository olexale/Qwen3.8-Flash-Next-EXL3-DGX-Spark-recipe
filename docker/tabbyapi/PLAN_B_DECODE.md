# Track B: faster single-session decode on the Spark (TabbyAPI, exllamav3 fork)

Written 2026-09-24 for the session that picks this up. Read the whole file
before starting; then read `OPTIMIZATION_PLAN.md` ("Results (2026-09-24)", T2
and T5) and the main `README.md` sections "Levers that are closed (native
engine)" and "Where the time goes" — they record days of decode work that must
not be repeated.

## Goal

Make **token generation (decode) for one session** faster, which is how the
owner uses the server: pi coding sessions, one per project, output that is
mostly code and tool calls (edits that quote existing code). Output must not
change.

| | Now (2026-09-24) | Target |
|---|---:|---:|
| Decode, 400-token code answer, default sampling, one session (API, `concurrent_decode.py N=1 REPS=10`) | 54.0 tok/s median (49.7–60.6) | ≥ 60 tok/s |
| Decode on pi-style edit tool calls (engine, `draft_sweep.py` "tool" workload, draft 5 / conf 0.6) | 81.4 tok/s median (79.2–85.5) | ≥ 100 tok/s |
| Decode on prose (same) | 47.5 tok/s (44.4–53.2) | not lower |
| Three sessions at once (API, `N=3 REPS=5`) | 53.9 tok/s aggregate | not lower |
| Prefill, TTFT | see `PLAN_A_PREFILL.md` | not lower |

Honest expectation: decode here is limited by weight-decoding cost per
verify round, and the drafter's acceptance; ~60–65 tok/s on general code is
realistic, much more only on verbatim-copy output (tool calls). The targets
are what to report against; they are **not** a reason to stop (see "When to
stop").

## Constraints (the owner's; do not bend them)

- **Model behaviour must not change.** Speculative-decoding changes are fine
  because the fork only keeps a drafted token if it equals the target's own
  sample, so the output distribution is unchanged; everything else must be
  bit-identical or pass the gates below. No changes to sampling (`qwen38`
  preset), quantization, KV precision.
- **GPU clocks stay as they are** (`sudo nvidia-smi -lgc 0,1600`, graphics
  clock ~1,580–1,600 MHz). Never change or propose changing them.
- **No GPU rental, no drafter training** (the owner declined).
- **Do not run the vLLM container or image** (`qwen38-flash`,
  `qwen38-flash-next-exl3-vllm`); reading files out of the image is fine.
- **Concurrency is not the priority**: the only concurrent traffic seen was the
  owner's digest bot (a background batch job). Do not regress it, but do not
  optimise for it.
- **TabbyAPI (`qwen38-tabby`) may be stopped and restarted**: announce each
  downtime window in the chat first, keep windows short (5–15 min), restart
  right after, leave it on the best configuration. Engine benchmarks need it
  stopped (two model copies do not fit).
- **No warm-up request** added to startup.
- Engine changes ship as **patch scripts in `docker/tabbyapi/`** applied at
  image build time, failing on a missing anchor, each behind an `EXL3_*`
  environment variable (T5: `EXL3_PLD=1`, off by default until the owner
  approves). Forward new variables in `start_tabby.sh`, document them in
  `README.md`.
- **Small commits to `main`, pushed.** Never commit `.env`. Commit trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## When to stop and ask the owner

**Meeting the targets is not a reason to stop.** Keep working through the task
list (and ideas found on the way, e.g. from the B1 profile) while any
candidate still has measured promise. Stop only when one of these holds, then
report with numbers and what was tried:

- the remaining ideas are exhausted, or each remaining one is measured (or
  sized from the B1 profile) at less than ~2% on its workload;
- the next step needs the owner's approval or data (below).

Ask the owner (and keep working on other items meanwhile, if any):

- at the start of B3, for a recorded pi session (see B3); keep working on the
  synthetic workload meanwhile;
- before shipping anything not bit-identical / not lossless by construction;
- before a multi-day kernel rewrite.

Never bend a constraint to reach a number.

## Setup

| | |
|---|---|
| Machine | NVIDIA DGX Spark, GB10 (sm_121, 48 SMs), 128 GB unified memory, aarch64, 10 Cortex-X925 (cores 5–9, 15–19, used by TabbyAPI) + 10 A725 |
| SSH | `ssh gx10-b2fe.local` (user `ole`) |
| Repo on the Spark | `~/dev/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe`; locally `/Users/ole/dev/test/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe`. Commit locally, push, `git pull --ff-only` on the Spark |
| Model | `~/models/Qwen3.8-Flash-Next-EXL3` (EXL3 3.05 bpw, `*.native` config/index are the originals, the scripts mount those) |
| Engine | [vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051) + `patch_exllamav3_checkpoints.py`, `patch_exllamav3_fused_moe.py`, `patch_exllamav3_gr_collapse.py` |
| Server | TabbyAPI `2186cdb`, image `qwen38-exl3-tabby:latest`, container `qwen38-tabby`, port 18300; rollback tags `:pre-20260924`, `:pre-grcollapse` |
| Draft config | `config.yml`: `draft_mode: mtp`, `draft_num_tokens: 5`, `dynamic_draft: true`, `EXL3_DRAFT_CONFIDENCE=0.6` (image env), `EXL3_MTP_HEAD_N=65536` (draft uses a 64K-column slice of the output head) |
| GB10 knobs in the image | `EXL3_INT8_GEMV=0`, `EXL3_MOE_COOP_WIDE=1`, `EXL3_GR_INT8=1`, `EXL3_NGRAM_STREAM=1` (see README table) |

Model shape: 48 layers (36 GatedDeltaNet, 12 full attention), MoE 512 experts
top-10 (intermediate 640), hidden 2560, 4 hyper-connection streams mixed twice
per block, one MTP layer. A decode round = up to 5 sequential MTP draft
forwards + one target forward verifying 1 + drafts rows. Verify batches of ≤ 8
rows use the fused decode kernels (`BC_BlockSparseMLP.run_bszN`,
`gr_mix(_int8)` for the mixers); larger ones (several sessions) go through the
prefill-style fused MoE kernel.

## Tools (`docker/tabbyapi/tools/`)

- `draft_sweep.py` (via `SCRIPT=draft_sweep.py run_engine_bench.sh <name>`):
  decode tok/s and acceptance, sampled (preset sampling), per workload
  (`code`, `prose`, `tool` = pi-style edit tool call on a file in context),
  `NDTS`, `CONFS`, `REPS`, `NTOK`. One Generator per draft length; a
  calibrator per confidence. The template for any decode A/B.
- `greedy_ab.py` — greedy 400-token outputs + acceptance of two variants
  (`COMPARE=a,b`), writable `/out` (`~/scratch/out`, mode 777).
- `concurrent_decode.py` — decode through the running API with `N`
  sessions, `REPS` rounds (`min_tokens` forces full length).
- `module_times.py` — per-module exclusive time with device syncs (built for
  prefill; for decode, syncs distort CUDA-graph-captured parts — see B1).
- `api_bench.py`, `engine_bench.py`, `run_engine_bench.sh` — see
  `PLAN_A_PREFILL.md` for the dev loop (in-place extension build in
  `~/scratch/exl3dev`, mount the `.so` and changed `.py` files over the
  installed ones, ship as a patch + image rebuild into a new tag).
- `scripts/exl3_native/tuning/`: `kern_rounds.py` (kernels per decode round),
  `accept.py` (per-position acceptance), `moepath.py`, `ab_greedy.py`,
  `diverge.py` — older harnesses that expect `~/exllamav3` and
  `~/models/...`; run them in the image with those paths mounted.

## What is known (do not re-measure unless something changed)

- Decode round cost grows with verified rows: ~37 ms at 2 draft tokens, 52 ms
  at 5, 86 ms at 9 (per round), because each extra token pulls in its own
  experts (top-10 of 512; unique experts touched scale ~1:1 with rows).
- Profile (older build, no draft, 32k context): EXL3 dense GEMV/GEMM 47%,
  fused MoE 34%, bf16 GEMMs (mixers, router) 5%, GDN 2%, norms 1%, small
  elementwise 11%. Decode is bound by trellis decoding in the EXL3 kernels
  (3–5× the time the weight bytes would need at 273 GB/s) — that is ALU work,
  so the clock cap may matter for decode more than it seems; measure, do not
  assume.
- The decode round is already CUDA-graph-captured inside the engine (GDN and
  attention linear chains); wrapping more in graphs fails (nested capture).
- Closed levers (README): deeper speculation (`-ndt` 6–8), mixer
  restructuring for R = 6, graph capture of the round, n-gram drafting
  *instead of* MTP, dequant-once MoE for decode, row-batched fp16 mixers.
- T2 (2026-09-24): draft 3–7 × confidence 0.4–0.8, sampled, 10 samples/cell:
  nothing beats 5 / 0.6 with non-overlapping ranges. Tool calls accept 84–90%
  of drafted tokens and are the only class that gains from longer drafts
  (7 / 0.5: 88.1 [73.1–92.2] vs 81.4 [79.2–85.5]); prose is flat at 45–48.
- Greedy decode is not bit-reproducible run to run (batched verify / dynamic
  drafting): two runs of the same image diverged at tokens 82 and 148 on
  DevOps/prose prompts. Use that spread as the noise floor.

## Tasks

Order: B1 → B2 → B3. B2 items only where B1 shows time.

### B1. Where a decode round goes now (one ~15 min window)

The last per-kernel profile predates the fork's GB10 work, the fused-MoE
patch and `gr_collapse` (prefill-only, so decode should be unchanged). Build a
decode profile on the current image:

- Generate 400 tokens (code prompt, preset sampling, draft 5 / 0.6) inside
  `torch.profiler` with `record_function` scopes around: the MTP draft chain
  (`Generator.iterate_draftmodel_mtp_gen`, generator.py ~line 700), the target
  verify forward, sampling/acceptance (`iterate_gen`), and the host work in
  between. Report per round: GPU time per scope, GPU idle time between scopes
  (host bubbles), kernel table per scope.
- Split the target verify by module family (MoE `run_bszN`, dense EXL3
  GEMV/GEMM, mixers `gr_mix_int8`, GDN, attention, lm_head) with the kernel
  names from the profile.
- Tokens per round and acceptance from the job counters.

Deliverable: a table in this file ("Findings") with ms per round by part and
the idle share. Candidates follow from it.

### B2. Levers to check against B1

a. **Host bubbles**: GPU idle between draft steps / before verify / after
   sampling (per-round `.cpu()` syncs, Python in the dynamic-draft
   calibrator — `draft_confidence.py`, `c = conf.float().cpu()` per draft
   step in the non-device path). The device-resident draft chain
   (`EXL3_MTP_DEVICE_DRAFT`) is disabled when a calibrator is active
   (`dev_draft = _MTP_DEVICE_DRAFT and self.draft_calibrator is None`), i.e.
   **exactly in the deployed config** (dynamic drafting on). Check whether
   the dynamic-draft path could stay on-device (confidence threshold computed
   on the GPU, one sync per round instead of per step). Lossless.
b. **MTP draft chain cost**: 5 sequential MTP forwards (one layer + MoE +
   a 64K-column head slice); measure per-step ms. If significant: capture the
   draft step in a CUDA graph, or trim the head further (`EXL3_MTP_HEAD_N`
   32768 — check acceptance), all lossless.
c. **Dense EXL3 GEMV at 6 rows** (47% in the old profile): kernel choice per
   shape (the coop autotuner caches choices in the `qwen38-tabby-cache`
   volume; check what it picked), `EXL3_INT8_GEMV` was measured slower on
   GB10 — only revisit with a reason.
d. Anything else B1 shows above ~5% of a round.

### B3. T5: prompt-lookup drafting alongside MTP

Full motivation in `OPTIMIZATION_PLAN.md` "T5". Summary: when the text being
generated repeats text already in the context (pi's `edit` tool calls quote
`oldText` verbatim; file rewrites), draft the continuation of the match
(10–20 tokens) instead of the MTP head's ≤ 5; otherwise draft with MTP as
now. Verification is unchanged, so outputs are unchanged.

Implementation notes:

- The generator dispatches draft modes in `iterate()` (generator.py ~line
  540: dflash / mtp / draft model / n-gram); the fork asserts
  `not ngram_match_min` when a draft model is set (line ~160). The hybrid
  belongs inside `iterate_draftmodel_mtp_gen` (~line 700) and must keep the
  batched verify, the device-resident draft chain and dynamic drafting.
- Match search: per job, find the longest suffix of the last N (e.g. 8–16)
  generated tokens that occurs in the prompt + output ids (a rolling hash or
  suffix index built once per job, updated per round; do not scan 100k
  tokens per round in Python).
- Draft length: long only on long matches, capped (verify cost grows with
  rows: 52 ms at 5 drafts, 86 ms at 9). Batch size matters: with several jobs
  the verify rows add up; use lookup drafts only for the job(s) that have a
  match, MTP for the others.
- Rollback memory: GatedDeltaNet keeps per-token history for rejected drafts,
  sized by `Cache(max_history)` × batch; TabbyAPI sets `max_history` from
  `draft_num_tokens` (`/app/backends/exllamav3/model.py` ~line 723). A cap of
  16–20 needs a bigger `max_history`: estimate the memory (the three-session
  budget is ~80 GiB system-wide; `max_batch_size: 4`) and decide whether the
  cap or the batch size gives.
- Workload: start with a **synthetic pi-style edit workload** (extend
  `draft_sweep.py`'s `tool` prompt: a real source file in context, tasks that
  make the model call `edit` with multi-line `oldText`/`newText`, and a
  whole-file rewrite). For the keep/remove decision, a **recorded real pi
  session** is required: ask the owner at the start of B3 whether you may
  set `logging.log_prompt: true` temporarily (restart needed) for one of
  their normal sessions, or whether they can export one from pi; it is their
  data — store it only on the Spark outside the repo, never commit it.
- Keep only if edit-heavy turns get clearly faster and code/prose/chat are
  not slower than 2% (median of ≥ 10 samples, non-overlapping ranges). If it
  fails, remove the patch and record the negative result in `README.md` next
  to the other closed levers.

## Gates for every change

1. **Lossless**: greedy `greedy_ab.py` old vs new (twice each, as in
   `PLAN_A_PREFILL.md` "Gates"): divergence no earlier than old-vs-old,
   acceptance within the same spread. For drafting changes, additionally
   check that sampled output statistics are unchanged by construction
   (only drafts that equal the target's sample are kept — read the verify
   code path you touch).
2. **Speed**: `draft_sweep.py` (code / prose / tool, ≥ 10 reps, median and
   range) old vs new; then through the API after deploying:
   `concurrent_decode.py N=1 REPS=10` and `N=3 REPS=5`, `api_bench.py`.
3. **No regressions**: prefill/TTFT (`api_bench.py`, `~/scratch/cold600.py`),
   memory under `three_sessions.py` ≤ ~80 GiB, 128k needle (command in
   `PLAN_A_PREFILL.md`).

## Reporting

After each shipped change: update `README.md` "What to expect" and the
`EXL3_*` table, add a dated section to `OPTIMIZATION_PLAN.md` "Results"
(numbers; what was tried and did not help), commit, push, pull on the Spark.
Raw logs stay in `logs/` on the Spark (gitignored).

## Findings (2026-09-25)

### B1: where a decode round goes now

`tools/decode_profile.py` (engine, code prompt, 400 tokens, preset sampling, draft 5 /
conf 0.6 with a calibrator, as deployed). Unprofiled 69.8 tok/s median of 5
[67.5–72.8]; the profiled run 70.6 tok/s, so the profiler does not distort it. 104
decode rounds, 3.8 tokens per round (3.5 drafted, 79% accepted). Logs:
`logs/bench_b1_prof.log`, `logs/bench_b1_prof_mod.log` (with `MODSCOPES=1`: the
verify forward split by block submodule).

| Part of a round | GPU ms | share |
|---|---:|---:|
| **Target verify forward** (1 + ~3.5 rows) | **43.2** | 81% |
| – MoE (routed experts `exl3_moe_coop_a/b/rot` 15.6, shared expert 2.2, router 1.3) | 19.2 | 36% |
| – GatedDeltaNet (qkv/z `exl3_mgemm` 4.8, recurrent rule 2.4, out_proj 2.3, rest 0.8) | 10.4 | 20% |
| – hyper-connection mixers (`gr_dots_i8` 3.6 + `gr_finalize_i8` 3.4, 96 sites) | 7.0 | 13% |
| – attention (12 layers: qkv 1.5, paged attention 0.9, o 0.7, rest 0.4) | 3.5 | 7% |
| – lm_head (full 248K head, 1 call) | 1.7 | 3% |
| – mixer apply, norms, rest | 0.7 | 1% |
| MTP draft chain (3.4 steps: block 2.2 + 64K head slice 1.6 + mixer) | 4.1 | 8% |
| MTP prefill of accepted positions | 0.45 | 1% |
| Sampling, acceptance, GDN state rewind (0.35) | 0.6 | 1% |
| **GPU idle** | **4.7** | **8.8%** |
| **Round (wall)** | **53.1** | |

Idle, by what the GPU was between (ms per round, gaps per round):

| Between | ms | gaps |
|---|---:|---:|
| kernels inside the verify forward | 2.6 | 1,140 (~2.3 µs each) |
| sampling / acceptance host work (`iterate_gen` after the forward) | 0.9 | 51 |
| kernels inside the MTP draft steps | 0.5 | 145 |
| draft-step readbacks (calibrator, per step) and draft → verify | 0.4 | 15 |
| MTP prefill of accepted positions | 0.2 | 27 |

What that says, for B2:

- **(a) Host bubbles are small.** The dynamic-draft per-step `.cpu()` path costs
  ~0.4 ms per round (0.8%); the whole host part of `iterate_gen` another ~0.9 ms
  (1.7%). The host runs well ahead of the GPU during the verify (10 ms of host
  time for 43 ms of GPU). The largest idle item is the ~1,140 gaps of ~2.3 µs
  between kernels inside the verify: GPU-side launch spacing, not host
  bubbles, and only fewer kernels would remove it.
- **(b) MTP draft chain: 4.1 ms per round (8%).** ~0.65 ms per step for the
  block and 0.47 ms for the 64K head slice (105 MB, at bandwidth). Halving the
  slice (`EXL3_MTP_HEAD_N=32768`) saves at most ~0.8 ms per round (1.5%), and
  fewer drafts land in the slice.
- **(c) Dense EXL3 GEMMs.** The qkv/z projections run at ~71% of the 273 GB/s
  spec, lm_head at bandwidth; the `out_proj` / `o_proj` GEMMs (32×128 tile, ~65
  µs for 9.8 MB) at ~55%, so at most ~0.7–1 ms per round (1.5–2%) there.
  The shared-expert GEMMs (1–2 MB) are latency-bound at 17–28 µs each.
- The routed-expert MoE is at ~82% of spec (README); the recurrent
  gated-delta rule reads and writes ~20 MB per layer (state plus the per-token
  rollback history) in 68 µs, at bandwidth; the mixers are a closed lever (README).
- No single item above is worth more than ~2% on code. The round is
  weight-streaming bound, so tokens per round is the lever: B3.

### B3: prompt-lookup drafting alongside MTP, synthetic workload

`patch_exllamav3_pld.py` (`EXL3_PLD=1`): per round, for a single active job, look up
the last 3 tokens of the sequence in an n-gram index of prompt + output (built once per
job, extended per round; last 4 end positions per n-gram); if the best match extends to
≥ `EXL3_PLD_MIN_MATCH` tokens, draft the ≤ `EXL3_PLD_MAX` tokens that followed it and skip
the MTP chain that round. The post-verify MTP prefill then also covers the round's first
position (no draft step wrote it). Verification is unchanged, so outputs are unchanged by
construction.

**Two harness fixes found on the way.** `draft_sweep.py`, `greedy_ab.py` and
`decode_profile.py` encoded prompts without `encode_special_tokens`, so the chat markup
reached the model as plain text (T2 and B1 above ran on such prompts; relative results
stand). And the old `tool` workload uses Qwen2.5-style JSON tool calls, where `oldText` is
a JSON string with escaped newlines and quotes, so it is not a verbatim copy of the file.
This model's template writes tool calls as `<function=edit><parameter=oldText>` blocks
with raw multi-line values. The new `edit` and `rewrite` workloads (a real ~110-line file
from the image in context, two `edit` calls / the whole file back through `write`) are
rendered with the model's own `chat_template.jinja`, thinking off.

**Verify cost by rows** (round wall time, from `ROUNDSTAT=1`): MTP rounds 37 ms at 2
rows → 60–64 ms at 6 (incl. ~1.2 ms per draft step); lookup rounds 59 ms at 6 rows,
66.5 ms at 8, **126 ms at 11, 144–149 ms at 16**. Above 8 rows the MoE leaves the fused
decode kernels (`MAX_BSZN = 8`) for the prefill-style fused kernel, which costs ~45 ms
more per round. So lookup drafts of 15 tokens lose (first try, cap 15 / min match 6:
edit 83 vs 91 tok/s, JSON tool calls 60 vs 78), and the cap has to be 7.

**Offline simulation** (dumped sequences of MTP-only runs as ground truth, lookup
acceptance = common prefix with the real continuation, measured round costs): cap 7 /
min match 8: edit x1.21, rewrite x1.27, code x1.00, JSON tool calls x1.04. With the
8-row limit lifted (cap 15 at the 8-row path's ~4.6 ms per row): edit x1.32–1.36,
rewrite x1.50.

**Engine A/B, cap 7 / min match 8** (`draft_sweep.py PLDS=0,1`, draft 5 / 0.6, preset
sampling, 10 samples per cell, in-process toggle; `logs/bench_pld_ab3.log`):

| tok/s, median [range] | MTP only | MTP + lookup | |
|---|---:|---:|---:|
| edit (2 `edit` calls, 243 tokens) | 91.3 [89.8–91.5] | **106.3 [106.2–106.4]** | +16% |
| rewrite (whole file, ~818 tokens) | 87.3 [86.9–88.1] | **107.3 [106.5–108.6]** | +23% |
| code | 70.9 [67.4–72.9] | 71.2 [67.6–74.1] | ±0 |
| prose | 46.9 [44.4–48.5] | 46.6 [43.1–51.2] | −0.6% |
| JSON tool call (old `tool`) | 76.4 [69.5–88.3] | 76.5 [56.4–91.0] | ±0 |

Lookup rounds: 7 drafts, 66.5 ms, 6.1 (edit) / 6.6 (rewrite) accepted, against MTP's
~64 ms for ≤ 5.9 tokens. Rollback history at cap 7: +2 x ~113 MB per batch slot
(+0.9 GiB at `max_batch_size` 4), only with `EXL3_PLD=1`.

### B3 continued: the 16-row decode MoE, adaptive length, the MTP gate

**16-row fused decode MoE** (`patch_exllamav3_bszn16.py`): the extension is built with
`MAX_BSZN 16` and the Python-side limit is a run-time setting, `EXL3_MOE_BSZN_MAX`
(default 8). The coop kernels already bound their slots at 256 and size their counters
from the Python buffers; split-k is off, so the launch geometry does not depend on the
scratch size. Verify rounds with 16 rows: 147 → 103 ms, 12 rows: 126 → 83 ms (the 8-row
path's ~4.6 ms per row continues). Parity (`tools/logit_dump.py`, separate processes):
with the limit at 16, the prompt prefill, 64 one-row decode steps and a 64-token MTP run
give bit-identical logits to the limit at 8; `greedy_ab.py` (`AGENT=1`) with the image at
8 vs the old image: identical 400 tokens on all five prompts. With the limit at 16, one
greedy_ab sequence diverged at token 50 (DevOps) after four other prompts in the
process, acceptance within 1.3 points; likely a different draft window → different verify
batch shape (the run-to-run effect noted under "What is known"), not pinned down.

**Adaptive length** (`EXL3_PLD_START=7`, `EXL3_PLD_MAX`): draft 7 lookup tokens, and up to
the cap right after a lookup round whose drafts were all accepted (a copy in progress).
Offline simulation: 7→15 edit x1.35, rewrite x1.49, code x1.00 (fixed 15: code x0.99).

**MTP-agreement gate** (`EXL3_PLD_GATE=1`, default): on a match, run MTP draft step 0 as
usual and switch to the lookup draft only if its first token equals the MTP head's (on the
host already in the dynamic-drafting path; one small readback otherwise). A disagreement
just continues the MTP chain, so a wrong lookup costs nothing. Greedy, engine, in process
(`logs/bench_gs_gate.log`, tok/s):

| | MTP only | lookup, ungated | **lookup, gated** | gated, min match 5 |
|---|---:|---:|---:|---:|
| code | 69.5 | 69.7 | 70.1 | 70.1 |
| code2 (greedy_ab's code prompt, repeats its code) | 74.9 | 73.6 | 74.3 | 73.7 |
| devops | 54.5 | 54.8 | 55.6 | 55.7 |
| prose | 49.3 | 50.7 | 49.4 | 50.0 |
| edit | 91.5 | 115.1 | **117.2** | 117.0 |
| rewrite | 87.2 | 115.7 | **124.0** | 123.2 |

Without the gate, `greedy_ab.py`'s code prompt was 4% slower (lookups on repeated code
fragments that the model does not continue verbatim).

**Cap** (greedy, gated, 16-row path, `logs/bench_gs_caps.log`):

| tok/s | MTP only | cap 7 | cap 9 | cap 11 | cap 15 |
|---|---:|---:|---:|---:|---:|
| edit | 91.6 | 106.4 | 108.6 | **119.2** | 117.5 |
| rewrite | 87.1 | 105.3 | 111.4 | 118.7 | **123.2** |
| code2 | 73.6 | 75.0 | 74.7 | 76.3 | 74.8 |
| rollback history, extra memory (batch 4 / batch 3) | | +0.9 / — | +1.8 / +1.0 | +2.7 / +1.4 | +4.5 / +2.7 GiB |

**Sampled, 10 samples per cell** (adaptive 7→15 ungated, 16-row path; `logs/bench_b16_ab.log`):
edit 90.9 → 120.1, rewrite 86.8 → 124.3, code 71.7 → 69.2 (ranges overlap: [66.3–77.4] vs
[66.7–73.7]), prose 47.2 → 47.8, JSON tool call 76.7 → 73.4 (overlap).

**Through the API** (image `qwen38-exl3-tabby:pldgate`, `EXL3_PLD=1 EXL3_MOE_BSZN_MAX=16`,
cap 15, gated; `tools/api_edit.py`, pi-style `edit` / `write` turns rendered by TabbyAPI,
TabbyAPI's own decode rate, 5 reps):

| | features off (deployed) | on |
|---|---:|---:|
| edit turn (2 `edit` calls, 286 tokens) | 82.3 [80.5–83.7] | **100.8** [96.7–102.8] |
| rewrite turn (whole file, 800 tokens) | 85.2 [84.6–85.8] | **140.1** [130.2–140.3] |
| one session, `concurrent_decode.py` (10 rounds) | 54.5 [51.1–56.9] | 53.1 [47.6–58.9] |
| two sessions, aggregate | 55.0 [53.5–57.8] | **64.0** [63.0–71.0] |
| three sessions, aggregate | 53.7 [51.9–54.7] | **63.2** [61.7–64.6] |
| memory, `three_sessions.py` (~117k each) | 78.6 GiB | **84.1 GiB** |

Two and three sessions gain from the 16-row path alone (verifies of 9–16 rows no longer
take the prefill kernel); it costs no memory. The memory is the rollback history at cap 15:
(16 − 6) x ~113 MB x 4 batch slots.

### Status (2026-09-25) and what needs the owner

Deployed: image `qwen38-exl3-tabby:pldgate` as `:latest` (rollback `:pre-pld`), both new
features **off**: greedy output bit-identical to the previous image, decode 54.5 tok/s one
session, 53.7 three, memory 78.6 GiB.

| | Before | With `EXL3_PLD=1 EXL3_MOE_BSZN_MAX=16` (API) | Target |
|---|---:|---:|---:|
| Decode, 400-token answer, one session | 54.0 | 53.1–54.5 (unchanged within noise) | ≥ 60 — not met |
| pi-style edit tool calls | 82 (API) / 81–91 (engine) | 101 (API) / 117–120 (engine) | ≥ 100 — met |
| Whole-file rewrite | 85 (API) | 140 (API) | |
| Prose | 47 (engine) | 47–48 | not lower — met |
| Three sessions | 53.9 | 63.2 | not lower — met |
| Prefill, TTFT | | unchanged (decode-only changes; follow-ups 1.4–1.8 s at 117k) | not lower |

Decisions for the owner:

1. **Turn on `EXL3_PLD=1` and `EXL3_MOE_BSZN_MAX=16`?** Lookup drafting is lossless by
   construction (only drafts equal to the target's own sample are kept). The 16-row path is
   bit-identical at ≤ 8 rows; batches of 9–16 rows (long lookup drafts, 2–3 sessions) now
   run the decode kernels instead of the prefill kernel, a different summation.
2. **Memory for the rollback history.** Cap 15 at `max_batch_size` 4 is 84 GiB. Within
   ~80: cap 11 with `max_batch_size: 3` (+1.4 GiB, ~80 GiB; three sessions still fit), or
   cap 7 at batch 4 (+0.9 GiB; edits +16% instead of +30%). A way around it (recompute the
   GatedDeltaNet state on rollback instead of storing per-token history) is multi-day
   engine work.
3. **The keep/remove decision needs a recorded pi session** (asked at the start of B3):
   `logging.log_prompt: true` for one normal session (restart), or an export from pi.

B2 is closed: every item is below ~2% (B1). The remaining one-session code number is
bounded by the verify cost per row (weight streaming) and MTP acceptance; nothing left in
the list is sized above ~2%.

### Enabled (2026-09-25, approved): cap 11, `max_batch_size: 3`

The owner approved decision 1 and chose cap 11 with `max_batch_size: 3`. Image
`qwen38-exl3-tabby:pld11` is `:latest` (rollback `:pre-pld`, or `EXL3_PLD=0
EXL3_MOE_BSZN_MAX=8` plus `max_batch_size: 4` for the old behaviour on the new image).
Defaults now: `EXL3_PLD=1`, `EXL3_PLD_MAX=11`, `EXL3_MOE_BSZN_MAX=16`; `config.yml`
`max_batch_size: 3`. Gates through the API:

| | features off (`:pldgate`) | enabled (`:pld11`) |
|---|---:|---:|
| edit turn (`api_edit.py`, 5 reps) | 82.3 [80.5–83.7] | **102.5** [96.8–104.0] |
| rewrite turn | 85.2 [84.6–85.8] | **131.9** [128.2–132.7] |
| one session (`concurrent_decode.py N=1 REPS=10`) | 54.5 [51.1–56.9] | 53.7 [48.5–60.3] |
| three sessions (`N=3 REPS=5`) | 53.7 [51.9–54.7] | **61.5** [56.7–65.2] |
| `api_bench.py`: cold 24.5k / follow-up / decode | 19.3 s / 1.51 s / 54 | 18.9 s / 1.42 s / 54.3 |
| `three_sessions.py`: follow-ups at ~117k | 1.5 s | 1.40 / 1.83 / 1.53 s |
| memory, `three_sessions.py` | 78.6 GiB | **81.5 GiB** |
| 128k needle (`ctxfill.py`, `MBS=3`), two seeds | | HIT, HIT (decode 74–76 tok/s) |

Memory is ~1.5 GiB above the estimate (78.6 + 1.4): about 1 GiB goes with the 16-row
path (the same excess showed at cap 15: +5.5 measured vs +4.5 estimated), most likely the
per-shape CUDA graphs and buffers of 9–16-row verifies. It is ~2% over the ~80 GiB
budget; cap 9 would save ~0.7 GiB more at some cost on rewrites (111 vs 119 tok/s engine).
`ctxfill.py` needs `MBS=3` now (its default 16 batch slots x 12 history steps does not fit
next to the model).
