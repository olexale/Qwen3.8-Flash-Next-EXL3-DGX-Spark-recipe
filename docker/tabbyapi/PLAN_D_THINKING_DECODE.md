# Track D: faster decode of thinking (weekend, 2026-09-26/27)

Written 2026-09-25. Read the whole file first, then `PLAN_B_DECODE.md` (Constraints, Setup,
Tools, "What is known", Findings) and `PLAN_C_SESSION_START.md` "What is known" (the
per-request diagnostic). Plan B's decode work (profile, prompt lookup, 16-row MoE) is the
baseline here; do not redo it.


## Update 2026-09-26 (read first)

- **Image:** `:latest` = `:toolmarkers` (adds `patch_tabbyapi_tool_markers.py`: quoted tool
  markers stay text, `reasoning_effort` aliases; TabbyAPI parser only, the engine is unchanged).
  Rollback: `:pre-toolmarkers` (= `:toolargs`). `tests/run_tests.sh` has 38 tests.
- **GPU clocks:** on 2026-09-26 the owner ran `nvidia-smi -rgc` for a comparison (uncapped,
  ~2,400 MHz). The "Now" numbers here were measured at the usual cap (~1,580 MHz). Clocks are
  still the owner's to set: never change them, but record `nvidia-smi --query-gpu=clocks.sm
  --format=csv` before each measurement, and compare before and after only within one clock
  state. At uncapped clocks cold prefill is ~1,590 tok/s (vs ~1,300 capped), and decode is
  code 83 / DevOps 67 / prose 56 tok/s greedy (`tools/compare_bench.py`, 2026-09-26).
- **Voice stack:** Whisper now runs on transformers instead of vLLM (`~/dev/stt-uk`, container
  `stt-uk-lite`, ~2.6 GB instead of ~8.2). With Higgs TTS (~13 GB) the stack is ~15.6 GB, not
  ~21. Both were stopped on 2026-09-26, and the owner restarts them. Memory gates use
  TabbyAPI's own GPU memory, as before.
- **Owner's rule, restated 2026-09-26: keep the quality.** Nothing that changes what the model
  writes ships without the gates and the owner's OK. `reasoning_effort` stays at the template
  default (`xhigh`), so don't propose lowering it.
- **Logs survive restarts:** `stop_tabby.sh` archives `docker logs` to `logs/qwen38-tabby-*.log`
  on the Spark, and the pi session files (`~/.pi/agent/sessions/*/*.jsonl` on the owner's Mac,
  UTC timestamps, provider `gx10`) give per-turn token usage. Matching the two by time is how
  the numbers below were made. The prompts are still the owner's data: use lengths and timings
  only.
- **Updated share of thinking** (145 pi turns in 8 sessions, 2026-09-23..25, from pi's own
  usage records): thinking is **56%** of generated tokens, not the 75% measured on two sessions.
  Tool calls are most of the rest (220 tool-call blocks, 14 text blocks). Over the 114 turns
  matched to the TabbyAPI logs, **decode is 77% of the owner's waiting time** (median 58 tok/s,
  ~360 tokens per turn). Decode +20% would have saved 13% of the total, while prefill at
  blazux's speed (1.85x) would have saved 10%. This plan is the biggest remaining lever. Tool
  calls are buffered until they parse, so their decode is waiting time too: report tool-call
  tok/s as its own number in D0 and D4, next to thinking.
- **Quality:** D1 keeps the output distribution exactly (proof and tests as below) and D2 is
  lossless, so both fit the owner's rule. D1 still needs the owner's explicit OK before the
  live trial.
- **Baseline for D0:** the archived logs (`logs/`) give a pre-D0 baseline of per-request decode
  rates, but they don't separate thinking from the rest. D0's `[decode-stats]` line is what
  splits them.

## Goal and end state

Make decode of **thinking** faster: it is 56% of the tokens the model generates in the owner's
pi sessions (145 turns, 2026-09-23..25; the first estimate was 75% from two sessions), and it
decodes at the slow "prose" rate (~47–50 tok/s, vs 100–130 on edit tool calls).

**Every candidate ends in exactly one of two states:** made permanent (on by default in
`:latest`, documented, tests in `tests/`) or reverted (patch and variables removed, image
rebuilt, the negative result recorded in `README.md` next to the other closed levers). Nothing
stays behind as a silent shadow mode or a default-off switch. The decision comes from a **live
trial in 2–3 of the owner's normal pi sessions** (D4), not from synthetic numbers alone.

| | Now | Target |
|---|---:|---:|
| Thinking decode, real pi sessions (tok/s during the thinking phase; measured in D0) | ~47–50 (estimate) | +10% or more |
| Tool-call and edit turns, code, three sessions, prefill, memory | Plan B "Enabled" / Plan C numbers | not worse than 2% |

## Constraints

As in `PLAN_B_DECODE.md` "Constraints" (GPU clocks untouched, no GPU rental, **no drafter
training**, no vLLM container, announced short downtime windows with an idle check,
patch scripts behind `EXL3_*` variables, small pushed commits with the Claude trailer, prompts
are the owner's data). Two additions:

- **Output distribution must not change.** Candidates that are lossless by construction
  (only drafts equal to the target's own sample are kept) need the usual gates. D1 changes
  *which* random draw the owner gets while keeping the distribution exactly (standard
  speculative sampling): prove it (test below) and **ask the owner before the live trial**.
- **Every patch comes with tests** in `docker/tabbyapi/tests/` (`run_tests.sh`, no model
  needed), and `run_tests.sh` passes on the new image before it is tagged `:latest`.

`reasoning_effort` (the template's `xhigh` default vs `medium`/`low`) would cut thinking tokens
far more than any decode change, but it changes what the model writes. **The owner decided on
2026-09-26 to keep the quality: it stays `xhigh`.** It is not part of this plan unless they ask (then: measure tokens and task outcomes on a fixed task
set, both settings).

## What is known

- Decode is bound by streaming weights per verify round; each extra verified row pulls in its
  own experts (top-10 of 512), so a round costs ~37 ms at 2 rows, ~52 at 5, ~66 at 8, ~103 at 16
  (16-row decode MoE). Tokens accepted per round is the lever (Plan B, B1).
- MTP drafting (5 drafts, dynamic, confidence 0.6) accepts ~79% on code and less on prose;
  deeper drafting, draft-length/confidence sweeps and tree-style extra rows were measured flat or
  negative (Plan B "What is known", T2; README "Levers that are closed").
- Prompt lookup (`EXL3_PLD`, cap 11, min match 8, MTP-agreement gate) helps copying (edits,
  rewrites) and does nothing measurable on prose/code. pi sends the thinking back, so earlier
  turns' thinking is already in the prompt that lookup indexes.
- The pi sampler preset (`qwen38`, `sampler_overrides/qwen38.yml`; the model's
  generation_config) samples with temperature, not greedy, so thinking is sampled text.

## Tasks

Order: D0 → D1 (size, then build) → D2 → D3 → D4 → D5. Stop a candidate early if its sizing
says < ~3% on thinking; record why.

### D0. Measure thinking decode in the owner's sessions (instrumentation that stays)

Add to the engine (or TabbyAPI) one log line per request, next to `[prefix-diag]`:

```
[decode-stats] arm A|B tokens N (thinking T, after </think> U) | time thinking s1, rest s2 |
  rounds R, drafted D, accepted K, lookup rounds L (accepted KL), <candidate counters>
```

Thinking phase = generated tokens up to and including `</think>`. No text, no token ids.
Parse it with a new `tools/decode_report.py` (per-arm medians and ranges of thinking tok/s,
tool-call tok/s, acceptance). This line is permanent (it is how D4 decides and how later work
is measured), documented in the README; the A/B arm field is `-` when no trial runs.

### D1. Speculative sampling instead of exact-match acceptance

**Idea.** For sampled decoding the fork keeps a drafted token d only if the target's own
sample equals d. With d drawn from the draft distribution q and the target's from p, that
accepts with probability Σ_x q(x)·p(x). Standard speculative sampling (Leviathan et al. 2023;
Chen et al. 2023) accepts d with probability min(1, p(d)/q(d)) and, on rejection, samples from
the normalized residual max(0, p − q); its output distribution is exactly p, and it accepts with
probability Σ_x min(p(x), q(x)) = 1 − TV(p, q). On flat distributions (thinking prose under
temperature) the difference can be large (p = q = uniform over 2 tokens: 0.5 vs 1.0). If the
fork drafts **greedily** (d = argmax q) and accepts on p(d), the rule is already optimal for a
point-mass q; the gain then comes from **sampling drafts from q** and using the ratio test.

**First, read the code** (`generator/generator.py` `iterate_draftmodel_mtp_gen`, the verify
loop in `iterate_gen`, `draft_confidence.py`, the sampler): how drafts are chosen (argmax or
sampled, with which temperature/top-k), and what "equal to the target's sample" compares.

**Then size it offline** (engine, TabbyAPI stopped, synthetic thinking prompts with thinking on
and the `qwen38` preset, e.g. `draft_sweep.py`'s prose/code prompts plus a few agent-style
reasoning prompts): at every drafted position record the target's filtered distribution p and
the MTP head's q after the same sampler transforms (temperature, top-k/top-p), and compute the
expected acceptance per position and the expected tokens per round under (a) the current rule,
(b) sampled drafts + ratio test. Convert to tok/s with the round-cost table. Continue only if
(b) is ≥ ~5% better on thinking.

**Build** (`patch_exllamav3_specsample.py`, `EXL3_SPEC_SAMPLE=1`): draft by sampling from q
with the target's sampler settings (keep q's probabilities of the drafted tokens), verify with
the ratio test, sample a rejected position from the residual; greedy requests (temperature 0)
keep the current path. Interactions to handle: dynamic draft length (confidence on q), prompt
lookup drafts (a lookup draft is a point mass: accept with p(d), residual p without d, i.e. the
current rule), the calibrator, logit processors/filters and banned strings, top-k/top-p
truncation (apply the same truncation to p and q before the test).

**Tests (must pass before any trial):** in `tests/test_spec_sample.py`, the acceptance and
residual step against a brute-force reference on small random distributions; a statistical test
that the emitted token frequencies match p (chi-square over 10^5–10^6 draws on small vocabularies,
with and without top-k/top-p) for a chain of 3 drafts; and that temperature 0 is unchanged.

### D2. Lookup drafting tuned for thinking

Thinking quotes identifiers, paths and phrases from tool results and earlier turns, in shorter
runs than edit tool calls. Two variants of `patch_exllamav3_pld.py`:

- **Phase-aware min match**: inside `<think>` use a shorter minimum match (try 4–6) and a short
  lookup draft (≤ 5 tokens, on the 8-row path), keeping the MTP-agreement gate.
- **Cross-request index** (the SuffixDecoding idea, Oliaro et al. 2024): an in-process index of
  the last N requests' outputs (all sessions; bounded memory; never written to disk), consulted
  when the job's own prompt/output has no match. It adds only what the current prompt lacks
  (other sessions, turns lost to context compaction), so size it before building.

Size both offline first with the D1 harness's synthetic prompts (acceptance per lookup round,
fraction of thinking rounds with a usable match), and in the engine with `draft_sweep.py`
(code/prose/edit must not get slower). Build only what sizes at ≥ ~3% on thinking.

### D3. Engine gates for each built candidate

As in `PLAN_B_DECODE.md` "Gates": greedy_ab (lossless-by-construction candidates: identical
greedy outputs or divergence no earlier than old-vs-old), `draft_sweep.py` code/prose/edit ≥ 10
reps (median and range), then through the API (`concurrent_decode.py` N=1/N=3, `api_edit.py`,
`api_bench.py`, fresh-server `three_sessions.py` memory). Plus `tests/run_tests.sh`.

### D4. Live trial in 2–3 of the owner's pi sessions

Ship the candidate(s) that passed D3 into `:latest` with a **per-request A/B switch**
(`EXL3_TRIAL_AB=1`: alternate the candidate on and off by request serial, so both arms see the
same sessions, tasks and context lengths). Tell the owner before starting (for D1: after their
explicit OK). Collect `[decode-stats]` over 2–3 normal sessions (aim for ≥ 60 requests with
thinking), then `tools/decode_report.py`:

- per arm: thinking tok/s (median, IQR, bootstrap 95% CI of the difference), tool-call tok/s,
  acceptance, TTFT;
- keep if thinking tok/s improves ≥ 5% with the CI excluding 0, and tool-call/edit turns are not
  worse than 2%; otherwise revert.

### D5. Decide and finish (make permanent or revert)

- **Keep:** candidate on by default, the A/B switch removed (or its default `0` and the
  variable deleted from `start_tabby.sh`), README "What to expect" and the `EXL3_*` table
  updated, a dated section in `OPTIMIZATION_PLAN.md` "Results", tests kept.
- **Revert:** remove the patch script, its Dockerfile lines, variables and tests; rebuild,
  run `tests/run_tests.sh`, tag `:latest`; add the measured negative result to `README.md`
  "Levers that are closed" with the numbers.

Either way `[decode-stats]` stays (D0), with `arm -`.

## Tools to write

- `tools/spec_sample_size.py`: D1/D2 offline sizing (records p and q per drafted position in
  process memory; prints expected acceptance and tokens per round per rule).
- `tools/decode_report.py`: parses `docker logs qwen38-tabby` `[decode-stats]` lines into the
  D4 table.
- `tests/test_spec_sample.py`, `tests/test_decode_stats.py` (the log line's parsing), tests for
  any D2 variant (the index and phase detection on synthetic token streams).

## Reporting

After D5: README, `OPTIMIZATION_PLAN.md` "Results" (dated; sizing numbers, trial table, the
decision), commit, push, pull on the Spark, and a short summary to the owner with the numbers.

## Findings (2026-09-26)

All numbers at uncapped clocks (2,405–2,411 MHz, `nvidia-smi --query-gpu=clocks.sm` before
each window), engine runs with TabbyAPI stopped.

### D0: `[decode-stats]` (shipped, `:latest` = `:decodestats`, rollback `:pre-decodestats`)

`patch_exllamav3_decode_stats.py` (`EXL3_DECODE_STATS=1`), `tools/decode_report.py`,
`tests/test_decode_stats.py`. The per-request line splits tokens, time and draft counters into
thinking (up to `</think>`), text and tool call (from the first `<tool_call>`); a round's time
is split over its tokens' phases. Live check: 116 tokens in 2.07 s, TabbyAPI's own line said
56.1 T/s. First live request: thinking accepted 43% of drafts, the tool call 87%.

### D1 sizing (`tools/spec_sample_size.py`)

What the code does: MTP drafts are the argmax of a 64K-column head slice, and the verify keeps
a draft only if the target's own sample (fused kernel: temperature, top-k, top-p, Gumbel) equals
it, so acceptance is p(d). TabbyAPI always adds no-op penalty steps, which leaves
`CustomSampler.reqs_past_ids` true, so the fork's batched verify never runs: every position is
sampled with its own launch and sync (a side finding; ~0.3 ms per round).

Synthetic prompts, thinking on, 7 prompts x 2 (then 10 prompts incl. thinking-off), expected
tokens per verify round from p and q at every drafted position (predicted 2.061 vs measured
2.087, so the model is right):

| thinking | current (p(argmax q)) | sampled q + ratio test |
|---|---:|---:|
| acceptance at draft position 0 | 0.593 | 0.741 |
| tokens per round, deployed dynamic windows | 2.061–2.085 | 2.281–2.304 (+10.5–10.7%) |
| best fixed window (pass 2, by measured round cost) | 2 drafts, 47.9 tok/s | 3 drafts, 56.7 tok/s |
| after `</think>` / thinking off | 3.299 | 3.370 (+2.2%) |

q at the target's temperature (1.0) is best (0.8: +9.8%, 0.6: +7.9%); a point mass at the
argmax where q is confident (tau) only lowers the gain (tau 0.9: +10.5%, 0.5: +6.2%).

### D2 (closed, not built)

Offline on the same runs (a lookup draft accepted up to its common prefix with what was
actually sampled; gated on the MTP head's first token; measured round costs):

| thinking rounds | match | used (gate) | est. thinking tok/s |
|---|---:|---:|---:|
| phase-aware min match 4 / 5 / 6 / 8, <= 5 drafts | 7.2 / 5.2 / 4.6 / 3.1% | 5.7 / 4.3 / 3.9 / 2.9% | x1.005 / 1.005 / 1.006 / 1.004 |
| + cross-request index (other prompts' outputs) | +1.2–2.3% | | x1.004–1.005 |

Matches land where MTP is already right, so both are below the 3% bar. In real sessions the
deployed lookup's thinking rounds are counted by `[decode-stats]` (`thinking ... lookup L`).

### D1 build (`patch_exllamav3_specsample.py`, `EXL3_SPEC_SAMPLE`, `EXL3_TRIAL_AB`)

Draft d ~ q (the head slice at the job's temperature, top-k, top-p), verify with the ratio
test, residual max(0, p - q) on rejection, the kernel's own sample of the last position as the
bonus token, one readback per round. p is exactly the kernel's: its histogram workspace holds
the max logit and the kept-set bound per row, and the kept set is recomputed with the same fp32
binning (`tests/test_spec_sample.py`: equal to a tie-aware eager reference, chi-square against
the kernel's samples). Chi-square over 300k 3-token chains (toy model, prefix-dependent p and a
different q, with top-k/top-p, a lookup point-mass position, tau point masses): z −0.37 / 0.44 /
0.22 / −0.42; the same harness with a wrong rule gives z 186–378.

Engine A/B (`draft_sweep.py VARIANTS="A:SPEC=0;B:SPEC=1"`, 10 reps, in process), per build:

| build | thinking pooled (CI) | code | prose | edit | cost per round |
|---|---:|---:|---:|---:|---|
| 1 (p over the whole vocabulary) | +8.8% [+4.3, +13.3] | −7.9% | +3% | −3.9% | +~2 ms |
| 2 (p on the top candidates) | +8.8% [+3.7, +13.7] | −0.8% | −2.0% | −1.0% | +0.8–1.6 ms |
| 3 (fewer launches) | +10.2% [+4.0, +16.5] | −0.5% | +1.5% | −3.1% | |
| 4 (argmax step 0 with a lookup candidate) | | | | −4.4% (accept 94 → 90%) | |
| **5 (thinking phase only)** | **+10.2% [+5.0, +15.8]** | identical path | identical path | −0.1% | |

Edits lost because in copy-heavy output p is sharper than the MTP head's q (p(a) = 0.99 with
q(a) = 0.8 accepts 0.8 by the ratio test, 0.99 by the argmax), so build 5 applies only while the
job is thinking; tcode +14.6%, tdebug +12.0% (median tok/s, mostly thinking).

Gates on build 5: `greedy_ab.py` (AGENT=1): old vs new with `EXL3_SPEC_SAMPLE=1` identical on
all five prompts (400 of 400 tokens), as old vs old. Through the API (`:specsample5` with
`EXL3_SPEC_SAMPLE=1 EXL3_TRIAL_AB=1` vs `:latest`, same clocks, one after the other):

| | candidate | `:latest` |
|---|---:|---:|
| `api_think.py` REPS=6 (per-arm, `[decode-stats]`) | B vs A thinking +9.9% pooled [−0.8, +22.3], 2.15 → 2.54 tokens per round, spec rounds in every B request, no overflow | |
| `concurrent_decode.py` N=1 / N=3 | 55.7 [49.6–61.3] / 68.8 [66.9–73.9] | 57.6 [49.2–64.2] / 70.8 [68.4–73.5] |
| `api_edit.py` edit / rewrite | 118.4 / 151.1 | 118.5 / 152.0 |
| `api_bench.py` cold 24k / follow-up TTFT | 16.57 / 1.15 s | 16.59 / 1.15 s |
| `three_sessions.py` (100k, follow-up TTFT) | 1.76 / 1.46 / 1.39 s | 1.75 / 1.43 / 1.39 s |
| TabbyAPI GPU memory after three sessions | 73,681 MiB | 77,531 MiB |

`tests/run_tests.sh`: 81/81 on `:specsample5`. Next: D4 needs the owner's OK.
