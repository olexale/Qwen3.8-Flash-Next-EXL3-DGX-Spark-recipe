# Track D: faster decode of thinking (weekend, 2026-09-26/27)

Written 2026-09-25. Read the whole file first, then `PLAN_B_DECODE.md` (Constraints, Setup,
Tools, "What is known", Findings) and `PLAN_C_SESSION_START.md` "What is known" (the
per-request diagnostic). Plan B's decode work (profile, prompt lookup, 16-row MoE) is the
baseline here; do not redo it.

## Goal and end state

Make decode of **thinking** faster: it is 75% of the tokens the model generates in the owner's
pi sessions (24% tool calls, 1% visible answer text; two sessions, 30 turns, 2026-09-25), and it
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
far more than any decode change, but it changes what the model writes: the owner's decision,
not part of this plan unless they ask (then: measure tokens and task outcomes on a fixed task
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
