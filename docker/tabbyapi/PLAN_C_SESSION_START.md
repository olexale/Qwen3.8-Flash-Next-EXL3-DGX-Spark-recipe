# Track C: faster session start and follow-up prefill on the Spark (TabbyAPI, exllamav3 fork)

Written 2026-09-25 for the session that picks this up. Read the whole file before starting,
then `PLAN_B_DECODE.md` ("Constraints", "Setup", "Tools") and `OPTIMIZATION_PLAN.md`
"Follow-up turns: re-prefilled answers, and where the time goes (2026-09-25)". Those record
the measurements this plan builds on; do not repeat them.


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
- **What Plan C is worth in real use** (114 pi turns in 8 sessions, 2026-09-23..25, matched
  to the TabbyAPI logs): waiting time was 23% TTFT and 77% decode. C2 saves up to ~8 s per session
  (turn 1 cold 4.5–4.6 s, turn 2 lost to checkpoints 4.7–6.3 s in both 09-25 sessions), about 3%
  of the total. C3 saves ~0.2 s per turn, about 1%. That is small, but it is low risk and
  within prefix-caching variation, so do it, and still run the gates. What Plan C cannot fix:
  follow-ups that bring 4.5–8.5k genuinely new tokens (tool results) on top of 10–30k of
  context run at ~900–1,200 tok/s, and only prefill throughput (the MoE kernel, A3h) speeds
  them up. The owner set that aside on 2026-09-26.
- **C4 is done:** `patch_tabbyapi_toolcall_args.py` shipped on 2026-09-25 (`TABBY_TOOLCALL_ARGS=1`).

## Goal

Cut the time to first token (TTFT) at the start of every pi session, and the fixed cost of every
follow-up turn. Output must not change (see Constraints).

| | Now (2026-09-25, image `:latest` = `:histstash`) | Target |
|---|---:|---:|
| Turn 1 of a pi session (~5.4k-token prompt: pi's system prompt + tools + the task) | 4.6 s (cold every session) | ≤ 1 s from the third session with the same prefix (C2b), if C1 shows the prefix repeats |
| Turn 2 of a pi session (~8.1k tokens) | 6.3 s (prefills all 8.1k again) | ≤ 3 s |
| Follow-up turn, ~350 new tokens after an answer (engine, `tools/followup_profile.py`) | 0.9–1.2 s | ≤ 0.8 s |
| Everything else (decode, three sessions, memory, prefill of long prompts) | see `PLAN_B_DECODE.md` "Enabled" | not worse |

Together about 8 s per pi session from the first two items, plus ~0.2 s per turn from the third.
The targets are what to report against. Reaching one is not a reason to stop (same rule as
Plan B, "When to stop").

## What is known (measured 2026-09-25, do not re-measure unless something changed)

**The per-request diagnostic.** `patch_exllamav3_prefix_diag.py` (`EXL3_PREFIX_DIAG=1`,
forwarded by `start_tabby.sh`, on in the running server since 2026-09-25) logs one line per
request:

```
[prefix-diag] prompt P | kv-matched A, resumed B, lost to checkpoints A-B | prev prompt Pp out O,
  lcp L (out+X | prompt-Y), marks in prev out: </think> .., <tool_call> .., <|im_end|> .. | </think> count prev N new M
```

`docker logs qwen38-tabby | grep prefix-diag` on the Spark. The line holds lengths and marker
positions only, never text. `lcp` is the longest common token prefix with the most similar
earlier sequence (the last 16 finished sequences are kept in process memory).

**Two real pi sessions (the owner's, 2026-09-25, 20 + 11 turns):**

- pi sends the thinking back (`</think>` counts equal) and every follow-up prompt matched the
  whole previous answer, except inside one tool call (see "Tool-call round trip" below).
- Since `EXL3_HIST_STASH=1` (shipped 2026-09-25), follow-ups resume at the previous answer's
  last page boundary: `lost to checkpoints 0` on all later turns. TTFT is 0.42–0.84 s for
  130–525 new tokens.
- **Turn 2 is the exception, in both sessions:** turn 1's prompt is 5,403 tokens, and turn 2's
  prompt diverges from it at token 5,187, i.e. 216 tokens before the end of turn 1's prompt
  (`lcp 5187 (prompt-216)`). That is about where the first user message starts. Turn 1 was
  prefilled cold in one forward, so its only recurrent checkpoint is at its last page (5,376).
  With no checkpoint at or before 5,187, turn 2 prefills all 8,137 tokens (6.3 s) where
  ~2,950 were new.
- Turn 1 is always cold (4.6 s for 5,403 tokens), although pi's system prompt and tool
  definitions (the first ~5.2k tokens) are the same in every session.
- Generated tokens across 30 turns: 75% thinking, 24% tool calls, 1% answer text.

**Where a follow-up's TTFT goes** (`tools/followup_profile.py`, engine, 16k conversation):

| new tokens | forward 1 (to the last page boundary) | forward 2 (partial last page) | first decode | TTFT | one forward |
|---:|---|---|---:|---:|---:|
| 100 | 512 rows, 690 ms | 44 rows, 230 ms | 40 ms | 0.96 s | 0.81 s |
| 350 | 768 rows, 900 ms | 38 rows, 215 ms | 42 ms | 1.18 s | 0.99 s |
| 1,000 | 1,280 rows, 1,250 ms | 176 rows, 427 ms | 41 ms | 1.73 s | 1.43 s |

A prefill forward costs ~0.2 s before doing any work: a few dozen rows already touch nearly all
512 experts, so the weights stream once per forward. The fork splits the partial last page into
its own forward (`job.py` `prefill()`, "separate forward pass for the last page to get the
latest possible checkpoint") only so it can stash the recurrent state at the page boundary.
Removing the split outright loses that checkpoint, and the next turn then falls back to an
older one (5.8 s in the test). Everything else in a follow-up (page allocation, state restore,
stashes, MTP draft prefill) is < 20 ms.

**How checkpoints work** (read `generator/job.py` `prefill`, `allocate_pages`,
`is_checkpoint_boundary`, `maybe_stash_recurrent`; `generator/pagetable.py`
`Sequence.allocate_pages`; `cache/recurrent.py`): a request resumes from the longest run of
cached 256-token pages whose last page also has a stashed recurrent state (keyed by the page
hash). Stashes are made at the end of a prefill forward when the position is on a checkpoint
boundary: every 32,768 tokens far from the prompt's end, every 2,048 near it (relative to the
resumed start), and at the last page via the split. `EXL3_HIST_STASH` adds every page boundary
of the output. Checkpoints are ~112 MiB each (36 GatedDeltaNet layers + PLE), kept on the device
(`patch_exllamav3_checkpoints.py`), in an LRU capped by `sysmem_recurrent_cache` (8 GiB).

**Tool-call round trip.** TabbyAPI parses the model's `<tool_call>` text into JSON arguments
(`/app/endpoints/OAI/utils/toolcall_formats/qwen3_coder.py`) and the chat template renders them
back on the next turn. The parser strips all whitespace around each value and `json.loads` it
regardless of the tool's schema (`common.py` `coerce_param_value`), so the re-rendered text can
differ from what the model generated. That breaks the prefix match inside the tool call, and
the arguments pi receives are wrong. It is being fixed separately; check whether
`patch_tabbyapi_toolcall_args.py` exists before assuming anything here.

## Constraints

As in `PLAN_B_DECODE.md` "Constraints": output unchanged (bit-identical, or passing the gates
below with the owner's approval), GPU clocks untouched, no GPU rental, no vLLM container,
announced short TabbyAPI downtime windows with an idle check first (`docker logs --since 3m
qwen38-tabby | grep -c "prompt tokens ·"`), no warm-up request, patch scripts behind `EXL3_*`
variables forwarded in `start_tabby.sh` and documented in `README.md`, small commits to `main`
with the Claude trailer, pushed, and pulled on the Spark. **The prompts are the owner's data:
never log or store prompt text or token ids beyond what `EXL3_PREFIX_DIAG` already logs without
asking first.**

Where a change only moves prefill split points, outputs follow the same variation that
prefix caching already causes (a cold prompt and a cached one are split differently today).
That is still not bit-identical: run the gates and ask before shipping.

## Tasks

Order: C1 → C2 → C3. C4 is independent.

### C1. What changes at token 5,187 between turn 1 and turn 2 (small, one short window)

Find the cause of the turn-2 divergence without reading prompt text. Extend the diagnostic
(`EXL3_PREFIX_DIAG=2`, not the default) to log, for the divergence point only:

- the position of the first `<|im_start|>user` in both prompts, and the offset of the
  divergence from it;
- the token classes on both sides for ±8 tokens (special / whitespace / other), plus whether
  the new prompt contains the old prompt's tail as a later substring (`(prompt-216)` tokens
  moved rather than changed).

Candidates to tell apart: pi appends something to the latest user message on turn 1 only (e.g.
a context block or reminder), pi changes the system prompt tail, or the chat template renders
the first user message differently once an assistant turn follows. If pi's change is
deliberate and fixed-size, a checkpoint just before the first `<|im_start|>user` (C2) covers
it. If the diff is at the system prompt's tail, anchor C2 earlier. Ask the owner before
logging any actual text, even decoded tokens.

**Second question for C1 (added 2026-09-26): does pi's prefix repeat across sessions?** Plan C
first assumed pi's system prompt + tools (~5.1k tokens) are identical in every session. No log
shows it: the only cross-session comparison (09-25 13:59) was against an unrelated request
(the diagnostic keeps 16 sequences). pi likely includes the working directory or project files,
so sessions in different projects may diverge early. Keep a longer, bounded history for
`EXL3_PREFIX_DIAG=2` (the page-hash chain of each first turn's prompt, never text), and report
for each new session's turn 1 the longest prefix shared with any earlier session's turn 1,
same project vs different project (the owner can tell which sessions were which). This sizes
C2b: if turn 1 prefixes don't repeat beyond a few hundred tokens, C2b is dropped.

### C2. Checkpoints where prompts actually diverge (turn 2, and turn 1 of repeat sessions)

**Several harnesses share this server** (pi, Paseo, other agents, a chat now and then) and run
in parallel (`max_batch_size: 3`). Restoring a checkpoint is safe for any mix: pages are keyed by
a hash that chains in the previous page's hash (`generator/pagetable.py` `prev_hash`), so a
checkpoint is only found by a prompt identical up to that page. What can go wrong is
performance: checkpoints nobody reuses (harnesses that put the date, working directory or git
status in the system prompt make a new prefix every session), extra ~0.2 s prefill forwards
that also stall the other sessions' decode, and eviction pressure on the shared 8 GiB
recurrent cache (~70 checkpoints of ~112 MiB). Evicting a live session's latest checkpoint
costs that session a full re-prefill, far more than C2 saves. So: **no pinning, no checkpoint
from a template guess alone, a small cap of its own.**

**C2a. Within a session: a checkpoint before the latest user message (turn 2, measured ~4–6 s).**
Turn 2 diverges from turn 1 at token 5,187, ~216 tokens before turn 1's end, about where the
user message starts (C1 confirms the cause). While prefilling a prompt whose last user
message starts past the resumed position, also split at the page boundary at or before that
message's `<|im_start|>user` and stash there. That is at most one extra checkpoint per request,
in the ordinary LRU, not pinned. If C1 shows the change is elsewhere (e.g. the system prompt's
tail), anchor there instead, by the same rule: a stable marker, at most one per request.

**C2b. Across sessions: a checkpoint where a new prompt shares a long prefix with a recent
different conversation (turn 1 of repeat sessions).** Only if C1's second question shows
repeating prefixes. Data-driven, not template-driven: when a prompt's longest common prefix
with a recent *different* sequence (the diagnostic already computes it) is ≥ `EXL3_CONV_CKPT_MIN`
tokens (default 2,048) and no checkpoint covers that point, split at the page boundary at or
before it and stash. A prefix seen once costs nothing; a harness with a stable system prompt
pays one extra forward on its second session and resumes from its third session on. These
checkpoints go into a separate small LRU (`EXL3_CONV_CKPT_MAX`, default 4 checkpoints,
~450 MiB) so they never push out a live session's checkpoints, and are never pinned.

Details to settle:

- Where: `Job.prefill` in `generator/job.py`, next to the `recurrent_last_page` logic. Split
  the chunk at the anchor (`prefill_end = anchor_b`) and stash there with
  `maybe_stash_recurrent(cache, PAGE_SIZE)` semantics. Check that `is_checkpoint_boundary` with
  an override does what the stash needs at that position.
- Only when the anchor is past the resumed position and not already stashed (look it up by
  page hash in `generator.recurrent_cache`).
- Two sessions reaching the same anchor at once: both may stash the same page hash. Check
  that `RecurrentCache` replaces or dedups the entry, and test it.
- Patch `patch_exllamav3_conv_ckpt.py`: `EXL3_CONV_CKPT=1` turns on C2a, `EXL3_CONV_CKPT=2` turns on
  C2a + C2b. Anchor-checked like the other patches. Add a counter to `[prefix-diag]`: anchor
  checkpoints made, reused, and evicted unused.

Measure through the API with a synthetic pi-like start: a ~5k-token system prompt + the four
pi tools, a task, turn 1, then turn 2 built the way pi builds it (after C1, reproduce pi's
change). Report TTFT for turn 2 (C2a), for turn 1 of the second and third sessions with the
same prefix (C2b), with and without. Then run the **multi-harness check** in Gates.

### C3. One prefill forward instead of two (the ~0.15–0.3 s per follow-up)

Keep the checkpoint at the last page boundary, but take it from inside a single forward over
the whole chunk:

- GatedDeltaNet: the prefill runs FLA's chunked gated-delta rule, which computes the state at
  every 64-token chunk boundary (`chunk_gated_delta_rule_fwd_h`, `h` per chunk). Chunks start
  page-aligned, so the state at the last page boundary is `h[(b - start) / 64]`. Find where the
  fork calls it (`modules/gated_delta_net_fn/`, the `gated_delta_rule_fn` torch path; our
  `patch_exllamav3_gdn_nocopy.py` touches the same call) and return that state as well.
- Conv state at b: the last `conv_kernel_size` pre-conv inputs before b (bf16), from the
  forward's `mixed_qkv`. PLE: its conv window and the token-id context before b (the ids are
  known).
- Stash those as the checkpoint for page b (the same dict format as `GDNState.stash()`;
  `patch_exllamav3_hist_stash.py` builds one by hand and shows how to put it into the cache).
- **Parity first, before any speed work:** the stash from inside one forward vs the stash
  from the two-forward split, per layer. Bit-identical is expected for the recurrent state (the
  same chunk math on the same inputs); check it. The rows' outputs after b will likely not be
  bit-identical (MoE/attention batch shapes change), so C3 needs the greedy/needle gates and
  the owner's approval.

The same mechanism could make C2's extra forward free (take the anchor's state from the one
forward). Do C2 with the split first: it is simpler to verify.

### C4. Tool-call argument fidelity (done 2026-09-25, `patch_tabbyapi_toolcall_args.py`)

If `patch_tabbyapi_toolcall_args.py` does not exist yet, see "Tool-call round trip" above and
fix it first: it is a correctness bug, not a speed item.

## Tools

- `tools/followup_profile.py`: per-forward timing of a follow-up (engine, TabbyAPI stopped),
  `NOSPLIT=1` variant.
- `tools/hist_stash_test.py`: parity of history checkpoints against the engine's own
  rewind + stash, plus follow-up resume/TTFT/greedy comparison. It is the template for C3's
  parity test.
- `tools/api_edit.py`, `tools/api_bench.py`, `tools/three_sessions.py`,
  `tools/concurrent_decode.py`: the API gates (see Plan B).
- `~/scratch/turns.py` on the Spark: a two-turn tool conversation through the API (thinking on;
  `MODE=keep|drop` returns the reasoning or not). Note: it imports `api_edit`, which runs that
  benchmark at import time; guard it with `if __name__ == "__main__"` in `api_edit.py` first.

## Gates

As in `PLAN_B_DECODE.md` "Gates", plus TTFT for turn 1 of a second pi-like session and for turn 2
(C2), and the follow-up table above (C3). Memory: fresh-server `three_sessions.py`, TabbyAPI's
own GPU memory from `nvidia-smi --query-compute-apps` (the voice stack now shares the machine:
system-wide numbers include ~15.6 GiB of Higgs TTS + lite Whisper when they run). Also watch the recurrent
cache: C2b adds up to `EXL3_CONV_CKPT_MAX` × ~112 MiB.

**Multi-harness check (C2, added 2026-09-26).** Three sessions in parallel through the API,
each a multi-turn tool conversation with a different system prompt: pi-like (stable), one that
changes per session (date, working directory and git status in the system prompt, as Claude
Code does), and a short chat. Run it twice (two "sessions" of each harness). Pass when:

- the live sessions show no new `lost to checkpoints` in `[prefix-diag]` versus the same run
  without the patch, and their TTFT and decode are not worse than 2%;
- the per-session-changing harness creates no C2b checkpoint (its prefixes never repeat) and
  pays no extra forward beyond C2a's;
- TabbyAPI's GPU memory stays within the fresh-server `three_sessions.py` budget plus
  `EXL3_CONV_CKPT_MAX` × ~112 MiB;
- correctness: two prompts that share pages and then diverge before the anchor give greedy
  outputs identical to a cold prefill of each (the same check as `tools/hist_stash_test.py`).

## Reporting

As in Plan B: README "What to expect" and the `EXL3_*` table, a dated section in
`OPTIMIZATION_PLAN.md` "Results", commit, push, pull on the Spark. Then check the owner's next pi
session in `[prefix-diag]`: turn 2 `lost to checkpoints 0` (C2a); with C2b on, turn 1 of a
repeat session resumes at the shared prefix. Also check the anchor counters: checkpoints made
but never reused should stay low.


## Findings (2026-09-26)

Clocks were uncapped all day (2,405–2,444 MHz, the owner's `nvidia-smi -rgc`); every
before/after pair below ran in that state.

### C1: what changes between turn 1 and turn 2, and whether pi's prefix repeats

**`EXL3_PREFIX_DIAG=2`** (`patch_exllamav3_prefix_diag.py`, log-only, lengths and classes only)
adds a `diverge at L: msg #k role R +o, first user prev/new, last user prev/new | prev tail T at
new +d | classes ...` line and a `session start #n ... shared with #m S tokens` line (page hashes
of the last 64 session starts, no ids). 5 unit tests.

- **Not the chat template.** It renders every user message the same way whether or not an
  assistant turn follows, and `preserve_thinking` defaults to on.
- **Not pi's core or `project_report`.** Non-interactive pi (`pi -p --no-session`, the owner's
  extensions loaded) in an empty directory and in a clone of this repo, with and without a
  `project_report` call: turn 2 matched all of turn 1 every time (`lcp out+158`).
- **In the owner's own sessions it is the first user message, and only in some sessions.**
  Session files, lengths only: in the same project (qwen-image), with identical system-prompt
  section lengths, the 09-24 18:55 and 09-25 11:22 sessions had turn 1 = 5,205–5,209 tokens
  and turn 2 resumed at 5,120, while the 09-25 13:58 and 14:56 sessions had turn 1 = 5,403
  tokens for a 149-character user message (~35 tokens) and turn 2 prefilled from 0. So ~180
  tokens were in turn 1's user message that are neither in the session file nor sent again on
  turn 2. The divergence (5,187) sits just past that message's `<|im_start|>user`. That is the
  plan's first candidate (pi adds something to the latest user message on turn 1 only), from
  the interactive session (an extension or the TUI), not reproducible with `-p`. The server
  runs `EXL3_PREFIX_DIAG=2` now, so the owner's next interactive pi session logs the exact
  offset and whether the old tail moved.
- **pi's prefix repeats across sessions.** Diag level 2 on real pi runs: sessions in
  *different* projects share the prompt up to the working-directory line, 4,862 tokens (4,608
  page-aligned) of the ~5,244 before the first user message; sessions in the *same* project
  share everything up to the first user message (and a repeat session already resumes at turn
  1's last page when the user message starts after it: 5,120 of 5,273). So C2b is worth doing.

### C2: anchor checkpoints (`patch_exllamav3_conv_ckpt.py`, `EXL3_CONV_CKPT`, off by default)

`EXL3_CONV_CKPT=1` (C2a): while prefilling, also split at the page boundary at or before the
latest user message (the last `<|im_start|>user` that is not a tool response nor quoted inside
one) and stash there, in the ordinary LRU. `=2` adds C2b: split and stash at the end of a
≥ `EXL3_CONV_CKPT_MIN` (2,048) full-page prefix shared with a recent *different* prompt (one it
does not extend; the last 32 prompts' page hashes), in a separate LRU of `EXL3_CONV_CKPT_MAX`
(4) entries that does not count against `sysmem_recurrent_cache` and that the ordinary LRU never
evicts. An anchor is only used past the resumed position, before the prompt's last page boundary
(the fork stashes that one anyway), and when not stashed yet; two sessions reaching the same
anchor keep one entry. With `EXL3_PREFIX_DIAG` the log shows `anchor made a|b at P` and the
counters (made, reused, evicted unused). 9 unit tests.

**Synthetic pi-like start** (`tools/session_start.py`: a fresh nonce-prefixed ~5k-token
system prompt with the cwd ~380 tokens before its end and four pi tools; turn 1's user message
carries a ~180-token block that turn 2 does not; median of 3 reps, TTFT):

| | without | `EXL3_CONV_CKPT=2` |
|---|---:|---:|
| session 1, turn 1 (cold) | 4.07 s | 4.53 s (one extra forward) |
| **session 1, turn 2** | 5.15 s | **2.57 s** (resumes at the C2a anchor) |
| session 2 turn 1, same project | 0.48 s | 0.51 s |
| session 3 turn 1, second project | 3.84 s | 4.49 s (makes the C2b anchor) |
| **session 4 turn 1, third project** | 3.83 s | **1.29 s** (resumes at it) |

Targets: turn 2 ≤ 3 s is met (2.6 s; the owner's turn 2 was 6.3 s at the capped clock).
Turn 1 ≤ 1 s from the third session with the same prefix is not quite: 1.29 s, because the
cross-project prefix ends ~640 tokens before the end of the prompt (cwd, the rest of pi's
system prompt, the task); in the same project the repeat session is 0.5 s.

**Engine checks** (`tools/conv_ckpt_test.py`, TabbyAPI stopped): turn 2 resumes at 5,120
(2.0 s vs 4.9 s cold); a prompt that diverges before every anchor resumes at 0 (no wrong
restore); a sibling of it resumes at its C2b anchor (1.6 s vs 3.3 s); two sessions enqueued
together reach the same new anchor and the separate LRU stays consistent. Cold prefill is
deterministic (cold vs cold: first-token logits bit-identical), and a resumed prompt differs
from a cold one like any other change of split points: first-token KL 1e-4 to 0.6 on these
random-text prompts (top-1 equal in 9 of 10), against KL 1.7–3.1 between two different
prompts. The right reference is a cold prefill whose first chunk ends at the anchor (the same
forward shapes as resuming there, as the engine's 8,192-token chunking does on long prompts):
against it the resumed prompt's first-token logits are bit-identical in conversation 1 (both
prompts) and within KL 7e-4, top-10 identical, in conversation 0. So C2 changes outputs exactly
as moving a chunk boundary does. Greedy text is too noisy here to judge (cold vs cold diverges
within ~10 tokens).

Side observation, not C2's: a checkpoint the fork made at a 5,121-token prompt's last page
(5,120) was not used by the 7,703-token prompt that followed although all of its K/V pages were
cached (`kv-matched 7680, resumed 0`), while the same prompt resumed at 5,120 from an anchor made
inside a longer prompt. Not investigated here.

**Multi-harness check** (`tools/multi_harness.py`: pi-like, Claude-Code-like with date/cwd/git
status first in the system prompt, and a short chat, in parallel; 2 sessions × 4 turns each;
3 seeds; fresh server per arm):

- **Resumes:** requests with `lost to checkpoints` > 0: 16 → 8. The 8 left are the 512-token
  tool header shared across harnesses on each turn 1, the same in both arms. pi's turn 2 and
  its second session (5.4–6.4k tokens lost each without the patch) now resume.
- **Anchors:** 3, all C2a on pi's turn 1. The per-session-changing harness made none and
  paid no extra forward; no C2b was made (the only repeated long prefix was already covered
  by C2a).
- **Speed:** summed TTFT 206 → 193 s; decode means per harness 27.1 / 23.9 / 22.7 → 27.6 /
  26.2 / 27.2 tok/s (run-to-run spread is ±30% in parallel runs, so this only says no
  systematic loss). The one real cost: a request running alongside a pi turn 1 that makes an
  anchor waits for its extra forward, +0.06–0.4 s on ~8 s turn-1 requests (1–4%).
- **Memory:** TabbyAPI's peak GPU memory 76.4 vs 77.4 GiB after three seeds (it grows seed by
  seed in both arms as the recurrent cache fills). Fresh-server `three_sessions.py`: 72,995 vs
  72,899 MiB (+96 MiB, one checkpoint; budget +4 × 112 MiB), system 80.1 vs 79.7 GiB, no lost
  checkpoints in either arm.

**Known limitation:** TabbyAPI encodes chat markers inside message text as special tokens. A
chat marker quoted in a *user* message (not a tool result) looks like a message start, and
makes an anchor that nobody reuses: in `three_sessions.py`, whose filler quotes chat-format
examples, 5 such anchors cost +0.28/+0.35 s on 2 of 6 follow-ups. Markers inside tool results
are skipped.

**Not shipped as default:** split points move, so outputs are not bit-identical to before
(the same kind of variation prefix caching already causes). Needs the owner's OK.

### C3: one forward instead of two (parity done; stopped at ~1%, needs approval to ship)

FLA's chunked kernel (`vendor/fla/chunk_delta_h.py`) keeps the running state in fp32 registers
but stores the per-chunk states `h` in the input dtype (bf16); only the final state is fp32. So
`h[(b - start) / 64]` would be a bf16-rounded checkpoint (off by up to ~3.7e-3), not the state the
split stores. **`patch_exllamav3_fla_capture.py`** (not in the Dockerfile) adds an optional fp32
store of the state at one chunk index (`chunk_gated_delta_rule(..., capture_state=buf,
capture_chunk=n)`); nothing changes when it is not asked for.

- **Kernel parity** (`tools/fla_capture_parity.py`, no model): on identical inputs the captured
  state equals the split forward's final state bit for bit (T 257–8,192, six boundaries), and
  outputs and final state are unchanged by the capture.
- **Full model** (`tools/c3_parity.py`, TabbyAPI stopped): a checkpoint assembled from one
  forward (GDN: captured fp32 state + last 4 pre-conv inputs; PLE: conv-stream columns + token
  context) against the split's stash. Where the inputs are identical it is bit-identical (GDN
  layer 0, PLE on a cold prompt), so the assembly is exact. Deeper layers differ (max |diff| ~3
  on rare large values) because the rows before the boundary go through GEMM/MoE kernels whose
  path depends on the row count (5,888 vs 6,015 rows; 512 vs 593), the same effect as moving a
  chunk boundary; the checkpoint stays consistent with the K/V the same forward wrote.
  First-token logits one vs split: KL 1.2e-6 (cold 6k prompt), 5.2e-6 (follow-up); greedy 64
  tokens identical (cold) / diverge at 55 (follow-up, greedy noise level).
- **Speed:** follow-up TTFT 0.856 → 0.682 s (−0.17 s; 512 + 81 rows after a 6.1k context).

Remaining to ship: wire the capture into `Job.prefill` in place of the last-page split (GDN
module forward and PLE take the boundary from params, the job builds the stash dict as
`patch_exllamav3_hist_stash.py` does), then the greedy/needle gates. At ~0.17 s per turn (~1% of
waiting time) this is below Plan B's ~2% line, and it is not bit-identical, so it waits for the
owner's decision.
