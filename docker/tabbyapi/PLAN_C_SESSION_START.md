# Track C: faster session start and follow-up prefill on the Spark (TabbyAPI, exllamav3 fork)

Written 2026-09-25 for the session that picks this up. Read the whole file before starting,
then `PLAN_B_DECODE.md` ("Constraints", "Setup", "Tools") and `OPTIMIZATION_PLAN.md`
"Follow-up turns: re-prefilled answers, and where the time goes (2026-09-25)". Those record
the measurements this plan builds on; do not repeat them.

## Goal

Cut the time to first token (TTFT) at the start of every pi session, and the fixed cost of every
follow-up turn. Output must not change (see Constraints).

| | Now (2026-09-25, image `:latest` = `:histstash`) | Target |
|---|---:|---:|
| Turn 1 of a pi session (~5.4k-token prompt: pi's system prompt + tools + the task) | 4.6 s (cold every session) | ≤ 1 s once pi's prefix is cached |
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

### C2. A checkpoint where the conversation starts (the ~8 s)

During prefill, also split at the page boundary at or before the first `<|im_start|>user` token
(the end of pi's system prompt + tools, identical across sessions), when no recurrent
checkpoint covers that page yet. Same mechanism as the fork's last-page split: one extra
forward (~0.2 s), paid only when that checkpoint is not cached (first pi session after a
restart, or after LRU eviction). Then:

- turn 1 of every later session resumes at ~5,120 (the shared prefix) instead of 0: 4.6 s →
  ~0.5 s (the ~280 remaining tokens plus one forward);
- turn 2 resumes at ~5,120 instead of 0: 6.3 s → ~2.5 s.

Details to settle:

- The anchor: first `<|im_start|>user` (its id from the tokenizer), rounded down to a page
  boundary. With C1's answer, possibly the last `<|im_start|>user` whose content is not a
  `<tool_response>`.
- Where: `Job.prefill` in `generator/job.py`, next to the `recurrent_last_page` logic. Split
  the chunk at the anchor (`prefill_end = anchor_b`) and stash there with
  `maybe_stash_recurrent(cache, PAGE_SIZE)` semantics. Check that `is_checkpoint_boundary` with
  an override does what the stash needs at that position.
- Only when the anchor is past the resumed position and not already stashed (look it up by
  page hash in `generator.recurrent_cache`).
- Keep the checkpoint warm: it is the most reused one on the server. Consider pinning it in
  the LRU (a small `RecurrentCache` change) so a long session does not evict it.
- Patch `patch_exllamav3_conv_ckpt.py`, `EXL3_CONV_CKPT=1`, anchor-checked like the others.

Measure with a synthetic pi-like start through the API: a ~5k-token system prompt + the four
pi tools, a task, turn 1, then turn 2 built the way pi builds it (after C1, reproduce pi's
change). Report TTFT for turn 1 of a second session and for turn 2, with and without.

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
forward). Do C2 with the split first: it is simpler, and it pays only once per restart.

### C4. Tool-call argument fidelity (if not already fixed)

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
system-wide numbers include ~21 GiB of Higgs TTS + Whisper). Also watch the recurrent
cache: C2 adds one long-lived ~112 MiB checkpoint.

## Reporting

As in Plan B: README "What to expect" and the `EXL3_*` table, a dated section in
`OPTIMIZATION_PLAN.md` "Results", commit, push, pull on the Spark. Then check the owner's next pi
session in `[prefix-diag]`: turn 1 `resumed ≈ 5120`, turn 2 `lost to checkpoints 0`.
