"""Prefix-reuse diagnostic (EXL3_PREFIX_DIAG=1, off by default): log why a follow-up turn
re-prefills tokens that were already in the cache.

A follow-up request resumes from the longest cached page prefix (256-token pages) that also
has a stashed recurrent (GatedDeltaNet) state. Real pi turns re-prefilled roughly the whole
previous answer (2026-09-25 log: 22 new tokens expected, 866 prefilled). Two causes are
possible and need different fixes:

- the rendered prompt equals the previous prompt + output, but no recurrent checkpoint
  covers the output (checkpoints during generation come every 2,048 tokens), so the K/V
  pages match by hash and the state does not ("lost to checkpoints" below);
- the rendered prompt differs from what the model generated (thinking dropped by the
  client, tool calls re-serialised differently), so even the K/V pages stop matching
  ("lcp" inside the previous output).

Per job, at page allocation, one line on stdout (docker logs):

  [prefix-diag] prompt P | kv-matched A, resumed B, lost to checkpoints A-B | prev prompt Pp
  out O, lcp L (out+X), marks in prev out: </think> T, <tool_call> C | </think> prev N1 new N2

"prev" is the finished sequence (of the last 16 kept in memory) sharing the longest token
prefix with the new prompt. No token ids or text are logged; only lengths and positions of
the </think>, <tool_call> and <|im_end|> markers. The kept sequences live only in the
process's memory. Nothing changes in generation.

Usage: python3 patch_exllamav3_prefix_diag.py [exllamav3 package dir]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

HELPERS = '''

# --- prefix-reuse diagnostic (patch_exllamav3_prefix_diag.py) ---
import os as _pd_os
from collections import deque as _pd_deque
_PREFIX_DIAG = _pd_os.environ.get("EXL3_PREFIX_DIAG", "0") == "1"
_pd_ring = _pd_deque(maxlen = 16)  # (ids int64 cpu tensor, prompt length)
_pd_marks = None

def _pd_mark_ids(tokenizer):
    global _pd_marks
    if _pd_marks is None:
        _pd_marks = {}
        for name in ("</think>", "<tool_call>", "<|im_end|>"):
            try:
                _pd_marks[name] = tokenizer.single_id(name)
            except Exception:
                pass
    return _pd_marks

def _pd_first(ids, tid):
    if tid is None:
        return None
    hit = (ids == tid).nonzero()
    return int(hit[0, 0]) if hit.numel() else None

def _pd_lcp(a, b):
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    ne = (a[:n] != b[:n]).nonzero()
    return int(ne[0, 0]) if ne.numel() else n

def _prefix_diag_alloc(job, seq, cached_pages, kv_only_pages):
    try:
        ids = seq.sequence_ids.torch()[0].cpu()
        if getattr(job, "_pd_prompt_len", None) is None:
            job._pd_prompt_len = ids.shape[0]
        p = ids.shape[0]
        matched = (cached_pages + kv_only_pages) * PAGE_SIZE
        resumed = cached_pages * PAGE_SIZE
        line = (f"[prefix-diag] prompt {p} | kv-matched {matched}, resumed {resumed}, "
                f"lost to checkpoints {matched - resumed}")
        best = None
        for prev_ids, prev_p in _pd_ring:
            l = _pd_lcp(ids, prev_ids)
            if best is None or l > best[0]:
                best = (l, prev_ids, prev_p)
        if best is not None and best[0] > 0:
            l, prev_ids, prev_p = best
            out = prev_ids.shape[0] - prev_p
            marks = _pd_mark_ids(job.generator.tokenizer)
            prev_out = prev_ids[prev_p:]
            rel = lambda x: "-" if x is None else f"out+{x}"
            where = f"out+{l - prev_p}" if l >= prev_p else f"prompt-{prev_p - l}"
            n_think = lambda t: int((t == marks["</think>"]).sum()) if "</think>" in marks else -1
            line += (f" | prev prompt {prev_p} out {out}, lcp {l} ({where}), marks in prev out: "
                     f"</think> {rel(_pd_first(prev_out, marks.get('</think>')))}, "
                     f"<tool_call> {rel(_pd_first(prev_out, marks.get('<tool_call>')))}, "
                     f"<|im_end|> {rel(_pd_first(prev_out, marks.get('<|im_end|>')))} | "
                     f"</think> count prev {n_think(prev_ids)} new {n_think(ids)}")
        else:
            line += " | no earlier sequence shares a prefix"
        print(line, flush = True)
    except Exception as e:
        print(f"[prefix-diag] error: {e!r}", flush = True)

def _prefix_diag_done(job):
    try:
        p = getattr(job, "_pd_prompt_len", None)
        if p is None:
            return
        ids = job.sequences[0].sequence_ids.torch()[0].cpu().clone()
        # A requeued continuation extends an entry already kept; replace it
        for i, (prev_ids, _) in enumerate(_pd_ring):
            if prev_ids.shape[0] <= ids.shape[0] and _pd_lcp(ids, prev_ids) == prev_ids.shape[0]:
                del _pd_ring[i]
                break
        _pd_ring.append((ids, p))
        job._pd_prompt_len = None
    except Exception as e:
        print(f"[prefix-diag] error: {e!r}", flush = True)
'''

edits = {
    "generator/job.py": [
        ("        for seq in self.sequences:\n"
         "            allocated_pages, cached_pages, non_sequential_pages, stashed_recurrent_state = \\\n"
         "                seq.allocate_pages(self.pagetable, self.generator.recurrent_cache, protected_hashes)\n",
         "        for seq in self.sequences:\n"
         "            _pd_kv0 = self.pagetable.metrics.get(\"alloc_kv_only_pages\", 0)\n"
         "            allocated_pages, cached_pages, non_sequential_pages, stashed_recurrent_state = \\\n"
         "                seq.allocate_pages(self.pagetable, self.generator.recurrent_cache, protected_hashes)\n"
         "            if _PREFIX_DIAG:\n"
         "                _prefix_diag_alloc(self, seq, cached_pages,\n"
         "                                   self.pagetable.metrics.get(\"alloc_kv_only_pages\", 0) - _pd_kv0)\n"),
        ("    def deallocate_pages(self):\n        self.free_recurrent_state()\n",
         "    def deallocate_pages(self):\n"
         "        if _PREFIX_DIAG:\n"
         "            _prefix_diag_done(self)\n"
         "        self.free_recurrent_state()\n"),
    ],
}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_prefix_diag: anchor not found exactly once in {rel}")
        s = s.replace(old, new)
    s += HELPERS
    p.write_text(s)
    print("patched", rel)
