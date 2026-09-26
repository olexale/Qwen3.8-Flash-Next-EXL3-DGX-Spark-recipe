"""Anchor checkpoints (EXL3_CONV_CKPT, off by default): recurrent checkpoints where prompts
actually diverge, so a changed turn or a new session does not prefill the shared part again.

A request resumes from the longest cached page prefix whose last page has a stashed recurrent
(GatedDeltaNet) state. Prefill stashes one at the prompt's last page boundary (plus a coarse
grid), so a prompt that shares most of an earlier prompt but diverges before that prompt's
last page resumes from nothing (Plan C, 2026-09-25: pi's turn 2 diverged ~216 tokens before
the end of turn 1 and prefilled all 8.1k tokens; every new pi session prefilled the same
~5k-token system prompt and tools).

- EXL3_CONV_CKPT=1 (C2a, within a session): while prefilling a prompt, also split at the page
  boundary at or before its latest user message (the last <|im_start|>user that is not a
  tool response, nor quoted inside one) and stash there. At most one extra checkpoint per request, in the ordinary
  LRU, not pinned.
- EXL3_CONV_CKPT=2 (C2a + C2b, across sessions): additionally, when the prompt shares at least
  EXL3_CONV_CKPT_MIN tokens (default 2048) of full pages with a recent *different* prompt (one
  it does not extend), split and stash at the end of that shared prefix. These checkpoints go
  into a separate LRU of EXL3_CONV_CKPT_MAX entries (default 4, ~112 MiB each) that does not
  count against sysmem_recurrent_cache, so they never push out a live session's checkpoints,
  and the ordinary LRU never evicts them.

Anchors are only used when they lie past the resumed position, before the prompt's last page
boundary (which the fork stashes anyway), and are not stashed yet (looked up by page hash).
The recent prompts are kept as their chained page hashes only (last 32), in process memory.
With EXL3_PREFIX_DIAG on, one line per request reports anchor checkpoints made, reused and
evicted unused, and one line per anchor made gives its kind, position and the prompt length.

A split moves prefill chunk boundaries, like prefix caching already does; outputs are not
guaranteed bit-identical to an unsplit prefill (see Plan C, gates).

Usage: python3 patch_exllamav3_conv_ckpt.py [exllamav3 package dir]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

JOB_HELPERS = '''

# --- anchor checkpoints (patch_exllamav3_conv_ckpt.py) ---
import os as _cc_os
from collections import deque as _cc_deque
try:
    _CC_LEVEL = int(_cc_os.environ.get("EXL3_CONV_CKPT", "0") or 0)
except ValueError:
    _CC_LEVEL = 0
_CC_MIN = int(_cc_os.environ.get("EXL3_CONV_CKPT_MIN", "2048"))
_CC_MAX = int(_cc_os.environ.get("EXL3_CONV_CKPT_MAX", "4"))
_cc_recent = _cc_deque(maxlen = 32)  # page-hash lists of recent prompts (full pages)
_cc_ids = None
_cc_meta = {}  # anchor key -> [kind, times reused]
_cc_stats = {"made a": 0, "made b": 0, "reused": 0, "evicted unused": 0}

def _cc_token_ids(tokenizer):
    global _cc_ids
    if _cc_ids is None:
        d = {}
        for key, name in (("im_start", "<|im_start|>"), ("tool_response", "<tool_response>"),
                          ("tool_response_end", "</tool_response>")):
            try:
                d[key] = tokenizer.single_id(name)
            except Exception:
                d[key] = None
        try:
            t = tokenizer.encode("user").flatten().tolist()
            d["user"] = t[0] if len(t) == 1 else None
        except Exception:
            d["user"] = None
        _cc_ids = d
    return _cc_ids

def _cc_last_user(ids, t):
    """Position of the last <|im_start|>user that starts a user message: not a tool response,
    and not a chat marker quoted inside a tool response (a file with a chat template in it)"""
    if t["im_start"] is None or t["user"] is None:
        return None
    n = ids.shape[0]
    opens = closes = []
    if t["tool_response"] is not None and t.get("tool_response_end") is not None:
        opens = (ids == t["tool_response"]).nonzero().flatten().tolist()
        closes = (ids == t["tool_response_end"]).nonzero().flatten().tolist()
    last_before = lambda xs, p: max((x for x in xs if x < p), default = -1)
    for p in reversed((ids == t["im_start"]).nonzero().flatten().tolist()):
        if p + 1 < n and int(ids[p + 1]) == t["user"]:
            if t["tool_response"] is not None and bool((ids[p + 2 : p + 4] == t["tool_response"]).any()):
                continue
            if last_before(opens, p) > last_before(closes, p):
                continue
            return p
    return None

def _cc_shared_pages(hashes, recent):
    """Longest full-page prefix shared with a recent prompt that this one does not extend"""
    best = 0
    for h in recent:
        k = 0
        m = min(len(h), len(hashes))
        while k < m and h[k] == hashes[k]:
            k += 1
        if k < len(h):
            best = max(best, k)
    return best

def _cc_anchors(job, seq):
    """(position, kind) anchors for this job's prompt, ascending; computed once per job"""
    a = getattr(job, "_cc_anchor_list", None)
    if a is not None:
        return a
    a = {}
    rc = job.generator.recurrent_cache
    hashes = list(seq.page_hashes or [])
    seqlen = len(seq.sequence_ids) - 1
    try:
        if _CC_LEVEL >= 1:
            ids = seq.sequence_ids.torch()[0, :seqlen]
            p = _cc_last_user(ids, _cc_token_ids(job.generator.tokenizer))
            if p is not None and p >= PAGE_SIZE:
                pos = p // PAGE_SIZE * PAGE_SIZE
                if hashes[pos // PAGE_SIZE - 1] not in rc:
                    a[pos] = "a"
        if _CC_LEVEL >= 2:
            k = _cc_shared_pages(hashes, _cc_recent)
            pos = k * PAGE_SIZE
            if pos >= _CC_MIN and pos not in a and hashes[k - 1] not in rc:
                a[pos] = "b"
            if hashes:
                _cc_recent.append(hashes)
    except Exception as e:
        print(f"[conv-ckpt] error: {e!r}", flush = True)
        a = {}
    job._cc_anchor_list = sorted(a.items())
    return job._cc_anchor_list

def _cc_pick(job, seq, prefill_start, prefill_end):
    """The first anchor inside this chunk, or None. The prompt's last page boundary is left to
    the fork's own split"""
    last_page_b = (len(seq.sequence_ids) - 1) // PAGE_SIZE * PAGE_SIZE
    for pos, kind in _cc_anchors(job, seq):
        if prefill_start < pos <= prefill_end and pos < last_page_b:
            return pos, kind
    return None

def _cc_stash(job, seq, kind):
    pos = seq.kv_position
    if pos == 0 or pos % PAGE_SIZE or job.last_recurrent_checkpoint_pos == pos:
        return
    page = seq.allocated_pages[pos // PAGE_SIZE - 1]
    if page.kv_position != PAGE_SIZE:
        return
    rc = job.generator.recurrent_cache
    job.last_recurrent_checkpoint_pos = pos
    if page.phash in rc:  # stashed meanwhile (a parallel session): keep that one
        rc.get_stashed(page.phash)
        return
    if kind == "b":
        rc.put_conv(page.phash, job.recurrent_state, _CC_MAX)
    else:
        rc.put(page.phash, job.recurrent_state)
    _cc_meta[page.phash] = [kind, 0]
    _cc_stats["made " + kind] += 1
    if globals().get("_PREFIX_DIAG", False):
        print(f"[prefix-diag] anchor made {kind} at {pos}, prompt {len(seq.sequence_ids)}", flush = True)

def _cc_after_alloc(job, seq, cached_pages):
    """Count reuse of an anchor checkpoint and anchors evicted before any reuse"""
    rc = job.generator.recurrent_cache
    if rc is None:
        return
    if cached_pages > 0 and seq.page_hashes:
        key = seq.page_hashes[cached_pages - 1]
        m = _cc_meta.get(key)
        if m is not None:
            m[1] += 1
            _cc_stats["reused"] += 1
            if key in rc.conv_keys:
                rc.conv_keys.move_to_end(key)
    for key in [k for k in _cc_meta if k not in rc]:
        if _cc_meta.pop(key)[1] == 0:
            _cc_stats["evicted unused"] += 1
    if globals().get("_PREFIX_DIAG", False):
        print(f"[prefix-diag] anchors: made a {_cc_stats['made a']} b {_cc_stats['made b']}, "
              f"reused {_cc_stats['reused']}, evicted unused {_cc_stats['evicted unused']}, "
              f"live {len(_cc_meta)} (separate LRU {len(rc.conv_keys)})", flush = True)
'''

REC_PUT_CONV = '''

    def put_conv(self, key, state, max_entries):
        """
        Add an anchor checkpoint to a separate LRU of at most max_entries (patch_exllamav3_conv_ckpt.py).
        These entries do not count against max_size and the ordinary LRU never evicts them
        """
        if key in self:
            self.move_to_end(key)
            return
        while self.conv_keys and len(self.conv_keys) >= max_entries:
            old, _ = self.conv_keys.popitem(last = False)
            popped = self.pop(old, None)
            if popped is not None:
                self.metrics["stash_evictions"] += 1
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        if max_entries <= 0:
            return
        self[key] = state.stash()
        self.conv_keys[key] = True


    def clear(self):
        super().clear()
        self.conv_keys.clear()
'''

edits = {
    "generator/job.py": [
        # Split the chunk at an anchor
        ("            # For recurrent models, do a separate forward pass for the last page to get the latest possible checkpoint\n"
         "            recurrent_last_page = False\n",
         "            # Anchor checkpoints (patch_exllamav3_conv_ckpt.py): end the chunk at an anchor and stash there\n"
         "            _cc_anchor = None\n"
         "            if _CC_LEVEL and self.generator.recurrent_cache is not None and not self.embeddings:\n"
         "                _cc_anchor = _cc_pick(self, seq, prefill_start, prefill_end)\n"
         "                if _cc_anchor is not None:\n"
         "                    prefill_end = _cc_anchor[0]\n"
         "                    prefill_ids = seq.sequence_ids.torch_slice(prefill_start, prefill_end)\n"
         "\n"
         "            # For recurrent models, do a separate forward pass for the last page to get the latest possible checkpoint\n"
         "            recurrent_last_page = False\n"),
        ("                if recurrent_last_page:\n"
         "                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)\n",
         "                if recurrent_last_page:\n"
         "                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)\n"
         "                if _cc_anchor is not None:\n"
         "                    _cc_stash(self, seq, _cc_anchor[1])\n"),
        # Reuse / eviction counters
        ("            # Metrics\n"
         "            self.cached_pages += cached_pages\n",
         "            if _CC_LEVEL:\n"
         "                _cc_after_alloc(self, seq, cached_pages)\n"
         "\n"
         "            # Metrics\n"
         "            self.cached_pages += cached_pages\n"),
    ],
    "cache/recurrent.py": [
        ("        self.current_size = 0\n        self.model = model\n",
         "        self.current_size = 0\n        self.model = model\n"
         "        # Anchor checkpoints in their own LRU (patch_exllamav3_conv_ckpt.py)\n"
         "        self.conv_keys = OrderedDict()\n"),
        # The ordinary LRU skips anchor-LRU entries when evicting
        ("                if pt is not None:\n"
         "                    for k in self:\n"
         "                        if not pt.is_resumable(k):\n",
         "                if pt is not None:\n"
         "                    for k in self:\n"
         "                        if k not in self.conv_keys and not pt.is_resumable(k):\n"),
        ("                    popped_key, popped = self.popitem(last = False)\n",
         "                    popped_key = next(k for k in self if k not in self.conv_keys)\n"
         "                    popped = self.pop(popped_key)\n"),
        ("            popped = self.pop(k)\n"
         "            self.metrics[\"stash_pruned\"] += 1\n",
         "            popped = self.pop(k)\n"
         "            self.conv_keys.pop(k, None)\n"
         "            self.metrics[\"stash_pruned\"] += 1\n"),
        # ... and does not count them against max_size
        ("        for v in self.values():\n"
         "            if id(v) in seen:\n",
         "        for k, v in self.items():\n"
         "            if k in self.conv_keys:\n"
         "                continue\n"
         "            if id(v) in seen:\n"),
        ("    def prune_stranded(self) -> int:\n",
         REC_PUT_CONV.lstrip("\n") + "\n\n    def prune_stranded(self) -> int:\n"),
    ],
}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_conv_ckpt: anchor not found exactly once in {rel}:\n{old}")
        s = s.replace(old, new)
    if rel == "generator/job.py":
        s += JOB_HELPERS
    p.write_text(s)
    print("patched", rel)
