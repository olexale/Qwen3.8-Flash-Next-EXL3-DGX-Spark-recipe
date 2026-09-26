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

EXL3_PREFIX_DIAG=2 adds two lines (Plan C, C1), still lengths, positions and classes only:

  [prefix-diag] diverge at L: msg #k role R (tool response yes/no) +o, first user prev U1 new
    U2, last user prev V1 new V2 | prev tail T at new +d (matched M) | classes prev ....|....
    new ....|....

when a prompt diverges inside the previous prompt: which message (counted by <|im_start|>,
role from the token after it), the offset into it, where the first and latest user messages
start, whether the previous prompt's tail after L reappears later in the new prompt (tokens
moved rather than changed; d = tokens inserted), and the classes of the 8 tokens on each side
of L (S special, N whitespace with a newline, W other whitespace, O other). And for a prompt
with no earlier assistant turn (a session's first request):

  [prefix-diag] session start #n prompt P, first user U | shared with #m (HH:MM:SS) S tokens,
    first user there U'

where S is the longest page-aligned prefix shared with any earlier session start, from the
chained page hashes of the last 64 session starts (hashes only, no ids).

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
import time as _pd_time
from collections import deque as _pd_deque
try:
    _PD_LEVEL = int(_pd_os.environ.get("EXL3_PREFIX_DIAG", "0") or 0)
except ValueError:
    _PD_LEVEL = 0
_PREFIX_DIAG = _PD_LEVEL >= 1
_pd_ring = _pd_deque(maxlen = 16)  # (ids int64 cpu tensor, prompt length)
_pd_starts = _pd_deque(maxlen = 64)  # level 2: (n, time, page hashes, first user, prompt length)
_pd_start_count = 0
_pd_marks = None
_pd_struct = None

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

def _pd_struct_ids(tokenizer):
    global _pd_struct
    if _pd_struct is None:
        s = {"special": set(getattr(tokenizer, "extended_id_to_piece", {}).keys()), "roles": {}}
        for key, name in (("im_start", "<|im_start|>"), ("tool_response", "<tool_response>")):
            try:
                s[key] = tokenizer.single_id(name)
            except Exception:
                s[key] = None
        for r in ("system", "user", "assistant"):
            try:
                t = tokenizer.encode(r).flatten().tolist()
                if len(t) == 1:
                    s["roles"][t[0]] = r
            except Exception:
                pass
        _pd_struct = s
    return _pd_struct

def _pd_msgs(ids, st):
    """(position, role, is a tool response) of each <|im_start|> in ids"""
    out = []
    if st["im_start"] is None:
        return out
    n = ids.shape[0]
    for p in (ids == st["im_start"]).nonzero().flatten().tolist():
        role = st["roles"].get(int(ids[p + 1]), "?") if p + 1 < n else "-"
        tr = st["tool_response"] is not None and bool((ids[p + 1 : p + 4] == st["tool_response"]).any())
        out.append((p, role, tr))
    return out

def _pd_users(msgs):
    u = [p for p, role, tr in msgs if role == "user" and not tr]
    return (u[0], u[-1]) if u else (None, None)

def _pd_classes(tokenizer, ids, a, b, st):
    s = ""
    for i in range(max(a, 0), min(b, ids.shape[0])):
        t = int(ids[i])
        if t in st["special"]:
            s += "S"
            continue
        try:
            txt = tokenizer.tokenizer.decode([t])
        except Exception:
            txt = ""
        s += ("N" if "\\n" in txt else "W") if txt and txt.isspace() else "O"
    return s

def _pd_find(hay, needle):
    k = needle.shape[0]
    if k == 0 or hay.shape[0] < k:
        return -1
    hit = (hay.unfold(0, k, 1) == needle).all(dim = 1).nonzero()
    return int(hit[0, 0]) if hit.numel() else -1

def _pd_diverge_line(tokenizer, ids, prev_ids, prev_p, l):
    st = _pd_struct_ids(tokenizer)
    msgs = _pd_msgs(ids, st)
    before = [m for m in msgs if m[0] <= l]
    if before:
        mp, role, tr = before[-1]
        where = f"msg #{len(before)} role {role} (tool response {'yes' if tr else 'no'}) +{l - mp}"
    else:
        where = "before the first message"
    pu = _pd_users(_pd_msgs(prev_ids[:prev_p], st))
    nu = _pd_users(msgs)
    fmt = lambda x: "-" if x is None else str(x)
    tail = prev_ids[l:prev_p]
    d = _pd_find(ids[l:], tail[:32])
    if d >= 0:
        m = _pd_lcp(tail, ids[l + d:])
        moved = f"prev tail {tail.shape[0]} at new +{d} (matched {m})"
    else:
        moved = f"prev tail {tail.shape[0]} not found later"
    cls = lambda x: (_pd_classes(tokenizer, x, l - 8, l, st) + "|" +
                     _pd_classes(tokenizer, x, l, l + 8, st))
    return (f"[prefix-diag] diverge at {l}: {where}, first user prev {fmt(pu[0])} new {fmt(nu[0])}, "
            f"last user prev {fmt(pu[1])} new {fmt(nu[1])} | {moved} | "
            f"classes prev {cls(prev_ids)} new {cls(ids)}")

def _pd_session_start(job, seq, ids):
    """Level 2: log how much of a session's first prompt an earlier session's first prompt
    shares (page granularity, from the chained page hashes)"""
    global _pd_start_count
    st = _pd_struct_ids(job.generator.tokenizer)
    msgs = _pd_msgs(ids, st)
    p = ids.shape[0]
    if any(role == "assistant" and pos < p - 8 for pos, role, _ in msgs):
        return
    hashes = list(seq.page_hashes or [])
    first_user = _pd_users(msgs)[0]
    best = None
    for n, t, h, fu, pp in _pd_starts:
        k = 0
        while k < min(len(h), len(hashes)) and h[k] == hashes[k]:
            k += 1
        if best is None or k >= best[0]:
            best = (k, n, t, fu)
    _pd_start_count += 1
    now = _pd_time.strftime("%H:%M:%S")
    line = f"[prefix-diag] session start #{_pd_start_count} prompt {p}, first user {first_user}"
    if best is None:
        line += " | no earlier session start"
    else:
        k, n, t, fu = best
        line += f" | shared with #{n} ({t}) {k * PAGE_SIZE} tokens, first user there {fu}"
    print(line, flush = True)
    _pd_starts.append((_pd_start_count, now, hashes, first_user, p))

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
            if best is None or l >= best[0]:  # ties: the most recent
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
        if _PD_LEVEL >= 2:
            if best is not None and 0 < best[0] < best[2]:
                print(_pd_diverge_line(job.generator.tokenizer, ids, best[1], best[2], best[0]), flush = True)
            # A requeued continuation extends a kept sequence; it starts no session
            if (not getattr(job, "is_requeued", False) and
                    (best is None or best[0] < best[1].shape[0])):
                _pd_session_start(job, seq, ids)
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
