"""Per-request decode statistics (EXL3_DECODE_STATS=1, on by default): one log line per
finished job that splits decode into the thinking phase and the rest (Plan D, D0).

  [decode-stats] at HH:MM:SS arm A | prompt P, ttft s | tokens N (thinking T, text X, tool C) |
    time thinking s1, text s2, tool s3 | thinking rounds R, drafted D, accepted K, lookup L
    (drafted DL, accepted KL) | text ... | tool ... [| candidate counters]

Phases, by generated token: thinking = up to and including </think> (only when the prompt
ends inside an open <think>, i.e. thinking is on); tool = from the first <tool_call> after
thinking to the end; text = the rest (the answer between </think> and the first tool call).
Tool calls are buffered by TabbyAPI until they parse, so their decode is waiting time too.

Each verify round's wall time (from the end of the previous round, or from the job's first
decode step) is split over the phases of the tokens it accepted, in proportion; a round's
draft counters (rounds, drafted, accepted, lookup rounds) go to the phase of its first token.
Rounds without a draft count as rounds with 0 drafted. Tokens per second of a phase = its
tokens / its time.

"arm" is "-" unless a per-request A/B trial runs (EXL3_TRIAL_AB, Plan D, D4). No text and no
token ids are logged, only counts and times. Nothing changes in generation.

Usage: python3 patch_exllamav3_decode_stats.py [exllamav3 package dir]
Run once at image build time, after patch_exllamav3_pld.py and patch_exllamav3_prefix_diag.py;
exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

JOB_HELPERS = '''

# --- per-request decode statistics (patch_exllamav3_decode_stats.py) ---
import os as _ds_os
_DECODE_STATS = _ds_os.environ.get("EXL3_DECODE_STATS", "1") == "1"
_DS_PHASES = ("thinking", "text", "tool")
_ds_ids = None

def _ds_mark_ids(tokenizer):
    global _ds_ids
    if _ds_ids is None:
        _ds_ids = {}
        for name in ("<think>", "</think>", "<tool_call>"):
            try:
                _ds_ids[name] = tokenizer.single_id(name)
            except Exception:
                _ds_ids[name] = None
    return _ds_ids

def _ds_new(prompt_tail, marks, t_start):
    """Stats for one request. Thinking is on when the prompt ends inside an open <think>"""
    tail = [int(t) for t in prompt_tail]
    o = max((i for i, t in enumerate(tail) if t == marks.get("<think>")), default = -1)
    c = max((i for i, t in enumerate(tail) if t == marks.get("</think>")), default = -1)
    return {
        "think_on": o > c,
        "n": 0, "think_end": None, "tool_start": None,
        "t_start": t_start, "t": t_start,
        "tok": dict.fromkeys(_DS_PHASES, 0), "time": dict.fromkeys(_DS_PHASES, 0.0),
        "cnt": {p: [0, 0, 0, 0, 0, 0] for p in _DS_PHASES},  # rounds, drafted, accepted, lookup rounds, lookup drafted, lookup accepted
        "extra": {},
    }

def _ds_phase(ds, k):
    """Phase of generated token k (1-based)"""
    if ds["think_on"] and (ds["think_end"] is None or k <= ds["think_end"]):
        return "thinking"
    if ds["tool_start"] is not None and k >= ds["tool_start"]:
        return "tool"
    return "text"

def _ds_token(ds, token_id, marks):
    ds["n"] += 1
    k = ds["n"]
    if ds["think_on"] and ds["think_end"] is None:
        if token_id == marks.get("</think>"):
            ds["think_end"] = k
    elif ds["tool_start"] is None and token_id == marks.get("<tool_call>"):
        ds["tool_start"] = k
    ds["tok"][_ds_phase(ds, k)] += 1

def _ds_round(ds, n_tokens, drafted, lookup, now):
    """One verify round that accepted n_tokens tokens (the last n_tokens counted by _ds_token)"""
    if n_tokens <= 0:
        ds["t"] = now
        return
    k0 = ds["n"] - n_tokens + 1
    dt = max(now - ds["t"], 0.0)
    ds["t"] = now
    per = {}
    for k in range(k0, ds["n"] + 1):
        p = _ds_phase(ds, k)
        per[p] = per.get(p, 0) + 1
    for p, m in per.items():
        ds["time"][p] += dt * m / n_tokens
    c = ds["cnt"][_ds_phase(ds, k0)]
    acc = min(n_tokens - 1, drafted)
    c[0] += 1
    c[1] += drafted
    c[2] += acc
    if lookup:
        c[3] += 1
        c[4] += drafted
        c[5] += acc

def _ds_line(ds, arm, prompt_len, ttft, clock):
    tok, tm, cnt = ds["tok"], ds["time"], ds["cnt"]
    n = sum(tok.values())
    parts = [
        f"[decode-stats] at {clock} arm {arm}",
        f"prompt {prompt_len}, ttft {ttft:.2f} s",
        f"tokens {n} (thinking {tok['thinking']}, text {tok['text']}, tool {tok['tool']})",
        f"time thinking {tm['thinking']:.3f} s, text {tm['text']:.3f} s, tool {tm['tool']:.3f} s",
    ]
    for p in _DS_PHASES:
        r, d, a, lr, ld, la = cnt[p]
        parts.append(f"{p} rounds {r}, drafted {d}, accepted {a}, lookup {lr} (drafted {ld}, accepted {la})")
    if ds["extra"]:
        parts.append(", ".join(f"{k} {v}" for k, v in ds["extra"].items()))
    return " | ".join(parts)

def _ds_on_token(job, token_id):
    try:
        ds = getattr(job, "_ds", None)
        if ds is None:
            seq = job.sequences[0]
            n = len(seq.sequence_ids)
            marks = _ds_mark_ids(job.generator.tokenizer)
            t0 = job.time_first_token or time.time()
            ds = job._ds = _ds_new(seq.sequence_ids.torch_slice(max(n - 16, 0), n).flatten().tolist(), marks, t0)
            ds["prompt"] = n
            ds["ttft"] = t0 - job.time_enqueue if getattr(job, "time_enqueue", None) else 0.0
        _ds_token(ds, token_id, _ds_ids)
    except Exception as e:
        job._ds = False
        print(f"[decode-stats] error: {e!r}", flush = True)

def _ds_emit(job):
    try:
        ds = getattr(job, "_ds", None)
        if not ds:
            return
        job._ds = None
        print(_ds_line(ds, getattr(job, "_trial_arm", "-"), ds["prompt"], ds["ttft"],
                       time.strftime("%H:%M:%S")), flush = True)
    except Exception as e:
        print(f"[decode-stats] error: {e!r}", flush = True)
'''

GEN_HELPERS = '''

# --- per-request decode statistics (patch_exllamav3_decode_stats.py) ---
from .job import _DECODE_STATS, _ds_round as _ds_round_

def _ds_on_round(job, n_tokens, drafted, lookup):
    ds = getattr(job, "_ds", None)
    if not ds:
        return
    try:
        _ds_round_(ds, n_tokens, drafted, lookup, time.time())
    except Exception as e:
        job._ds = False
        print(f"[decode-stats] error: {e!r}", flush = True)
'''

edits = {
    "generator/job.py": [
        ("        # Accept token\n"
         "        self.new_tokens += 1\n",
         "        # Accept token\n"
         "        self.new_tokens += 1\n"
         "        if _DECODE_STATS and self.new_tokens >= 1 and getattr(self, \"_ds\", None) is not False:\n"
         "            _ds_on_token(self, next_token_i)\n"),
        ("    def deallocate_pages(self):\n",
         "    def deallocate_pages(self):\n"
         "        if _DECODE_STATS and self.is_finished:\n"
         "            _ds_emit(self)\n"),
    ],
    "generator/generator.py": [
        ("                    completed_jobs.append(job)\n"
         "                accepted_lengths.append(1)\n",
         "                    completed_jobs.append(job)\n"
         "                if _DECODE_STATS:\n"
         "                    _ds_on_round(job, 1, 0, False)\n"
         "                accepted_lengths.append(1)\n"),
        ("                accepted_lengths.append(accepted_length)\n"
         "                j += 1\n",
         "                if _DECODE_STATS and rejected != -1:\n"
         "                    _lk = draft_tokens is not None and bool(getattr(self, \"_pld_round\", False))\n"
         "                    _ds_on_round(job, accepted_length,\n"
         "                                 0 if draft_tokens is None else\n"
         "                                 (getattr(job, \"_pld_drafted\", draft_tokens.shape[-1]) if _lk else draft_tokens.shape[-1]),\n"
         "                                 _lk)\n"
         "                accepted_lengths.append(accepted_length)\n"
         "                j += 1\n"),
    ],
}
helpers = {"generator/job.py": JOB_HELPERS, "generator/generator.py": GEN_HELPERS}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_decode_stats: anchor not found exactly once in {rel}")
        s = s.replace(old, new)
    if rel == "generator/generator.py" and "\nimport time\n" not in s:
        s = s.replace("import logging\n", "import logging\nimport time\n", 1)
    s += helpers[rel]
    p.write_text(s)
    print("patched", rel)
