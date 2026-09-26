#!/usr/bin/env python3
"""Offline sizing for Plan D: speculative sampling (D1) and lookup drafting in thinking (D2).

Runs inside the qwen38-exl3-tabby image, TabbyAPI stopped:

    SCRIPT=spec_sample_size.py docker/tabbyapi/tools/run_engine_bench.sh specsize

Synthetic prompts with thinking on (the model's chat template, as TabbyAPI renders them),
sampled with the qwen38 preset (temperature 1.0, top_k 20, top_p 0.95), MTP drafting as
deployed (5 drafts, dynamic, confidence 0.6), prompt lookup off. Nothing here is the owner's
data; the numbers stay in process memory.

Pass 0 (no capture): per-phase decode tok/s and the wall time of a round by draft width.
Pass 1 (capture, dynamic drafting) and pass 2 (capture, a fixed window of 5 drafts): at every
drafted position i of a round, from the target's verify logits and the MTP head's logits
(the 64K-column head slice, as drafted), both filtered like the target's sampler
(temperature, then top-k, then top-p over the top-k set):

    a_cur  = p_i(d_i), d_i = argmax q_i       the current rule (greedy draft, exact match)
    a_spec = sum_x min(p_i(x), q_i(x))        sampled draft + ratio test (1 - TV)

and the expected tokens per round 1 + sum_j prod_{i<j} a_i for both (positions after i are
conditioned on the greedy drafts, as in the run; the usual approximation). q is also
evaluated at lower draft temperatures (Q_TEMPS): any q keeps the output exact, so the best
one is a free parameter. Converting to tok/s uses pass 0's round cost per width: with the
same windows the cost is the same, so the ratio of expected tokens per round is the speedup;
for fixed windows (pass 2) each rule's best window is compared.

D2: for each thinking round in pass 1, a phase-aware prompt lookup (n-gram 3, min match
MIN_MATCHES, draft <= LOOKUP_MAX, gated on the MTP head's first draft token) is simulated on
the finished sequence: its accepted length is the common prefix of the lookup draft with the
tokens actually generated (a sample from the target, like a verify), and a lookup round
replaces that round's MTP result at the lookup width's cost.

Env: PROMPTS=all|name,name  REPS=2  NTOK=900  Q_TEMPS=1.0,0.8,0.6  MIN_MATCHES=4,5,6,8
     LOOKUP_MAX=5  PASSES=0,1,2
"""
import os, time, statistics, collections, json
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler
from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator
import exllamav3.generator.generator as G
import exllamav3.architecture.qwen4_exp_mtp as M

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
REPS = int(os.environ.get("REPS", "2"))
NTOK = int(os.environ.get("NTOK", "900"))
Q_TEMPS = [float(x) for x in os.environ.get("Q_TEMPS", "1.0,0.8,0.6").split(",")]
MIN_MATCHES = [int(x) for x in os.environ.get("MIN_MATCHES", "4,5,6,8").split(",")]
LOOKUP_MAX = int(os.environ.get("LOOKUP_MAX", "5"))
PASSES = [int(x) for x in os.environ.get("PASSES", "0,1,2").split(",")]
TOP_K, TOP_P, TEMP = 20, 0.95, 1.0
NDT = 5

# ---- prompts (thinking on) ----
import jinja2, json as _json
def _raise(m): raise RuntimeError(m)
_env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
_env.globals["raise_exception"] = _raise
_env.filters["tojson"] = lambda x, indent=None, **k: _json.dumps(x, ensure_ascii=False, indent=indent)
_TPL = _env.from_string(open(os.path.join(MODEL, "chat_template.jinja")).read())
def render(msgs, tools=None):
    return _TPL.render(messages=msgs, tools=tools, add_generation_prompt=True)
def _fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": list(props)}}}
PI_TOOLS = [
    _fn("read", "Read the contents of a file.", path="Path to the file"),
    _fn("edit", "Edit a file by replacing exact text. oldText must match the file exactly, including whitespace.",
        path="Path to the file", oldText="Exact text to replace", newText="Replacement text"),
    _fn("write", "Write a file, replacing its whole content.", path="Path to the file", content="New file content"),
    _fn("bash", "Run a shell command.", command="The command"),
]
import exllamav3.generator.draft_confidence as _dc
REAL_PATH = "exllamav3/generator/draft_confidence.py"
REAL = open(_dc.__file__).read()
SYS = "You are a coding agent working in the user's repository. Use the tools to inspect and change files."
def agent(task, tool_out=None, tool_call=None):
    tc = tool_call or {"name": "read", "arguments": {"path": REAL_PATH}}
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": task},
            {"role": "assistant", "content": "", "tool_calls": [{"function": tc}]},
            {"role": "tool", "content": REAL if tool_out is None else tool_out}]
    return render(msgs, PI_TOOLS)
PYTEST_OUT = """============================= test session starts ==============================
collected 12 items

tests/test_confidence.py ....F.......                                    [100%]

=================================== FAILURES ===================================
_________________________ test_estimate_below_lowest_bin _______________________

    def test_estimate_below_lowest_bin():
        cal = DraftConfidenceCalibrator(0.6)
        for s in (5.0, 6.0, 7.0):
            for _ in range(50):
                cal.add_label(s, True)
>       assert cal.estimate(1.0) == 0.0
E       AssertionError: assert 1.0 == 0.0
E        +  where 1.0 = <bound method DraftConfidenceCalibrator.estimate of <...>>(1.0)

tests/test_confidence.py:41: AssertionError
=========================== short test summary info ============================
FAILED tests/test_confidence.py::test_estimate_below_lowest_bin - AssertionError
========================= 1 failed, 11 passed in 0.21s =========================
"""
PROMPTS = {
    "code": render([{"role": "user", "content": "Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example."}]),
    "debug": render([{"role": "user", "content": "This function should return the k most frequent words, ties broken alphabetically, but the output order is sometimes wrong. Find the bug and fix it.\n\n```python\nfrom collections import Counter\n\ndef top_k_words(text: str, k: int) -> list[str]:\n    counts = Counter(text.lower().split())\n    ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)\n    return [w for w, _ in ranked[:k]]\n```"}]),
    "edit": agent("In " + REAL_PATH + ", make estimate() return 0.0 when no populated bin is at or below the score instead of using the nearest bin above, and rename decay_step() to age_step(). Use the edit tool."),
    "failing": agent("The test below fails. Find out why and fix the code with the edit tool.", PYTEST_OUT,
                     {"name": "bash", "arguments": {"command": "python -m pytest tests/test_confidence.py -q"}}),
    "review": agent("Review " + REAL_PATH + ": is the calibration statistically sound? List concrete problems, then propose the smallest fix for the most important one."),
    "devops": render([{"role": "user", "content": "Explain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML."}]),
    "puzzle": render([{"role": "user", "content": "Three boxes are labeled apples, oranges and mixed; every label is wrong. You may draw one fruit from one box without looking inside. How do you relabel all boxes correctly? Explain the reasoning step by step."}]),
}
sel = os.environ.get("PROMPTS", "all")
if sel != "all":
    PROMPTS = {k: v for k, v in PROMPTS.items() if k in sel.split(",")}

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=32768, max_batch_size=2, max_history=NDT, **qkw)
dcache = Cache(dm, max_num_tokens=32768, max_batch_size=2, max_history=NDT, **qkw)
dm.load(progressbar=False)
model.load(progressbar=False, max_chunk_size=8192, max_batch_size=2)
ids = {k: tok.encode(v, add_bos=False, encode_special_tokens=True) for k, v in PROMPTS.items()}
IM_END = tok.single_id("<|im_end|>")
THINK_END = tok.single_id("</think>")
VOCAB = tok.actual_vocab_size
print("CONFIG", f"prompts {', '.join(f'{k}:{v.shape[-1]}' for k, v in ids.items())} REPS={REPS} NTOK={NTOK} "
      f"Q_TEMPS={Q_TEMPS} MIN_MATCHES={MIN_MATCHES} LOOKUP_MAX={LOOKUP_MAX} PASSES={PASSES} "
      f"MTP_HEAD_N={M._MTP_HEAD_N}", flush=True)
if hasattr(G, "_PLD"):
    G._PLD = False

# ---- capture: the MTP head slice's logits per draft step ----
CAP = {"on": False, "draft": [], "rounds": [], "job": None, "plen": 0, "skipped": 0}
_real_ext = M.ext
class _ExtProxy:
    def __getattr__(self, n): return getattr(_real_ext, n)
    def exl3_gemm(self, x, tr, y, *a):
        r = _real_ext.exl3_gemm(x, tr, y, *a)
        if CAP["on"] and y.dim() == 2 and y.shape[-1] == M._MTP_HEAD_N:
            CAP["draft"].append(y.float().clone())
        return r
M.ext = _ExtProxy()

def filt(l, T):
    """(ids, probs) of the sampler's kept set: temperature, top-k, top-p over the top-k set"""
    v, i = torch.topk(l.float() / T, TOP_K, dim=-1)
    pr = torch.softmax(v, dim=-1)
    c = torch.cumsum(pr, dim=-1)
    pr = pr * ((c - pr) < TOP_P)
    return i, pr / pr.sum(dim=-1, keepdim=True)

def in_thinking():
    job = CAP["job"]
    seq = job.sequences[0].sequence_ids
    n = len(seq)
    if n <= CAP["plen"]:
        return True
    return not bool((seq.torch_slice(CAP["plen"], n) == THINK_END).any())

def analyze(L):
    """One verify window: L = target logits (w + 1, V) for the w drafts stashed this round"""
    dr = CAP["draft"]
    L = L[:len(dr), :VOCAB]
    ip, pp = filt(L, TEMP)
    Y = torch.cat(dr, dim=0)
    d = torch.argmax(Y, dim=-1)
    a_cur = (pp * (ip == d[:, None])).sum(-1)
    specs = []
    for T in Q_TEMPS:
        iq, pq = filt(Y, T)
        m = (ip[:, :, None] == iq[:, None, :])
        specs.append((torch.minimum(pp[:, :, None], pq[:, None, :]) * m).sum((1, 2)))
    st = torch.stack([a_cur] + specs, dim=1).cpu().tolist()
    CAP["rounds"].append({"think": in_thinking(), "a": st, "w": len(dr),
                          "pos": len(CAP["job"].sequences[0].sequence_ids), "d0": int(d[0])})

# The verify forward: the target model's forward right after the draft steps (TabbyAPI's
# sampler has penalty steps, so the fork samples a verify window position by position; the
# logits are taken here instead, for every position of the window)
_real_forward = model.forward
def _forward(*a, **k):
    out = _real_forward(*a, **k)
    if CAP["on"] and CAP["draft"]:
        if torch.is_tensor(out) and out.dim() == 3 and out.shape[0] == 1 and out.shape[1] == len(CAP["draft"]) + 1:
            analyze(out[0])
        else:
            CAP["skipped"] += 1
        CAP["draft"].clear()
    return out
model.forward = _forward

def make_sampler():
    # As TabbyAPI builds it: penalty steps (no-ops at their defaults), temperature, top-k, top-p
    return ComboSampler(temperature=TEMP, top_k=TOP_K, top_p=TOP_P)

# ---- one generation ----
def run(gen, name, seed, stats):
    torch.manual_seed(seed)
    job = Job(input_ids=ids[name], max_new_tokens=NTOK, stop_conditions=[IM_END], sampler=make_sampler())
    CAP["job"], CAP["plen"], CAP["draft"] = job, ids[name].shape[-1], []
    gen.enqueue(job)
    widths = []
    _mtp = gen.iterate_draftmodel_mtp_gen
    def track(*a, **k):
        d = _mtp(*a, **k)
        widths.append(0 if d is None else d.shape[-1])
        return d
    gen.iterate_draftmodel_mtp_gen = track
    started = False
    while gen.num_remaining_jobs():
        widths.clear()
        think = in_thinking() if started else True
        n0, a0 = job.new_tokens, job.accepted_draft_tokens
        ta = time.perf_counter()
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
        dt = time.perf_counter() - ta
        if job.new_tokens > max(n0, 0) and started and widths:
            stats.append({"think": think, "w": widths[-1], "dt": dt, "tok": job.new_tokens - max(n0, 0),
                          "acc": job.accepted_draft_tokens - a0})
        if job.new_tokens > 0:
            started = True
    del gen.iterate_draftmodel_mtp_gen
    return job.sequences[0].sequence_ids.torch().view(-1).tolist(), ids[name].shape[-1]

def exp_tokens(a, w):
    e, p = 1.0, 1.0
    for i in range(w):
        p *= a[i]
        e += p
    return e

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=2, max_chunk_size=8192, num_draft_tokens=NDT,
                dynamic_draft_tokens=True, draft_confidence=0.6)
cal = DraftConfidenceCalibrator(0.6)
# warm-up: kernels, and the calibrator on a few runs as a running server would have it
gen.draft_calibrator = cal
for name in PROMPTS:
    run(gen, name, 0, [])
print("warm-up done", flush=True)

cost = collections.defaultdict(list)     # width -> round wall times (pass 0)
phase = {"think": [0, 0.0], "rest": [0, 0.0]}
seqs = []                                 # (name, rep, ids, prompt_len, rounds) of pass 1
results = {}
for ps in PASSES:
    for rep in range(REPS):
        for name in PROMPTS:
            gen.draft_calibrator = None if ps == 2 else cal
            CAP["on"] = ps > 0
            CAP["rounds"] = []
            stats = []
            seq, plen = run(gen, name, 1000 + rep, stats)
            CAP["on"] = False
            if ps == 0:
                for s in stats:
                    cost[s["w"]].append(s["dt"])
                    ph = phase["think" if s["think"] else "rest"]
                    ph[0] += s["tok"]; ph[1] += s["dt"]
                nt = sum(1 for s in stats if s["think"])
                print(f"PASS0 {name:<8} rep {rep}: {len(seq) - plen} tokens, {nt} thinking rounds of {len(stats)}", flush=True)
            else:
                results.setdefault(ps, []).append((name, rep, CAP["rounds"], stats))
                if ps == 1:
                    seqs.append((name, rep, seq, plen, CAP["rounds"]))
                th = [r for r in CAP["rounds"] if r["think"]]
                print(f"PASS{ps} {name:<8} rep {rep}: {len(seq) - plen} tokens, {len(th)} thinking rounds of "
                      f"{len(CAP['rounds'])}", flush=True)

# ---- report ----
def med(v): return statistics.median(v) if v else float("nan")
if 0 in PASSES:
    print("COST round wall ms by draft width (pass 0): " + ", ".join(
        f"w{w}: {1000 * med(v):.1f} (n {len(v)})" for w, v in sorted(cost.items())), flush=True)
    for k, (t, s) in phase.items():
        print(f"PHASE pass 0 {k}: {t} tokens in {s:.2f} s = {t / s if s else 0:.1f} tok/s", flush=True)
cw = {w: med(v) for w, v in cost.items()}

names = ["cur"] + [f"specT{T}" for T in Q_TEMPS]
for ps in sorted(results):
    for label, keep in (("thinking", True), ("rest", False)):
        rounds = [r for _, _, rr, _ in results[ps] for r in rr if r["think"] == keep]
        if not rounds:
            continue
        pos = collections.defaultdict(lambda: [[] for _ in names])
        for r in rounds:
            for i, a in enumerate(r["a"]):
                for j in range(len(names)):
                    pos[i][j].append(a[j])
        print(f"ACCEPT pass {ps} {label}: {len(rounds)} rounds; mean acceptance per position ("
              + " / ".join(names) + ")", flush=True)
        for i in sorted(pos):
            print(f"ACCEPT   pos {i}: " + " / ".join(f"{statistics.fmean(v):.3f}" for v in pos[i])
                  + f"  (n {len(pos[i][0])})", flush=True)
        ex = [sum(exp_tokens([a[j] for a in r["a"]], r["w"]) for r in rounds) / len(rounds) for j in range(len(names))]
        base = ex[0]
        print(f"EXPECT pass {ps} {label}: tokens per round, same windows: " + ", ".join(
            f"{n} {e:.3f} ({100 * (e / base - 1):+.1f}%)" for n, e in zip(names, ex)), flush=True)
        if ps == 1:
            meas = [s["tok"] for _, _, _, st in results[ps] for s in st if s["think"] == keep and s["w"] > 0]
            print(f"EXPECT pass 1 {label}: measured tokens per drafted round {statistics.fmean(meas):.3f} "
                  f"(predicted cur {base:.3f})", flush=True)
        if ps == 2 and cw:
            for j, n in enumerate(names):
                best = []
                for k in range(1, NDT + 1):
                    if k not in cw: continue
                    e = sum(exp_tokens([a[j] for a in r["a"]], k) for r in rounds) / len(rounds)
                    best.append((e / cw[k], k, e))
                if best:
                    tps, k, e = max(best)
                    print(f"WINDOW pass 2 {label} {n}: best fixed window {k} drafts, {e:.3f} tokens/round, "
                          f"{tps:.1f} tok/s (by pass-0 cost); all: " +
                          ", ".join(f"k{kk} {t:.1f}" for t, kk, _ in sorted(best, key=lambda x: x[1])), flush=True)

# ---- D2: phase-aware lookup on pass 1 sequences ----
def ngram_index(seqids, start=0, ng=3):
    idx = collections.defaultdict(list)
    for p in range(max(ng - 1, start + ng - 1), len(seqids)):
        idx[tuple(seqids[p - ng + 1:p + 1])].append(p)
    return idx

def lookup_in(src, idx, qry, n, limit, min_match, max_len, ng=3):
    """Longest match of the suffix of qry[:n] among the ng-gram end positions p < limit of src
    (as patch_exllamav3_pld.py, over all positions, not only the last 4); the draft is what
    followed it in src. Returns (match length, draft) or (0, None)"""
    if n < ng + 1:
        return 0, None
    best_m, best_p = 0, -1
    for p in idx.get(tuple(qry[n - ng:n]), ()):
        if p >= limit:
            break
        m = ng
        while m < 64 and p - m >= 0 and n - 1 - m >= 0 and src[p - m] == qry[n - 1 - m]:
            m += 1
        if m >= best_m:
            best_m, best_p = m, p
    if best_m < min_match:
        return 0, None
    d = src[best_p + 1:best_p + 1 + max_len]
    return (best_m, d) if len(d) > 1 else (0, None)

def simulate(mm, cross):
    """Thinking rounds of pass 1 with a gated lookup (min match mm, <= LOOKUP_MAX drafts) from the
    sequence itself and, with cross, the other prompts' outputs (a cross-request index) when the
    sequence has no match"""
    n_think = hits = hits_x = gated = 0
    base_t = new_t = base_c = new_c = 0.0
    acc_l = []
    for si, (name, rep, seq, plen, rounds) in enumerate(seqs):
        idx = ngram_index(seq)
        others = [(o[2], ngram_index(o[2], o[3])) for o in seqs if o[0] != name] if cross else []
        for r in rounds:
            if not r["think"]:
                continue
            n_think += 1
            pos = r["pos"]
            e_mtp = exp_tokens([a[0] for a in r["a"]], r["w"])
            c_mtp = cw.get(r["w"], cw[max(cw)])
            base_t += e_mtp; base_c += c_mtp
            d = None
            if pos < len(seq):
                _, d = lookup_in(seq, idx, seq, pos, pos - 1, mm, LOOKUP_MAX)
                if d is None and cross:
                    best = (0, None)
                    for src, sidx in others:
                        m, dd = lookup_in(src, sidx, seq, pos, len(src) - 1, mm, LOOKUP_MAX)
                        if dd is not None and m > best[0]:
                            best = (m, dd)
                    d = best[1]
                    hits_x += d is not None
            hits += d is not None
            if d is not None and d[0] == r["d0"]:
                gated += 1
                real = seq[pos:pos + len(d)]
                a = 0
                while a < len(d) and a < len(real) and d[a] == real[a]:
                    a += 1
                acc_l.append(a)
                new_t += a + 1
                new_c += cw.get(len(d), cw[max(cw)])
            else:
                new_t += e_mtp; new_c += c_mtp
    if n_think:
        print(f"LOOKUP thinking {'self+cross' if cross else 'self'} min_match {mm}: {n_think} rounds, "
              f"match in {100 * hits / n_think:.1f}% ({100 * hits_x / n_think:.1f}% cross only), "
              f"used (gate) {100 * gated / n_think:.1f}%, mean accepted {statistics.fmean(acc_l) if acc_l else 0:.2f} "
              f"of <= {LOOKUP_MAX}; est. thinking tok/s x{(new_t / new_c) / (base_t / base_c):.3f}", flush=True)

if seqs and cw:
    for mm in MIN_MATCHES:
        simulate(mm, False)
        simulate(mm, True)
print(f"CAPTURE windows skipped (draft steps vs verify rows mismatch): {CAP['skipped']}", flush=True)
print("DONE", flush=True)
