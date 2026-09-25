#!/usr/bin/env python3
"""Where a speculative decode round goes: per-scope GPU time, host bubbles, kernel tables.

Runs inside the qwen38-exl3-tabby image (SCRIPT=decode_profile.py run_engine_bench.sh <name>),
TabbyAPI stopped. Same setup as draft_sweep.py (MTP draft, dynamic drafting with a calibrator,
the qwen38 preset's sampling). Scopes are record_function ranges added by monkeypatching:

  round            Generator.iterate (one decode round)
  draft            iterate_draftmodel_mtp_gen: the MTP draft chain
    draft_fwd      one MTP forward (input layer + block)
    draft_head     sample_from_state: mixer + head slice + argmax
    draft_host     the rest of the chain (conf readback, calibrator)
  gen              iterate_gen
    verify         model.forward on 1 + drafts rows
    mtp_prefill    draft_model.prefill of accepted positions
    sample         the rest of iterate_gen (sampling, acceptance, bookkeeping)
With MODSCOPES=1, the verify forward is further split by block submodule (attn = GDN or
attention, mlp = MoE, attn_hc/mlp_hc mix and apply_, norms, lm_head, final mixer).

A kernel is attributed to the innermost scope open on the host thread when it was launched
(CUPTI correlation id -> runtime call). GPU idle gaps are attributed to the pair
(scope of the kernel before the gap -> scope of the kernel after it).

Env: NTOK=400 WORKLOAD=code REPS=5 (unprofiled timing runs) NDT=5 CONF=0.6 MODSCOPES=0
     TRACE=/out/decode_trace.json (optional, keep the chrome trace)
"""
import os, time, json, statistics, collections, functools
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler
from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
NTOK = int(os.environ.get("NTOK", "400"))
WORKLOAD = os.environ.get("WORKLOAD", "code")
REPS = int(os.environ.get("REPS", "5"))
NDT = int(os.environ.get("NDT", "5"))
CONF = float(os.environ.get("CONF", "0.6"))
MODSCOPES = os.environ.get("MODSCOPES", "0") == "1"
TRACE = os.environ.get("TRACE", "")
print("CONFIG", f"NTOK={NTOK} WORKLOAD={WORKLOAD} REPS={REPS} NDT={NDT} CONF={CONF} MODSCOPES={MODSCOPES}", flush=True)

NOTHINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPTS = {
    "code": "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example.<|im_end|>\n" + NOTHINK,
    "prose": "<|im_start|>user\nWrite a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm.<|im_end|>\n" + NOTHINK,
}

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=16384, max_batch_size=4, max_history=NDT, **qkw)
dcache = Cache(dm, max_num_tokens=16384, max_batch_size=4, max_history=NDT, **qkw)
dm.load(progressbar=False)
model.load(progressbar=False, max_chunk_size=8192, max_batch_size=4)
ids = tok.encode(PROMPTS[WORKLOAD], add_bos=False, encode_special_tokens=True)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=4, max_chunk_size=8192, num_draft_tokens=NDT,
                dynamic_draft_tokens=True, draft_confidence=CONF)
gen.draft_calibrator = DraftConfidenceCalibrator(CONF)

stats = {}
def run(seed):
    torch.manual_seed(seed)
    job = Job(input_ids=ids, max_new_tokens=NTOK, stop_conditions=[],
              sampler=ComboSampler(temperature=1.0, top_k=20, top_p=0.95))
    gen.enqueue(job); t0 = None; rounds = 0
    while gen.num_remaining_jobs():
        streaming = t0 is not None
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming" and t0 is None: t0 = time.perf_counter()
        rounds += streaming
    dt = time.perf_counter() - t0
    tot = job.accepted_draft_tokens + job.rejected_draft_tokens
    stats.update(rounds=rounds, drafted=tot, accepted=job.accepted_draft_tokens, dt=dt)
    return (NTOK - 1) / dt, 100 * job.accepted_draft_tokens / max(tot, 1), rounds

for s in range(3): run(s)  # warm-up: kernel tuning, calibrator burn-in
tps = []
for rep in range(REPS):
    t, a, n = run(1000 + rep)
    tps.append(t)
    print(f"SAMPLE rep={rep} {t:6.1f} tok/s accept {a:4.0f}% rounds {n} ms/round {1000 * stats['dt'] / n:5.1f}", flush=True)
print(f"RESULT unprofiled {statistics.median(tps):.1f} tok/s median [{min(tps):.1f}-{max(tps):.1f}]", flush=True)

# ---- scopes ----
def scoped(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        with record_function(name):
            return fn(*a, **k)
    return w

G = Generator
G.iterate = scoped("round", G.iterate)
G.iterate_draftmodel_mtp_gen = scoped("draft", G.iterate_draftmodel_mtp_gen)
G.iterate_gen = scoped("gen", G.iterate_gen)
dm.forward = scoped("draft_fwd", dm.forward)
dm.sample_from_state = scoped("draft_head", dm.sample_from_state)
dm.prefill = scoped("mtp_prefill", dm.prefill)
model.forward = scoped("verify", model.forward)
if MODSCOPES:
    from exllamav3.modules.transformer import TransformerBlock
    for b in model.modules:
        if not isinstance(b, TransformerBlock): continue
        kind = type(b.attn).__name__ if b.attn else "none"
        b.attn.forward = scoped("v_" + kind, b.attn.forward)
        b.mlp.forward = scoped("v_moe", b.mlp.forward)
        for hc in ("attn_hc", "mlp_hc"):
            m = getattr(b, hc, None)
            if m is None: continue
            m.mix = scoped("v_hc_mix", m.mix)
            m.apply_ = scoped("v_hc_apply", m.apply_)
        for nm in ("attn_norm", "mlp_norm"):
            m = getattr(b, nm, None)
            if m is not None: m.forward = scoped("v_norm", m.forward)
    lm = model.modules[model.logit_layer_idx]
    lm.forward = scoped("v_lm_head", lm.forward)
    fm = model.modules[model.logit_layer_idx - 1]
    fm.forward = scoped("v_final_mixer", fm.forward)
    model.modules[0].forward = scoped("v_embed", model.modules[0].forward)

run(7)  # warm the wrapped path
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t, a, n = run(1000)
print(f"RESULT profiled {t:.1f} tok/s accept {a:.0f}% rounds {n} drafted {stats['drafted']} accepted {stats['accepted']}", flush=True)
path = TRACE or "/tmp/decode_trace.json"
prof.export_chrome_trace(path)

# ---- analysis ----
ev = json.load(open(path))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
ann, rt, gpu = [], {}, []
for e in ev:
    if e.get("ph") != "X": continue
    c = e.get("cat", "")
    if c == "user_annotation":
        ann.append((e["ts"], e["ts"] + e["dur"], e["name"], e["tid"]))
    elif c in ("cuda_runtime", "cuda_driver"):
        cid = e.get("args", {}).get("correlation")
        if cid is not None: rt[cid] = (e["ts"], e["tid"], e["name"])
    elif c in ("kernel", "gpu_memcpy", "gpu_memset"):
        gpu.append((e["ts"], e["ts"] + e["dur"], e["name"], e.get("args", {}).get("correlation"), c))
gpu.sort()
rounds = sorted([a for a in ann if a[2] == "round"])
# innermost scope open on the launching thread, per correlation id (sweep with a stack per thread)
pts = collections.defaultdict(list)
for s_, e_, n_, tid in ann:
    pts[tid] += [(s_, 0, n_), (e_, 2, n_)]
for cid, (ts, tid, _) in rt.items():
    pts[tid].append((ts, 1, cid))
scope_of = {}
for tid, v in pts.items():
    v.sort(key=lambda x: (x[0], x[1])); st = []
    for ts, typ, x in v:
        if typ == 0: st.append(x)
        elif typ == 2:
            if x in st: del st[len(st) - 1 - st[::-1].index(x)]
        else: scope_of[x] = st[-1] if st else "none"

def round_of(ts):
    lo, hi = 0, len(rounds) - 1
    while lo <= hi:
        m = (lo + hi) // 2
        if rounds[m][0] <= ts <= rounds[m][1]: return m
        if ts < rounds[m][0]: hi = m - 1
        else: lo = m + 1
    return None

# attribute kernels
kscope = []
for s, e, name, cid, cat in gpu:
    r = rt.get(cid)
    sc, rd = ("none", None) if r is None else (scope_of.get(cid, "none"), round_of(r[0]))
    kscope.append((s, e, name, sc, rd, cat))
# keep rounds 2..N-1 (skip the prefill round and the tail)
used = set(range(2, len(rounds) - 1))
K = [k for k in kscope if k[4] in used]
nr = len(used)
t_first = rounds[min(used)][0]; t_last = rounds[max(used)][1]
wall = t_last - t_first
busy_by_scope = collections.Counter(); kern = collections.defaultdict(collections.Counter); kcount = collections.Counter()
for s, e, name, sc, rd, cat in K:
    busy_by_scope[sc] += e - s; kern[sc][name] += e - s; kcount[(sc, name)] += 1
# union busy and gaps (all GPU activity in the window, any round)
W = [k for k in kscope if k[1] > t_first and k[0] < t_last]
W.sort()
busy = 0.0; gaps = collections.Counter(); gapn = collections.Counter(); cur_s, cur_e, cur_sc = None, None, None
for s, e, name, sc, rd, cat in W:
    s = max(s, t_first); e = min(e, t_last)
    if cur_e is None:
        cur_s, cur_e, cur_sc = s, e, sc; continue
    if s > cur_e:
        busy += cur_e - cur_s
        gaps[(cur_sc, sc)] += s - cur_e; gapn[(cur_sc, sc)] += 1
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
    cur_sc = sc
if cur_e is not None: busy += cur_e - cur_s
idle = wall - busy
print(f"PROFILE rounds {nr}  wall {wall/1e3:.1f} ms  = {wall/1e3/nr:.2f} ms/round;  GPU busy {busy/1e3/nr:.2f} ms/round, idle {idle/1e3/nr:.2f} ms/round ({100*idle/wall:.1f}%)")
print("SCOPE GPU time per round (kernels attributed by launch scope), ms")
for sc, v in busy_by_scope.most_common():
    print(f"SCOPE {sc:<16} {v/1e3/nr:7.3f} ms  {100*v/sum(busy_by_scope.values()):5.1f}%")
print("SCOPE idle gaps per round by (before -> after), ms, gaps per round")
for k, v in gaps.most_common(20):
    print(f"SCOPE gap {k[0]:>14} -> {k[1]:<14} {v/1e3/nr:7.3f} ms  {gapn[k]/nr:5.1f}")
# host time per scope (CPU range durations) per round
hs = collections.Counter(); hn = collections.Counter()
for s, e, n, tid in ann:
    rd = round_of(s)
    if rd in used: hs[n] += e - s; hn[n] += 1
print("SCOPE host range per round, ms (count per round)")
for n, v in hs.most_common():
    print(f"SCOPE host {n:<16} {v/1e3/nr:7.3f} ms  ({hn[n]/nr:.2f})")
print("TIME kernels per scope, ms per round, calls per round")
for sc, _ in busy_by_scope.most_common():
    for name, v in kern[sc].most_common(int(os.environ.get("KTOP", "12"))):
        print(f"TIME {sc:<14} {v/1e3/nr:7.3f} ms {kcount[(sc, name)]/nr:6.1f}x  {name[:110]}")
print("DONE", flush=True)
