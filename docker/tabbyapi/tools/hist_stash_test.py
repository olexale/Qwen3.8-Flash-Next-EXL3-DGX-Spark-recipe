#!/usr/bin/env python3
"""Checks for patch_exllamav3_hist_stash.py (EXL3_HIST_STASH), engine, TabbyAPI stopped.

1. Parity, at every checkpoint the patch captures during decode: snapshot the job's
   recurrent state buffers, run the engine's own rewind to the page boundary + stash()
   (what the fork's boundary truncation would store), compare with the history copy bit for
   bit, restore the buffers. Prints PARITY checks/mismatches per layer type.
2. Follow-ups: per conversation (CTX-token prompt, OUT greedy tokens, then NEW new tokens),
   the follow-up's resume position, time to first token and FUP greedy tokens, with the patch
   on and off (off: the recurrent cache is cleared first and the turn replayed, so the
   follow-up resumes at the previous prompt's last page as before). Prints where the two
   follow-ups' greedy tokens diverge, if they do.

  SCRIPT=hist_stash_test.py IMAGE=qwen38-exl3-tabby:histstash \
    docker/tabbyapi/tools/run_engine_bench.sh hs -e EXL3_HIST_STASH=1 -e CHUNK=8192 -e MBS=3

Env: CTX=12000 OUT=400 NEW=350 FUP=128 CONVS=3
"""
import os, time, random
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import generator as gm
from exllamav3.generator.sampler import GreedySampler
from exllamav3.constants import PAGE_SIZE

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
MBS = int(os.environ.get("MBS", "3")); CHUNK = int(os.environ.get("CHUNK", "8192"))
CTX = int(os.environ.get("CTX", "12000")); OUT = int(os.environ.get("OUT", "400"))
NEW = int(os.environ.get("NEW", "350")); FUP = int(os.environ.get("FUP", "128"))
CONVS = int(os.environ.get("CONVS", "3"))
assert gm._HS_ON, "run with EXL3_HIST_STASH=1"
print(f"CONFIG MBS={MBS} CHUNK={CHUNK} CTX={CTX} OUT={OUT} NEW={NEW} FUP={FUP} CONVS={CONVS}", flush=True)

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config); dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=262144, max_batch_size=MBS, max_history=5, **qkw)
dcache = Cache(dm, max_num_tokens=262144, max_batch_size=MBS, max_history=5, **qkw)
dm.load(progressbar=False); model.load(progressbar=False, max_chunk_size=CHUNK, max_batch_size=MBS)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=MBS, max_chunk_size=CHUNK, recurrent_cache_size=8192 * 1024**2,
                num_draft_tokens=5, dynamic_draft_tokens=True, draft_confidence=0.6)
print("loaded", flush=True)

# ---- 1. parity wrapper ----
par = {"checks": 0, "mismatch": 0, "by_type": {}}
orig_capture = gm._hs_capture
def checked_capture(g, job, state, n):
    if not gm._HS_ON or state is None or state.last_history <= 0:
        return orig_capture(g, job, state, n)
    T = state.last_history + 1; p0 = state.position - T; keep = T - n
    b = ((p0 + keep) // PAGE_SIZE) * PAGE_SIZE
    if b <= p0:
        return orig_capture(g, job, state, n)
    k = b - p0
    layers = state.cache.get_all_recurrent_layers()
    snaps = {key: [t[state.slot].clone() for t in l.get_state_tensors()] for key, l in layers.items()}
    orig_capture(g, job, state, n)
    ours = job._hs_pending[1] if getattr(job, "_hs_pending", None) else \
        (job._hs_last[2] if getattr(job, "_hs_last", None) and job._hs_last[1] == b else None)
    pos, lh = state.position, state.last_history
    state.rewind(T - k)
    ref = state.stash()
    for key, l in layers.items():
        for t, s in zip(l.get_state_tensors(), snaps[key]):
            t[state.slot].copy_(s)
    state.position, state.last_history = pos, lh
    if ours is None:
        print(f"PARITY no stash captured at {b}", flush=True); par["mismatch"] += 1; return
    for key, l in layers.items():
        name = type(l).__name__
        ok = all(torch.equal(a.cpu(), r.cpu()) for a, r in zip(ours[key], ref[key]))
        d = par["by_type"].setdefault(name, [0, 0]); d[0] += 1; d[1] += (not ok)
        par["checks"] += 1; par["mismatch"] += (not ok)
        if not ok:
            md = max((a.float().cpu() - r.float().cpu()).abs().max().item() for a, r in zip(ours[key], ref[key]))
            print(f"PARITY MISMATCH {name} {key} at b={b} k={k} T={T} n={n}: max abs diff {md}", flush=True)
    assert ours["position"] == ref["position"] == b, (ours["position"], ref["position"], b)
gm._hs_capture = checked_capture

# ---- 2. follow-ups ----
SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()
src_ids = tok.encode(SRC, add_bos=False)[0]
def text(n, seed):
    random.seed(seed); parts, k = [f"salt {random.random()}\n"], 0
    while k < n:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("".join(parts), add_bos=False)[0][:n]
enc = lambda s: tok.encode(s, add_bos=False, encode_special_tokens=True)[0]

def run(ids, max_new):
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=max_new, min_new_tokens=max_new,
              sampler=GreedySampler())
    t = time.time(); gen.enqueue(job); first = None; out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming":
                if first is None: first = time.time()
                if r.get("token_ids") is not None: out.append(r["token_ids"][0])
    return (first or time.time()) - t, torch.cat(out), job.cached_pages * PAGE_SIZE

run(torch.cat([text(1024, 1), enc("<|im_end|>\n<|im_start|>assistant\n<think>\n")]), 2)
for c in range(CONVS):
    base = torch.cat([enc("<|im_start|>user\n"), text(CTX + 37 * c, 100 + c),
                      enc("\nContinue the text.<|im_end|>\n<|im_start|>assistant\n<think>\n")])
    res = {}
    for arm in ("on", "off"):
        gm._HS_ON = arm == "on"
        gen.recurrent_cache.clear(); gen.recurrent_cache.update_total_size()
        _, ans, _ = run(base, OUT)
        follow = torch.cat([base, ans, enc("<|im_end|>\n<|im_start|>user\n<tool_response>\n"), text(NEW, 900 + c),
                            enc("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")])
        dt, fo, resumed = run(follow, FUP)
        res[arm] = (ans, fo)
        print(f"RESULT conv {c} {arm:>3}: prompt {base.shape[0]} + answer {ans.shape[0]}, follow-up {follow.shape[0]}: "
              f"resumed {resumed} (answer ends {base.shape[0] + ans.shape[0]}), ttft {dt:.3f}s", flush=True)
    gm._HS_ON = True
    a_on, f_on = res["on"]; a_off, f_off = res["off"]
    same_ans = a_on.shape == a_off.shape and torch.equal(a_on, a_off)
    n = min(f_on.shape[0], f_off.shape[0])
    ne = (f_on[:n] != f_off[:n]).nonzero()
    div = int(ne[0, 0]) if ne.numel() else None
    print(f"RESULT conv {c}: answers identical: {same_ans}; follow-up greedy tokens "
          f"{'identical (' + str(n) + ')' if div is None else 'diverge at ' + str(div)}", flush=True)
print(f"PARITY checks {par['checks']} mismatches {par['mismatch']} by type {par['by_type']}", flush=True)
print(f"METRICS {gm._hs_metrics}", flush=True)
print("DONE", flush=True)
