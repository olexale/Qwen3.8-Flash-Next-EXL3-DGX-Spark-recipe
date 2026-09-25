#!/usr/bin/env python3
"""Where the time to first token of a follow-up turn goes (engine, no TabbyAPI).

The pi-style case: a CTX-token conversation, the model answers OUT tokens (MTP drafting as
deployed), then the next request is that conversation + the answer + NEW tokens (a tool
result). TabbyAPI's log fits these at ~0.66 s + 0.73 ms per new token. Per follow-up this
prints the time to first token and every timed call in it: target forwards (rows, ms), MTP
draft-model prefills, page allocation, recurrent-state restore and stashes, in order. Each
timed call is bracketed by device syncs, so the parts add up to slightly more than the
unprofiled TTFT (printed first).

Runs in the qwen38-exl3-tabby image via run_engine_bench.sh (TabbyAPI stopped):

  SCRIPT=followup_profile.py docker/tabbyapi/tools/run_engine_bench.sh fup -e CHUNK=8192 -e MBS=3

Env: CTX=30000  OUT=300  NEWS=100,350,1000  REPS=2
     NOSPLIT=1  also run each follow-up with the last-page split removed (one forward
                instead of two; the recurrent checkpoint at the prompt's last page is then
                not taken)
"""
import os, sys, time, random, inspect, textwrap
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import job as jobmod
from exllamav3.generator.sampler import GreedySampler

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
MBS = int(os.environ.get("MBS", "3"))
CHUNK = int(os.environ.get("CHUNK", "8192"))
CTX = int(os.environ.get("CTX", "30000"))
OUT = int(os.environ.get("OUT", "300"))
NEWS = [int(x) for x in os.environ.get("NEWS", "100,350,1000").split(",")]
REPS = int(os.environ.get("REPS", "2"))
NOSPLIT = os.environ.get("NOSPLIT", "1") == "1"
NDT = 5
print(f"CONFIG MBS={MBS} CHUNK={CHUNK} CTX={CTX} OUT={OUT} NEWS={NEWS} REPS={REPS}", flush=True)

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=262144, max_batch_size=MBS, max_history=NDT, **qkw)
dcache = Cache(dm, max_num_tokens=262144, max_batch_size=MBS, max_history=NDT, **qkw)
t0 = time.time()
dm.load(progressbar=False)
model.load(progressbar=False, max_chunk_size=CHUNK, max_batch_size=MBS)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=MBS, max_chunk_size=CHUNK, recurrent_cache_size=8192 * 1024**2,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.6)

SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()
src_ids = tok.encode(SRC, add_bos=False)[0]
def text(n, seed):
    random.seed(seed); parts, k = [f"salt {random.random()}\n"], 0
    while k < n:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("".join(parts), add_bos=False)[0][:n]
enc = lambda s: tok.encode(s, add_bos=False, encode_special_tokens=True)[0]
ASK = enc("\nContinue.<|im_end|>\n<|im_start|>assistant\n<think>\n")

def run(ids, max_new, label, quiet=False):
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=max_new, sampler=GreedySampler())
    t = time.time(); gen.enqueue(job); first = None; out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming":
                if first is None: first = time.time()
                if r.get("token_ids") is not None: out.append(r["token_ids"][0])
    dt = (first or time.time()) - t
    if not quiet:
        print(f"RESULT {label:<34} prompt={ids.shape[-1]:>6} ttft={dt:6.3f}s", flush=True)
    return dt, (torch.cat(out) if out else torch.empty(0, dtype=torch.long))

# ---- timing wrappers (active only while TRACE is on) ----
TRACE = {"on": False, "ev": []}
def wrap(obj, name, label, rows=None):
    f = getattr(obj, name)
    def w(*a, **k):
        if not TRACE["on"]:
            return f(*a, **k)
        torch.cuda.synchronize(); t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize()
        n = rows(a, k) if rows else ""
        TRACE["ev"].append((label, n, (time.perf_counter() - t) * 1e3))
        return r
    setattr(obj, name, w)
ids_rows = lambda a, k: (k.get("input_ids") if "input_ids" in k else a[0]).shape[-1]
wrap(model, "forward", "target forward", ids_rows)
wrap(dm, "prefill", "MTP prefill", ids_rows)
wrap(dm, "forward", "MTP forward", ids_rows)
wrap(jobmod.Job, "allocate_pages", "allocate pages")
wrap(jobmod.Job, "prepare_for_queue", "prepare (hash pages)")
wrap(jobmod.Job, "maybe_stash_recurrent", "stash (maybe)")
wrap(gen.cache, "new_from_stashed", "restore recurrent state")

# NOSPLIT variant of Job.prefill: drop the separate forward for the last page
src = textwrap.dedent(inspect.getsource(jobmod.Job.prefill))
anchor = "if prefill_start < last_page_b <= prefill_end:"
assert src.count(anchor) == 1
ns = dict(vars(jobmod))
exec(src.replace(anchor, "if False and prefill_start < last_page_b <= prefill_end:"), ns)
prefill_split, prefill_nosplit = jobmod.Job.prefill, ns["prefill"]

run(torch.cat([text(1024, 1), ASK]), 2, "warmup", quiet=True)
salt = 0
for new in NEWS:
    for variant in (["split", "nosplit"] if NOSPLIT else ["split"]):
        for rep in range(REPS):
            salt += 1
            jobmod.Job.prefill = prefill_split
            base = torch.cat([enc("<|im_start|>user\n"), text(CTX, 1000 + salt), ASK])
            _, ans = run(base, OUT, "turn 1", quiet=True)
            follow = torch.cat([base, ans, enc("<|im_end|>\n<|im_start|>user\n<tool_response>\n"),
                                text(new, 5000 + salt), enc("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")])
            jobmod.Job.prefill = prefill_split if variant == "split" else prefill_nosplit
            # the same follow-up twice: timed without syncs, then traced (pages restored from the
            # cache the same way both times: first run's prefill writes pages the trace run reuses,
            # so the trace run gets a fresh salt instead)
            dt, _ = run(follow, 2, f"new={new} {variant} rep {rep}")
            salt += 1
            base = torch.cat([enc("<|im_start|>user\n"), text(CTX, 1000 + salt), ASK])
            _, ans = run(base, OUT, "turn 1", quiet=True)
            follow = torch.cat([base, ans, enc("<|im_end|>\n<|im_start|>user\n<tool_response>\n"),
                                text(new, 5000 + salt), enc("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")])
            TRACE["on"] = True; TRACE["ev"] = []
            dtt, _ = run(follow, 2, f"new={new} {variant} rep {rep} traced")
            TRACE["on"] = False
            tot = 0
            for label, n, ms in TRACE["ev"]:
                tot += ms
                print(f"  TIME {label:<24} rows={n!s:>5} {ms:8.1f} ms", flush=True)
            print(f"  TIME sum of timed calls {tot:.0f} ms of {dtt*1e3:.0f} ms", flush=True)
jobmod.Job.prefill = prefill_split
print("DONE", flush=True)
