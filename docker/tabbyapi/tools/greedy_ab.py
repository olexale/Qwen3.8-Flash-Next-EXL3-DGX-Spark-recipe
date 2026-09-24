#!/usr/bin/env python3
"""Greedy A/B for engine changes: the deployed draft setup (MTP, 5 tokens, dynamic, 0.6),
400 greedy tokens on the code, DevOps and prose prompts. Writes token ids, decode rate and
draft acceptance to /out/greedy_<TAG>.json; run once per variant and compare with
COMPARE=a,b (no model load).

  SCRIPT=greedy_ab.py run_engine_bench.sh gA -e TAG=off -e EXL3_MOE_FUSED_UNIFORM=0 -v $PWD/logs:/out
  SCRIPT=greedy_ab.py run_engine_bench.sh gB -e TAG=on -v $PWD/logs:/out
  SCRIPT=greedy_ab.py run_engine_bench.sh cmp -e COMPARE=off,on -v $PWD/logs:/out
"""
import os, json, time
OUT = os.environ.get("OUT", "/out")
if os.environ.get("COMPARE"):
    a, b = os.environ["COMPARE"].split(",")
    A = json.load(open(f"{OUT}/greedy_{a}.json")); B = json.load(open(f"{OUT}/greedy_{b}.json"))
    for k in A:
        x, y = A[k]["ids"], B[k]["ids"]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"RESULT {k:<7} first diverging token {n:>4} of {min(len(x), len(y))}  "
              f"{a}: {A[k]['tps']:5.1f} tok/s {A[k]['acc']:4.1f}%   {b}: {B[k]['tps']:5.1f} tok/s {B[k]['acc']:4.1f}%")
    raise SystemExit
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler
MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
TAG = os.environ.get("TAG", "run"); N = int(os.environ.get("NTOK", "400"))
PROMPTS = {
 "code": "Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each.",
 "devops": "Explain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML.",
 "prose": "Write a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm. Use varied sentence structure and specific sensory details.",
}
config = Config.from_directory(MODEL); tok = Tokenizer.from_config(config)
model = Model.from_config(config); dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=16384, max_batch_size=4, max_history=5, **qkw)
dcache = Cache(dm, max_num_tokens=16384, max_batch_size=4, max_history=5, **qkw)
dm.load(progressbar=False); model.load(progressbar=False, max_chunk_size=8192, max_batch_size=4)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=4, max_chunk_size=8192, num_draft_tokens=5,
                dynamic_draft_tokens=True, draft_confidence=0.6)
def run(p, n):
    ids = tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_bos=False)
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job); out = []; t0 = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                if t0 is None: t0 = time.perf_counter()
                if r.get("token_ids") is not None: out += r["token_ids"].view(-1).tolist()
    dt = time.perf_counter() - t0
    tot = job.accepted_draft_tokens + job.rejected_draft_tokens
    return out, (len(out) - 1) / dt, 100 * job.accepted_draft_tokens / max(tot, 1)
run("Say hi.", 16)
res = {}
for k, p in PROMPTS.items():
    run(p, 16)
    ids, tps, acc = run(p, N)
    res[k] = {"ids": ids, "tps": tps, "acc": acc}
    print(f"RESULT {TAG} {k:<7} {len(ids)} tok {tps:5.1f} tok/s accept {acc:4.1f}%", flush=True)
json.dump(res, open(f"{OUT}/greedy_{TAG}.json", "w"))
print("DONE", flush=True)
