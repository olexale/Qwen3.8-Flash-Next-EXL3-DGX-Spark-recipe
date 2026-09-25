#!/usr/bin/env python3
"""Dump greedy logits for a bit-identity check between two builds or settings that must be
fixed at load time (a separate process each). Two phases on greedy_ab.py's DevOps prompt:

  plain  no draft model: the prompt prefill's logits, then NEW one-row decode steps
  mtp    the deployed draft setup (MTP, 5 tokens, dynamic 0.6): logits of every emitted token,
         i.e. rows of multi-row verify forwards

  SCRIPT=logit_dump.py run_engine_bench.sh ld_a -v ~/scratch/out:/out -e TAG=a
  SCRIPT=logit_dump.py run_engine_bench.sh ld_b -v ~/scratch/out:/out -e TAG=b -e SOME_SETTING=1
  SCRIPT=logit_dump.py run_engine_bench.sh ld_cmp -v ~/scratch/out:/out -e COMPARE=a,b

Env: NEW=64  PHASES=plain,mtp
"""
import os
import torch
OUT = os.environ.get("OUT", "/out")
if os.environ.get("COMPARE"):
    a, b = os.environ["COMPARE"].split(",")
    A = torch.load(f"{OUT}/logits_{a}.pt"); B = torch.load(f"{OUT}/logits_{b}.pt")
    for k in A:
        x, y = A[k], B[k]
        m = min(x.shape[0], y.shape[0])
        rows = [i for i in range(m) if not torch.equal(x[i], y[i])]
        first = rows[0] if rows else None
        d = (x[:m] - y[:m]).abs().max().item() if m else float("nan")
        print(f"RESULT {k:<6} rows {m}, differing rows {len(rows)}, first differing row {first}, max abs diff {d:.3g}")
    raise SystemExit
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler
MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
TAG = os.environ.get("TAG", "run"); NEW = int(os.environ.get("NEW", "64"))
PHASES = os.environ.get("PHASES", "plain,mtp").split(",")
P = ("<|im_start|>user\nExplain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to "
     "scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML.<|im_end|>\n"
     "<|im_start|>assistant\n<think>\n\n</think>\n\n")
config = Config.from_directory(MODEL); tok = Tokenizer.from_config(config)
model = Model.from_config(config); dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
c1 = Cache(model, max_num_tokens=8192, max_batch_size=4, max_history=5, **qkw)
c2 = Cache(model, max_num_tokens=8192, max_batch_size=4, max_history=5, **qkw)
dc = Cache(dm, max_num_tokens=8192, max_batch_size=4, max_history=5, **qkw)
dm.load(progressbar=False); model.load(progressbar=False, max_chunk_size=8192, max_batch_size=4)
ids = tok.encode(P, add_bos=False, encode_special_tokens=True)
out = {}
def run(gen, name):
    job = Job(input_ids=ids, max_new_tokens=NEW, sampler=GreedySampler(), stop_conditions=[], return_logits=True)
    gen.enqueue(job); L = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if "logits" in r: L.append(r["logits"].float().cpu().reshape(-1, r["logits"].shape[-1]))
    out[name] = torch.cat(L)
    print(f"RESULT {TAG} {name}: {out[name].shape[0]} logit rows", flush=True)
if "plain" in PHASES:
    run(Generator(model=model, cache=c1, tokenizer=tok, max_batch_size=4, max_chunk_size=8192), "plain")
if "mtp" in PHASES:
    run(Generator(model=model, cache=c2, tokenizer=tok, draft_model=dm, draft_cache=dc, max_batch_size=4,
                  max_chunk_size=8192, num_draft_tokens=5, dynamic_draft_tokens=True, draft_confidence=0.6), "mtp")
torch.save(out, f"{OUT}/logits_{TAG}.pt")
print("DONE", flush=True)
