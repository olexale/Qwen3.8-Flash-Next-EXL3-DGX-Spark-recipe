#!/usr/bin/env python3
"""GatedResidual.mix in prefill: fused collapse (ext.gr_collapse) vs the torch expression,
on real stream stacks captured during a prefill. Reports time per call and the largest
difference in the mixed output (fp16) and in the post gates.

Runs inside the image with a build that has patch_exllamav3_gr_collapse.py
(SCRIPT=gr_collapse_ab.py run_engine_bench.sh <name>), TabbyAPI stopped.
"""
import os, time, random
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.hyperconnections as HC

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
config = Config.from_directory(MODEL); tok = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=16384, max_batch_size=1, layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
model.load(progressbar=False, max_chunk_size=8192, max_batch_size=1)
gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1, max_chunk_size=8192)
SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read(); src_ids = tok.encode(SRC, add_bos=False)[0]
def build(n, seed):
    random.seed(seed); parts, k = [f"Session {seed} salt {random.random()}\n"], 0
    while k < n:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("<|im_start|>user\n" + "".join(parts), add_bos=False)[0][:n]
caps = []
_mix = HC.GatedResidual._mix
def cap(self, streams, cached=True):
    if streams.shape[0] * streams.shape[1] > 32 and len(caps) < 40 and random.random() < 0.3:
        caps.append((self, streams.clone()))
    return _mix(self, streams, cached)
HC.GatedResidual._mix = cap
job = Job(input_ids=build(8300, 5).unsqueeze(0), max_new_tokens=1, sampler=GreedySampler())
gen.enqueue(job)
while gen.num_remaining_jobs():
    for r in gen.iterate(): pass
HC.GatedResidual._mix = _mix
print(f"captured {len(caps)} mix inputs, rows {sorted(set(s.shape[1] for _, s in caps))}", flush=True)

@torch.inference_mode()
def run(site, s, mode):
    HC._GR_COLLAPSE = mode != "torch"
    return site._mix(s, cached=False)
modes = ("torch", "fused")
tt = dict.fromkeys(modes, 0.0); worst = dict.fromkeys(modes, 0.0); same = dict.fromkeys(modes, 0)
diff_el = dict.fromkeys(modes, 0); tot_el = 0
for site, s in caps:
    for m in modes:
        for _ in range(2): run(site, s, m)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5): run(site, s, m)
        torch.cuda.synchronize(); tt[m] += (time.perf_counter() - t) / 5
    ref = run(site, s, "torch")[1]; tot_el += ref.numel()
    for m in modes[1:]:
        out = run(site, s, m)[1]
        worst[m] = max(worst[m], (ref.float() - out.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-9))
        same[m] += int(torch.equal(ref, out)); diff_el[m] += int((ref != out).sum().item())
rows = sorted(set(s.shape[1] for _, s in caps))
print(f"RESULT {len(caps)} calls, rows {rows[0]}..{rows[-1]}: torch {tt['torch']*1e3:.1f} ms", flush=True)
for m in modes[1:]:
    print(f"RESULT {m:<5} {tt[m]*1e3:.1f} ms (x{tt['torch']/tt[m]:.2f}); calls bit-identical {same[m]}/{len(caps)}; "
          f"elements differing {diff_el[m]}/{tot_el} ({100*diff_el[m]/tot_el:.4f}%); max rel diff {worst[m]:.2e}", flush=True)
HC._GR_COLLAPSE = True
print("DONE", flush=True)
