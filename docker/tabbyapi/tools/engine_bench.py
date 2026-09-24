#!/usr/bin/env python3
"""TTFT / prefill A/B on the exllamav3 Generator, built the way TabbyAPI builds it.

Runs inside the qwen38-exl3-tabby image (see run_engine_bench.sh), without
TabbyAPI in the loop. Prints one RESULT line per prompt: time to first token
and prompt tokens / TTFT. Workload: a 1k warm-up, cold prompts of SIZES tokens,
then a 20k conversation and a follow-up turn that adds ~600 tokens to it.

Env knobs (defaults = TabbyAPI 2186cdb with the pre-2026-09-23 config.yml;
the shipped config.yml uses CHUNK=8192):
  MBS=4          max_batch_size (Cache, model.load, Generator)
  CHUNK=2048     max_chunk_size
  LOAD_MBS=1     pass max_chunk_size/max_batch_size to model.load like TabbyAPI (0 = don't)
  RCACHE_MB=4096 recurrent_cache_size
  VISION=1       load the vision tower too
  SIZES=600,3000,12000,20000,40000   cold prompt sizes
  PROFILE=0      >0: also torch-profile one cold prompt of that many tokens and
                 print the top kernels / CPU ops and copy_ shapes
  EXL3_* (read by exllamav3 itself)
"""
import os, sys, time, random
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
MBS = int(os.environ.get("MBS", "4"))
CHUNK = int(os.environ.get("CHUNK", "2048"))
LOAD_MBS = os.environ.get("LOAD_MBS", "1") == "1"
RCACHE = int(os.environ.get("RCACHE_MB", "4096")) * 1024**2
VISION = os.environ.get("VISION", "1") == "1"
CS = 262144
NDT = 5
tag = " ".join(f"{k}={os.environ.get(k, d)}" for k, d in
               (("MBS", "4"), ("CHUNK", "2048"), ("LOAD_MBS", "1"), ("RCACHE_MB", "4096"),
                ("VISION", "1"), ("EXL3_NGRAM_STREAM", "1")))
print("CONFIG", tag, flush=True)

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
vm = Model.from_config(config, component="vision") if VISION and "vision" in config.model_classes else None
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=CS, max_batch_size=MBS, max_history=NDT, **qkw)
dcache = Cache(dm, max_num_tokens=CS, max_batch_size=MBS, max_history=NDT, **qkw)
t0 = time.time()
if vm: vm.load(progressbar=False)
dm.load(progressbar=False)
lkw = dict(max_chunk_size=CHUNK, max_batch_size=MBS) if LOAD_MBS else {}
model.load(progressbar=False, **lkw)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=MBS, max_chunk_size=CHUNK, recurrent_cache_size=RCACHE,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.6)

SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()
src_ids = tok.encode(SRC, add_bos=False)[0]

def build(n_tokens, seed):
    """Unique prompt of ~n_tokens: salted header + slices of the source text."""
    random.seed(seed)
    parts, n = [f"Session {seed} salt {random.random()}\n"], 0
    while n < n_tokens:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); n += 512
    ids = tok.encode("<|im_start|>user\n" + "".join(parts), add_bos=False)[0][:n_tokens]
    return ids

def tail(ids_text):
    return tok.encode(ids_text, add_bos=False)[0]

END = tail("\nSummarize the above in one line.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

def ttft(ids, label):
    ids = ids.unsqueeze(0)
    job = Job(input_ids=ids, max_new_tokens=2, sampler=GreedySampler())
    t = time.time(); gen.enqueue(job); first = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if first is None and r.get("stage") == "streaming":
                first = time.time()
    first = first or time.time()
    dt = first - t
    print(f"RESULT {label:<28} prompt={ids.shape[-1]:>6} ttft={dt:6.2f}s  -> {ids.shape[-1]/dt:6.0f} tok/s overall", flush=True)
    return ids[0]

import threading
low = {"avail": 1 << 60}
def watch():
    while True:
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f}
        low["avail"] = min(low["avail"], m["MemAvailable"]); time.sleep(0.2)
threading.Thread(target=watch, daemon=True).start()

ttft(torch.cat([build(1024, 1), END]), "warmup 1k")
for n in [int(x) for x in os.environ.get("SIZES", "600,3000,12000,20000,40000").split(",") if x]:
    ttft(torch.cat([build(n, 100 + n), END]), f"cold {n}")
base = torch.cat([build(20000, 7), END])
ttft(base, "cold 20k (turn 1)")
ttft(torch.cat([base, tail("Sure.<|im_end|>\n<|im_start|>user\n"), build(600, 8)[3:], END]), "turn 2: 20k cached + 600 new")
print(f"MEM min MemAvailable {low['avail']/1024**2:.1f} GiB, torch peak alloc {torch.cuda.max_memory_allocated()/1024**3:.1f} GiB", flush=True)

PROFILE = int(os.environ.get("PROFILE", "0"))
if PROFILE:
    from torch.profiler import profile, ProfilerActivity
    ids = torch.cat([build(PROFILE, 3), END])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        ttft(ids, f"profiled {PROFILE}")
        torch.cuda.synchronize()
    ev = prof.key_averages()
    print(f"PROFILE self GPU {sum(e.self_device_time_total for e in ev)/1e6:.2f}s "
          f"self CPU {sum(e.self_cpu_time_total for e in ev)/1e6:.2f}s")
    print(ev.table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=80))
    print(ev.table(sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=80))
    ev2 = prof.key_averages(group_by_input_shape=True)
    for e in sorted([e for e in ev2 if e.key == "aten::copy_"], key=lambda e: -e.self_device_time_total)[:8]:
        print(f"SHAPE copy_ gpu={e.self_device_time_total/1e6:.2f}s cpu={e.cpu_time_total/1e6:.2f}s "
              f"count={e.count} shapes={e.input_shapes}")
print("DONE", tag, flush=True)
