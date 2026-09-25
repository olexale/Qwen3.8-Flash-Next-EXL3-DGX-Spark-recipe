#!/usr/bin/env python3
"""End-to-end parity of a run-time switch: the same cold prompts prefilled with TOGGLE=0 and
TOGGLE=1 in one process (page table reset in between, so nothing is reused), greedy, the
logits of the first NEW tokens compared bit for bit. Also times each prefill.

Runs inside the image (SCRIPT=prefill_parity.py run_engine_bench.sh <name> [dev mounts]),
TabbyAPI stopped. Env: TOGGLE=EXL3_QSA_STAGE  SIZES=20000,3000  NEW=4  CHUNK=8192
"""
import os, time, random
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
TOGGLE = os.environ.get("TOGGLE", "EXL3_QSA_STAGE")
SIZES = [int(x) for x in os.environ.get("SIZES", "20000,3000").split(",")]
NEW = int(os.environ.get("NEW", "4")); CHUNK = int(os.environ.get("CHUNK", "8192"))
config = Config.from_directory(MODEL); tok = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=65536, max_batch_size=1, layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
model.load(progressbar=False, max_chunk_size=CHUNK, max_batch_size=1)
gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1, max_chunk_size=CHUNK)
SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read(); src_ids = tok.encode(SRC, add_bos=False)[0]
def build(n, seed):
    random.seed(seed); parts, k = [f"Session {seed} salt {random.random()}\n"], 0
    while k < n:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("<|im_start|>user\n" + "".join(parts), add_bos=False)[0][:n]
def run(ids, flag):
    os.environ[TOGGLE] = flag
    gen.pagetable.reset_page_table()
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=NEW, sampler=GreedySampler(), return_logits=True)
    torch.cuda.synchronize(); t = time.perf_counter(); gen.enqueue(job)
    logits, toks, ttft = [], [], None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if "logits" in r:
                if ttft is None: torch.cuda.synchronize(); ttft = time.perf_counter() - t
                logits.append(r["logits"].float().cpu())
            if "token_ids" in r: toks.append(r["token_ids"].flatten().cpu())
    return torch.cat([l.reshape(-1, l.shape[-1]) for l in logits]), torch.cat(toks), ttft
run(build(1024, 1), "1")
ok = True
for n in SIZES:
    ids = build(n, 30 + n)
    la, ta, sa = run(ids, "0"); lb, tb, sb = run(ids, "1")
    m = min(la.shape[0], lb.shape[0])
    same = torch.equal(la[:m], lb[:m]) and torch.equal(ta, tb)
    diff = (la[:m] - lb[:m]).abs().max().item() if m else float("nan")
    ok &= same
    print(f"RESULT {n} tokens: {TOGGLE}=0 first token {sa:.2f}s, =1 {sb:.2f}s; logit rows {m}, "
          f"bit-identical {same} (max abs diff {diff:.3g}), tokens {ta.tolist()} vs {tb.tolist()}", flush=True)
print(f"RESULT all bit-identical: {ok}", flush=True)
print("DONE", flush=True)
