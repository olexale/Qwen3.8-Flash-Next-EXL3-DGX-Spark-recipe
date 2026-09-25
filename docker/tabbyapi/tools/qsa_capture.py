#!/usr/bin/env python3
"""Capture real inputs of the QSA sparse attention kernel (qsa_sparse_attend_rows) during
cold prefills, so kernel variants can be A/B tested offline (qsa_ab.py) with TabbyAPI running.

One attention layer (LAYER-th QSA call of each forward) per forward pass is saved, for a cold
600-token prompt and a cold PROMPT-token prompt (chunked like TabbyAPI). Files go to
/out/qsa_caps/NN.pt (mount a writable dir at /out).

Runs inside the image (SCRIPT=qsa_capture.py run_engine_bench.sh <name> -v ~/scratch/out:/out),
TabbyAPI stopped. Env: PROMPT=20000  CHUNK=8192  LAYER=5
"""
import os, random
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.attention_fn.qsa_triton as QT

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
PROMPT = int(os.environ.get("PROMPT", "20000")); CHUNK = int(os.environ.get("CHUNK", "8192"))
LAYER = int(os.environ.get("LAYER", "5"))
OUT = "/out/qsa_caps"; os.makedirs(OUT, exist_ok=True)
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

n_layers = sum(1 for m in model.modules if "Attention" in type(getattr(m, "attn", None)).__name__)
calls, saved = [0], []
_orig = QT.qsa_sparse_attend_rows
def cap(q, k, v, indices, sm_scale, block_table=None, page_size=0, qc=None, n_kv_heads=None):
    o = _orig(q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads)
    i = calls[0]; calls[0] += 1
    if i % n_layers == LAYER and q.shape[0] > 1:
        cpu = lambda t: None if t is None else t.detach().cpu()
        d = dict(q=cpu(q), k=cpu(k), v=cpu(v), indices=cpu(indices), sm_scale=sm_scale,
                 block_table=cpu(block_table), page_size=page_size, n_kv_heads=n_kv_heads,
                 qc=None if qc is None else (cpu(qc[0]), cpu(qc[1]), qc[2], qc[3]), out=cpu(o))
        path = f"{OUT}/{len(saved):02d}.pt"; torch.save(d, path); saved.append(path)
        valid = (indices >= 0).sum(1).float()
        print(f"CAPTURE {path}: rows {q.shape[0]}, K_pad {indices.shape[1]}, valid/row "
              f"min {valid.min().item():.0f} mean {valid.mean().item():.0f} max {valid.max().item():.0f}, "
              f"paged {block_table is not None}, qc {None if qc is None else qc[2:]}", flush=True)
    return o
QT.qsa_sparse_attend_rows = cap
for n, seed in ((600, 21), (PROMPT, 22)):
    job = Job(input_ids=build(n, seed).unsqueeze(0), max_new_tokens=1, sampler=GreedySampler())
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
QT.qsa_sparse_attend_rows = _orig
print(f"RESULT {len(saved)} captures, {calls[0]} QSA calls, {n_layers} attention layers", flush=True)
print("DONE", flush=True)
