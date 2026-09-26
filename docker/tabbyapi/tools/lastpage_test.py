#!/usr/bin/env python3
"""k*256+1-token prompts and their last-page checkpoint (patch_exllamav3_lastpage.py), engine,
TabbyAPI stopped.

For each EXL3_LASTPAGE level (0 = the fork as it is, 1, 2; switched at runtime, so the patch must
be applied: PATCH=1 applies it inside the container, which needs `-u 0`) and each prompt length
k*256 + d (d = 1, and the controls 0 and 2): a fresh random prompt P, then
  run 1   P cold (greedy, NEW tokens)
          page chain of P's full pages: which pages have their content hash, which recurrent
          checkpoints exist under content hashes and under placeholder hashes
  run 2   P again (the exact re-send)
  run 3   P + ~300 new tokens (a follow-up that diverges right after P)
Prints the resume position and TTFT of each run, and for run 2 whether the first-token logits
and the greedy tokens equal run 1's (the resumed job should see exactly run 1's inputs).

  SCRIPT=lastpage_test.py IMAGE=qwen38-exl3-tabby:convckpt \\
    docker/tabbyapi/tools/run_engine_bench.sh lastpage -u 0 -e HOME=/home/tabby -e PATCH=1 \\
    -v $PWD/docker/tabbyapi/patch_exllamav3_lastpage.py:/tmp/patch_lastpage.py:ro

Env: K=6,20 NEW=32 LEVELS=0,1,2 MBS=3 CHUNK=8192
"""
import os, subprocess, sys, time, random
if os.environ.get("PATCH") == "1":
    subprocess.run([sys.executable, "/tmp/patch_lastpage.py"], check=True)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import job as jm
from exllamav3.generator.pagetable import tensor_hash_checksum, is_content_hash
from exllamav3.generator.sampler import GreedySampler
from exllamav3.constants import PAGE_SIZE

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
MBS = int(os.environ.get("MBS", "3")); CHUNK = int(os.environ.get("CHUNK", "8192"))
KS = [int(x) for x in os.environ.get("K", "6,20").split(",")]
NEW = int(os.environ.get("NEW", "32"))
LEVELS = [int(x) for x in os.environ.get("LEVELS", "0,1,2").split(",")]
patched = hasattr(jm, "_LP_LEVEL")
print(f"CONFIG MBS={MBS} CHUNK={CHUNK} K={KS} NEW={NEW} LEVELS={LEVELS} patched={patched}", flush=True)
if not patched:
    LEVELS = [0]

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

SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()
src_ids = tok.encode(SRC, add_bos=False)[0]
def text(n, seed):
    rnd = random.Random(seed); parts, k = [f"salt {rnd.random()}\n"], 0
    while k < n:
        a = rnd.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("".join(parts), add_bos=False)[0][:n]
enc = lambda s: tok.encode(s, add_bos=False, encode_special_tokens=True)[0]
GEN = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n")

def run(p):
    j = Job(input_ids=p.unsqueeze(0), max_new_tokens=NEW, min_new_tokens=NEW,
            sampler=GreedySampler(), identifier=0, return_logits=True)
    gen.enqueue(j)
    t = time.time(); first = None; out = []; lg = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming":
                first = first or time.time()
                if r.get("token_ids") is not None: out.append(r["token_ids"][0])
                if lg is None and r.get("logits") is not None:
                    lg = r["logits"].reshape(-1, r["logits"].shape[-1])[0].float().cpu()
    return (first or time.time()) - t, torch.cat(out), j.cached_pages * PAGE_SIZE, lg

def chain(p):
    """P's full pages: content-hashed live pages, checkpoints under content / placeholder hashes."""
    pt, rc = gen.pagetable, gen.recurrent_cache
    n = (p.shape[0] - 1) // PAGE_SIZE
    h, hs = None, []
    for i in range(n):
        h = tensor_hash_checksum(p[i * PAGE_SIZE:(i + 1) * PAGE_SIZE].unsqueeze(0), h); hs.append(h)
    live = [i for i, x in enumerate(hs) if pt.get_live_page(x) is not None]
    ck = [dict.get(rc, x)["position"] for x in hs if x in rc]
    carry = [dict.get(rc, x)["position"] for x in hs if x in rc and "mtp_carry" in dict.get(rc, x)]
    ph = sorted(v["position"] for k, v in rc.items() if not is_content_hash(k))
    return (f"full pages {n}, live with content hash {len(live)}"
            f"{'' if len(live) == n else ' (missing ' + str(sorted(set(range(n)) - set(live))) + ')'}; "
            f"checkpoints under content hashes at {ck}, with MTP carry at {carry}; "
            f"under placeholder hashes at {ph}")

def logit_cmp(a, b):
    if a is None or b is None:
        return "no logits"
    if torch.equal(a, b):
        return "bit-identical"
    m = torch.isfinite(a) & torch.isfinite(b); a, b = a[m], b[m]
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return f"max|d| {float((a - b).abs().max()):.4f} KL {float((lb.exp() * (lb - la)).sum()):.2e}"

def same(x, y):
    n = min(x.shape[0], y.shape[0]); ne = (x[:n] != y[:n]).nonzero()
    return "identical" if not ne.numel() else f"diverge at {int(ne[0, 0])}"

run(torch.cat([text(1024, 1), GEN]))
seed = 100
for level in LEVELS:
    if patched:
        jm._LP_LEVEL = level
    for k in KS:
        for d in (1, 0, 2):
            seed += 10
            L = k * PAGE_SIZE + d
            p = torch.cat([text(L - GEN.shape[0], seed), GEN])
            assert p.shape[0] == L
            f = torch.cat([p, text(300, seed + 1), GEN])
            gen.recurrent_cache.clear(); gen.recurrent_cache.update_total_size()
            s0 = dict(jm._lp_stats) if patched else {}
            r1 = run(p)
            c = chain(p)
            r2 = run(p)
            r3 = run(f)
            st = {x: jm._lp_stats[x] - s0[x] for x in s0} if patched else {}
            print(f"RESULT level {level} L {L} (k={k}, +{d}): cold resumed {r1[2]} ttft {r1[0]:.3f}s | "
                  f"re-send resumed {r2[2]} ttft {r2[0]:.3f}s, first logits vs cold {logit_cmp(r1[3], r2[3])}, "
                  f"greedy {same(r1[1], r2[1])} | follow-up {f.shape[0]} resumed {r3[2]} ttft {r3[0]:.3f}s"
                  f"{' | ' + str(st) if st else ''}", flush=True)
            print(f"CHAIN level {level} L {L} after cold: {c}", flush=True)
print("DONE", flush=True)
