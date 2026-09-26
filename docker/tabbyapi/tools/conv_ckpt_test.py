#!/usr/bin/env python3
"""Checks for patch_exllamav3_conv_ckpt.py (EXL3_CONV_CKPT), engine, TabbyAPI stopped.

Per conversation, pi-like token prompts: a ~SYS-token system prompt, then
  p1   user (a ~180-token block + task), generation prompt          (turn 1)
  p2   user (task only), assistant tool call, tool result, gen      (turn 2: diverges after C2a's anchor)
  p3   the system prompt up to ~60%, then different text, user, gen (diverges before every anchor)
  p3b  the same as p3 up to its divergence from p1, then its own text (resumes at p3's C2b anchor)
  p4a, p4b  two new sessions whose system prompt differs from p1's in its last 10% (another
       working directory), enqueued together: both reach the same anchor at once
Arms: "on" (EXL3_CONV_CKPT=2 behaviour: p1, then p2, p3, p3b, p4a+p4b in that order, cache kept),
"prefix" (anchors off; an earlier prompt that ends at the anchor gives the fork's own checkpoint
there, as prefix caching does today), "chunked" (anchors off, a cold prefill whose first chunk ends
at the anchor: the same forward shapes as resuming there, so its first-token logits should equal
"on" bit for bit), "cold" and "cold2" (anchors off, recurrent cache cleared before each prompt: the reference, twice,
for the run-to-run noise floor of greedy decoding with MTP). Prints the resume position and TTFT
per prompt, where the FUP greedy tokens of "on" and "cold2" diverge from "cold", and how the
first sampled position's logits differ (greedy text alone is too noisy on these prompts: cold vs
cold2 already diverges within ~10 tokens).

  SCRIPT=conv_ckpt_test.py IMAGE=qwen38-exl3-tabby:convckpt \
    docker/tabbyapi/tools/run_engine_bench.sh cc -e EXL3_HIST_STASH=1 -e CHUNK=8192 -e MBS=3

Env: SYS=5200 FUP=128 CONVS=2
"""
import os, time, random
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import job as jm
from exllamav3.generator.sampler import GreedySampler
from exllamav3.constants import PAGE_SIZE

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
MBS = int(os.environ.get("MBS", "3")); CHUNK = int(os.environ.get("CHUNK", "8192"))
SYS = int(os.environ.get("SYS", "5200")); FUP = int(os.environ.get("FUP", "128"))
CONVS = int(os.environ.get("CONVS", "2"))
print(f"CONFIG MBS={MBS} CHUNK={CHUNK} SYS={SYS} FUP={FUP} CONVS={CONVS}", flush=True)

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

def run_many(prompts, max_new):
    jobs = [Job(input_ids=p.unsqueeze(0), max_new_tokens=max_new, min_new_tokens=max_new,
                sampler=GreedySampler(), identifier=i, return_logits=True) for i, p in enumerate(prompts)]
    t = time.time(); first = {}; out = {i: [] for i in range(len(jobs))}; lg = {}
    for j in jobs:
        gen.enqueue(j)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming":
                i = r["identifier"]
                first.setdefault(i, time.time())
                if r.get("token_ids") is not None: out[i].append(r["token_ids"][0])
                if i not in lg and r.get("logits") is not None:
                    lg[i] = r["logits"].reshape(-1, r["logits"].shape[-1])[0].float().cpu()
    return [((first.get(i) or time.time()) - t, torch.cat(out[i]), j.cached_pages * PAGE_SIZE, lg.get(i))
            for i, j in enumerate(jobs)]
run = lambda p, n: run_many([p], n)[0]

def reset(level):
    jm._CC_LEVEL = level
    gen.recurrent_cache.clear(); gen.recurrent_cache.update_total_size()
    jm._cc_recent.clear(); jm._cc_meta.clear()

def logit_cmp(a, b):
    """First sampled position: max |logit diff|, KL(b || a) in nats, top-1 equal, top-10 overlap"""
    if a is None or b is None:
        return "no logits"
    m = torch.isfinite(a) & torch.isfinite(b)  # masked vocabulary entries are -inf
    if torch.equal(a, b):
        return "bit-identical"
    a, b = a[m], b[m]
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = float((lb.exp() * (lb - la)).sum())
    ta, tb = a.topk(10).indices.tolist(), b.topk(10).indices.tolist()
    return (f"max|d| {float((a - b).abs().max()):.4f} KL {kl:.2e} top1 {'=' if ta[0] == tb[0] else '!='} "
            f"top10 {len(set(ta) & set(tb))}/10")

def diverge(x, y):
    n = min(x.shape[0], y.shape[0]); ne = (x[:n] != y[:n]).nonzero()
    return "identical" if not ne.numel() else f"diverge at {int(ne[0, 0])}"

run(torch.cat([text(1024, 1), GEN]), 2)
for c in range(CONVS):
    s = 1000 * (c + 1)
    sys_ids = torch.cat([enc("<|im_start|>system\n"), text(SYS, s), enc("<|im_end|>\n")])
    task = text(40, s + 1)
    user = lambda body: torch.cat([enc("<|im_start|>user\n"), body])
    p1 = torch.cat([sys_ids, user(torch.cat([text(180, s + 2), task])), GEN])
    p2 = torch.cat([sys_ids, user(task),
                    enc("<|im_end|>\n<|im_start|>assistant\n<think>\nList the files.\n</think>\n\n<tool_call>\n"
                        "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
                        "<|im_end|>\n<|im_start|>user\n<tool_response>\n"), text(2400, s + 3),
                    enc("\n</tool_response>"), GEN])
    cut = int(SYS * 0.6)
    p3 = torch.cat([sys_ids[:cut], text(1600, s + 4), enc("<|im_end|>\n"), user(text(60, s + 5)), GEN])
    p3b = torch.cat([sys_ids[:cut], text(1600, s + 4)[:100], text(1500, s + 6), enc("<|im_end|>\n"),
                     user(text(60, s + 7)), GEN])
    sys2 = torch.cat([sys_ids[:int(SYS * 0.9)], text(SYS - int(SYS * 0.9), s + 12), enc("<|im_end|>\n")])
    p4a = torch.cat([sys2, user(torch.cat([text(180, s + 8), text(40, s + 9)])), GEN])
    p4b = torch.cat([sys2, user(torch.cat([text(180, s + 10), text(40, s + 11)])), GEN])
    names = ["p2", "p3", "p3b", "p4a", "p4b"]
    prompts = dict(zip(names, (p2, p3, p3b, p4a, p4b)))
    res = {}
    # on: the order a server would see them in, cache kept
    reset(2)
    r1 = run(p1, 1)
    res["on"] = {"p2": run(p2, FUP), "p3": run(p3, FUP), "p3b": run(p3b, FUP)}
    ra, rb = run_many([p4a, p4b], FUP)
    res["on"].update({"p4a": ra, "p4b": rb})
    rc = gen.recurrent_cache
    ok = all(k in rc for k in rc.conv_keys) and len(rc.conv_keys) <= jm._CC_MAX
    print(f"RESULT conv {c} on: p1 {p1.shape[0]} tokens ttft {r1[0]:.3f}s; anchors {jm._cc_stats}, "
          f"separate LRU {len(rc.conv_keys)} consistent {ok}", flush=True)
    # prefix: anchors off, but an earlier prompt ended exactly at the anchor, so the fork's own
    # last-page checkpoint is there (what prefix caching does today); "on" should equal it
    res["prefix"] = {}
    for n, at in (("p2", 5120), ("p3b", 3072)):
        reset(0)
        run(prompts[n][:at + 1], 1)
        have = sorted(v["position"] for v in gen.recurrent_cache.values())
        res["prefix"][n] = run(prompts[n], FUP)
        print(f"RESULT conv {c} {n:>3} prefix arm: checkpoints after the {at + 1}-token prompt at {have}", flush=True)
    # chunked: anchors off, a cold prefill whose first chunk ends at the anchor (the same forward
    # shapes as resuming there); first-token logits should equal "on" bit for bit
    res["chunked"] = {}
    for n, at in (("p2", 5120), ("p3b", 3072)):
        reset(0)
        gen.max_chunk_size = at
        try:
            res["chunked"][n] = run(prompts[n], FUP)
        finally:
            gen.max_chunk_size = CHUNK
    for arm in ("cold", "cold2"):
        res[arm] = {}
        for n in names:
            reset(0)
            res[arm][n] = run(prompts[n], FUP)
    for n in names:
        on, cold, cold2 = res["on"][n], res["cold"][n], res["cold2"][n]
        print(f"RESULT conv {c} {n:>3} ({prompts[n].shape[0]} tokens): on resumed {on[2]} ttft {on[0]:.3f}s, "
              f"cold ttft {cold[0]:.3f}s | greedy {FUP}: on vs cold {diverge(on[1], cold[1])}, "
              f"cold2 vs cold {diverge(cold2[1], cold[1])}", flush=True)
        print(f"RESULT conv {c} {n:>3} first-token logits: on vs cold {logit_cmp(on[3], cold[3])}; "
              f"cold2 vs cold {logit_cmp(cold2[3], cold[3])}", flush=True)
        if n in res["chunked"]:
            ch = res["chunked"][n]
            print(f"RESULT conv {c} {n:>3} vs cold prefill chunked at the anchor: logits "
                  f"{logit_cmp(on[3], ch[3])}, greedy {diverge(on[1], ch[1])}", flush=True)
        if n in res["prefix"]:
            pf = res["prefix"][n]
            print(f"RESULT conv {c} {n:>3} vs prefix caching today (resumed {pf[2]}): logits "
                  f"{logit_cmp(on[3], pf[3])}, greedy {diverge(on[1], pf[1])}", flush=True)
    # scale of a wrong state: p2's cold logits against p4a's (another prompt)
    print(f"RESULT conv {c} reference, two different prompts: {logit_cmp(res['cold']['p4a'][3], res['cold']['p2'][3])}", flush=True)
reset(2)
print("DONE", flush=True)
