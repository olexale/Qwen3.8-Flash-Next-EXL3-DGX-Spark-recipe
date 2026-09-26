#!/usr/bin/env python3
"""Plan C, C3 parity: a recurrent checkpoint at the prompt's last page boundary taken from inside
ONE prefill forward vs the one the fork stashes after splitting off the partial last page (two
forwards). Engine, TabbyAPI stopped; needs patch_exllamav3_fla_capture.py applied (run it in
the container first, as root, see below).

Per layer the checkpoint is assembled from the single forward:
  GatedDeltaNet  recurrent state: the fp32 state FLA's chunk kernel captures at the boundary
                 conv window: the last conv_kernel_size pre-conv inputs before the boundary (bf16)
  PLE            conv window: the columns of the forward's conv stream ending at the boundary
                 token context: the ids before the boundary
and compared with the split's stash, bit for bit (max |diff| where not). Cases: a cold prompt
(one forward from 0) and a follow-up resumed after a previous answer (one forward from the
resume point). Then the first-token logits of both arms and the greedy continuation.

  docker run --rm --gpus all -u 0 ... --entrypoint bash IMAGE -c \
    "python3 /t/patch_exllamav3_fla_capture.py && python3 /t/tools/c3_parity.py"
(run_engine_bench.sh's mounts; see the command in PLAN_C_SESSION_START.md)

Env: CTX=6000 OUT=300 NEW=350 FUP=64
"""
import os, time, random, inspect, textwrap
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator import job as jobmod
from exllamav3.generator.sampler import GreedySampler
from exllamav3.constants import PAGE_SIZE
import exllamav3.vendor.fla as fla
import exllamav3.modules.gated_delta_net as gdnmod
import exllamav3.modules.gated_delta_net_fn.conv1d as convmod
import exllamav3.modules.ple as plemod

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
MBS = int(os.environ.get("MBS", "3")); CHUNK = int(os.environ.get("CHUNK", "8192"))
CTX = int(os.environ.get("CTX", "6000")); OUT = int(os.environ.get("OUT", "300"))
NEW = int(os.environ.get("NEW", "350")); FUP = int(os.environ.get("FUP", "64"))
print(f"CONFIG MBS={MBS} CHUNK={CHUNK} CTX={CTX} OUT={OUT} NEW={NEW} FUP={FUP}", flush=True)

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config); dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=65536, max_batch_size=MBS, max_history=5, **qkw)
dcache = Cache(dm, max_num_tokens=65536, max_batch_size=MBS, max_history=5, **qkw)
dm.load(progressbar=False); model.load(progressbar=False, max_chunk_size=CHUNK, max_batch_size=MBS)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                max_batch_size=MBS, max_chunk_size=CHUNK, recurrent_cache_size=8192 * 1024**2,
                num_draft_tokens=5, dynamic_draft_tokens=True, draft_confidence=0.6)
print("loaded", flush=True)

# ---- capture hooks (active only while CAP["rb"] is set: the boundary relative to the forward's start) ----
CAP = {"rb": None, "cur": None, "by_module": {}}

orig_gdn_forward = gdnmod.GatedDeltaNet.forward
def gdn_forward(self, x, params, out_dtype=None):
    CAP["cur"] = self
    try:
        return orig_gdn_forward(self, x, params, out_dtype)
    finally:
        CAP["cur"] = None
gdnmod.GatedDeltaNet.forward = gdn_forward

orig_rule = fla.chunk_gated_delta_rule
def rule(q, k, v, *a, **kw):
    rb = CAP["rb"]
    if rb is not None and CAP["cur"] is not None and "capture_state" not in kw and \
            "rec" not in CAP["by_module"].get(id(CAP["cur"]), {}) and q.shape[1] > rb:
        buf = torch.full((q.shape[0], v.shape[2], k.shape[3], v.shape[3]), float("nan"),
                         device=q.device, dtype=torch.float)
        out = orig_rule(q, k, v, *a, capture_state=buf, capture_chunk=rb // 64, **kw)
        CAP["by_module"].setdefault(id(CAP["cur"]), {})["rec"] = buf
        return out
    return orig_rule(q, k, v, *a, **kw)
fla.chunk_gated_delta_rule = rule

def conv_hook(orig, layout):
    def f(mixed_qkv, *a, **kw):
        rb = CAP["rb"]
        if rb is not None and CAP["cur"] is not None and \
                "conv" not in CAP["by_module"].get(id(CAP["cur"]), {}):
            n = CAP["cur"].conv_kernel_size
            x = mixed_qkv[0] if not isinstance(mixed_qkv, tuple) else None
            if x is not None:
                cols = x[rb - n:rb, :].t() if layout == "bsd" else x[:, rb - n:rb]
                CAP["by_module"].setdefault(id(CAP["cur"]), {})["conv"] = cols.to(torch.bfloat16).clone()
        return orig(mixed_qkv, *a, **kw)
    return f
convmod.causal_conv1d_update_split = conv_hook(convmod.causal_conv1d_update_split, "bsd")
gdnmod.causal_conv1d_update = conv_hook(gdnmod.causal_conv1d_update, "bds")

orig_ple_fs = plemod.PLELayer.forward_streams
def ple_fs(self, streams, token_history, params, conv_state=None):
    delta, conv_stream = orig_ple_fs(self, streams, token_history, params, conv_state)
    rb = CAP["rb"]
    # the first forward after the boundary was set is the prefill; later (decode) forwards are ignored
    if rb is not None and conv_stream is not None and id(self) not in CAP["by_module"]:
        win, ctx = self.conv_state_len, self.ple_embedding.context_len
        CAP["by_module"][id(self)] = {"conv": conv_stream[0, :, rb:rb + win].clone(),
                                      "ids": token_history[0, rb:rb + ctx].clone()}
    return delta, conv_stream
plemod.PLELayer.forward_streams = ple_fs

# NOSPLIT prefill that records where the last page boundary falls in the single forward
src = textwrap.dedent(inspect.getsource(jobmod.Job.prefill))
anchor = "if prefill_start < last_page_b <= prefill_end:"
assert src.count(anchor) == 1
def _cap_set(a, b, e):
    CAP["rb"] = (b - a) if a < b <= e else None
    CAP["b"] = b
    return False
ns = dict(vars(jobmod)); ns["_cap_set"] = _cap_set
exec(src.replace(anchor, "if _cap_set(prefill_start, last_page_b, prefill_end):"), ns)
prefill_split, prefill_one = jobmod.Job.prefill, ns["prefill"]

SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()
src_ids = tok.encode(SRC, add_bos=False)[0]
def text(n, seed):
    rnd = random.Random(seed); parts, k = [f"salt {rnd.random()}\n"], 0
    while k < n:
        a = rnd.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("".join(parts), add_bos=False)[0][:n]
enc = lambda s: tok.encode(s, add_bos=False, encode_special_tokens=True)[0]

def run(ids, max_new):
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=max_new, min_new_tokens=max_new,
              sampler=GreedySampler(), return_logits=True)
    t = time.time(); gen.enqueue(job); first = None; out = []; lg = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming":
                if first is None: first = time.time()
                if r.get("token_ids") is not None: out.append(r["token_ids"][0])
                if lg is None and r.get("logits") is not None:
                    lg = r["logits"].reshape(-1, r["logits"].shape[-1])[0].float().cpu()
    return (first or time.time()) - t, torch.cat(out), job.cached_pages * PAGE_SIZE, lg

def stash_at(pos):
    for key, v in gen.recurrent_cache.items():
        if v["position"] == pos:
            return key, v
    return None, None

def drop_after(pos):
    rc = gen.recurrent_cache
    for key in [k for k, v in rc.items() if v["position"] > pos]:
        rc.pop(key)
    rc.update_total_size()

def assemble(ref):
    """The one-forward checkpoint in the stash's own format, for the layers in ref"""
    out = {}
    layers = cache.get_all_recurrent_layers()
    for key, l in layers.items():
        got = CAP["by_module"].get(id(l.module))
        if got is None:
            out[key] = None
        elif isinstance(l, gdnmod.GDNLayerState):
            out[key] = (got["rec"].unsqueeze(1), got["conv"])
        elif isinstance(l, plemod.PLELayerState):
            out[key] = (got["conv"].to(l.conv_state.dtype), got["ids"].to("cpu", torch.long))
        else:
            out[key] = None
    return out

def compare(ref, one, label):
    stats = {}
    for key, r in ref.items():
        if not isinstance(r, tuple):
            continue
        o = one.get(key)
        name = type(cache.get_all_recurrent_layers()[key]).__name__ if key in cache.get_all_recurrent_layers() else "?"
        d = stats.setdefault(name, [0, 0, 0.0, {}])
        d[0] += 1
        if o is None:
            d[1] += 1; d[3]["missing"] = d[3].get("missing", 0) + 1
            continue
        for part, (a, b) in enumerate(zip(r, o)):
            a, b = a.cpu(), b.cpu().reshape(a.shape).to(a.dtype)
            if not torch.equal(a, b):
                d[1] += 1
                md = (a.double() - b.double()).abs().max().item()
                d[2] = max(d[2], md)
                d[3][part] = d[3].get(part, 0) + 1
    for name, (n, bad, md, parts) in stats.items():
        print(f"RESULT {label} {name}: {n} layers, {n - bad if not parts else n - max(parts.values())} "
              f"fully bit-identical; mismatching parts {parts or 'none'}; max |diff| {md:.3e}", flush=True)

def diverge(x, y):
    n = min(x.shape[0], y.shape[0]); ne = (x[:n] != y[:n]).nonzero()
    return "identical" if not ne.numel() else f"diverge at {int(ne[0, 0])}"

def logit_cmp(a, b):
    if torch.equal(a, b):
        return "bit-identical"
    m = torch.isfinite(a) & torch.isfinite(b)
    la, lb = torch.log_softmax(a[m], -1), torch.log_softmax(b[m], -1)
    return f"max|d| {float((a - b).abs().max()):.4f} KL {float((lb.exp() * (lb - la)).sum()):.2e}"

run(torch.cat([text(1024, 1), enc("<|im_end|>\n<|im_start|>assistant\n<think>\n")]), 2)
ASK = enc("\nContinue the text.<|im_end|>\n<|im_start|>assistant\n<think>\n")
for case in range(2):
    base = torch.cat([enc("<|im_start|>user\n"), text(CTX + 57 * case, 300 + case), ASK])
    gen.recurrent_cache.clear(); gen.recurrent_cache.update_total_size()
    if case == 0:
        prompt, start_note = base, "cold"
    else:
        _, ans, _, _ = run(base, OUT)
        prompt = torch.cat([base, ans, enc("<|im_end|>\n<|im_start|>user\n<tool_response>\n"), text(NEW, 900),
                            enc("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")])
        start_note = "follow-up"
    keep = {k for k in gen.recurrent_cache}
    b = (prompt.shape[0] - 1) // PAGE_SIZE * PAGE_SIZE
    # split (the fork today)
    jobmod.Job.prefill = prefill_split
    t_s, out_s, res_s, lg_s = run(prompt, FUP)
    key, ref = stash_at(b)
    assert ref is not None, f"no split checkpoint at {b}"
    ref = {k: v for k, v in ref.items()}
    # one forward, capture at b
    rc = gen.recurrent_cache
    for k in [k for k in rc if k not in keep]:
        rc.pop(k)
    rc.update_total_size()
    CAP["by_module"].clear()
    jobmod.Job.prefill = prefill_one
    try:
        t_o, out_o, res_o, lg_o = run(prompt, FUP)
    finally:
        jobmod.Job.prefill = prefill_split
        rb = CAP["rb"]; CAP["rb"] = None
    print(f"RESULT {start_note}: prompt {prompt.shape[0]}, boundary {b}; split resumed {res_s} ttft {t_s:.3f}s, "
          f"one forward resumed {res_o} ttft {t_o:.3f}s (boundary at +{rb} in the forward)", flush=True)
    compare(ref, assemble(ref), start_note)
    print(f"RESULT {start_note}: first-token logits one vs split {logit_cmp(lg_o, lg_s)}; greedy {FUP}: "
          f"{diverge(out_o, out_s)}", flush=True)
print("DONE", flush=True)
