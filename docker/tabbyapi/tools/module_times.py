#!/usr/bin/env python3
"""Where a long prefill chunk's time goes, by module, with a device sync after every
module forward (so each module's GPU work is attributed to it; the total is a little
above the unsynchronised time). One cold prompt of TOKENS through the Generator.

Runs inside the image (SCRIPT=module_times.py run_engine_bench.sh <name>), TabbyAPI stopped.
Env: TOKENS=8192  CHUNK=8192  DRAFT=1 (load the MTP draft model like TabbyAPI)
"""
import os, time, random, collections
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
TOKENS = int(os.environ.get("TOKENS", "8192")); CHUNK = int(os.environ.get("CHUNK", "8192"))
DRAFT = os.environ.get("DRAFT", "1") == "1"
config = Config.from_directory(MODEL); tok = Tokenizer.from_config(config)
model = Model.from_config(config)
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
cache = Cache(model, max_num_tokens=65536, max_batch_size=4, max_history=5, **qkw)
dm = dc = None
if DRAFT:
    dm = Model.from_config(config, component="mtp")
    dc = Cache(dm, max_num_tokens=65536, max_batch_size=4, max_history=5, **qkw)
    dm.load(progressbar=False)
model.load(progressbar=False, max_chunk_size=CHUNK, max_batch_size=4)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc,
                max_batch_size=4, max_chunk_size=CHUNK, num_draft_tokens=5 if DRAFT else None,
                dynamic_draft_tokens=DRAFT, draft_confidence=0.6)
SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read(); src_ids = tok.encode(SRC, add_bos=False)[0]
def build(n, seed):
    random.seed(seed); parts, k = [f"Session {seed} salt {random.random()}\n"], 0
    while k < n:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return tok.encode("<|im_start|>user\n" + "".join(parts), add_bos=False)[0][:n]
def run(ids):
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=1, sampler=GreedySampler())
    torch.cuda.synchronize(); t = time.perf_counter(); gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
    torch.cuda.synchronize(); return time.perf_counter() - t
run(build(1024, 1)); run(build(TOKENS, 2))
print(f"RESULT unsynced cold {TOKENS}: {run(build(TOKENS, 3)):.2f}s", flush=True)

# Every module, exclusive time: its own synced wall time minus that of timed children.
# BlockSparseMLP counts as one (its experts are not called through forward)
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
def walk(m, out):
    for s in getattr(m, "modules", []):
        out.append(s)
        if not isinstance(s, BlockSparseMLP): walk(s, out)
    return out
dm_mods = set(id(x) for x in walk(dm, [])) if dm else set()
allm = walk(model, []) + (walk(dm, []) if dm else [])
acc = collections.Counter(); cnt = collections.Counter(); stack = []
patched = []
for m in allm:
    name = type(m).__name__ + ("@" + m.key.split(".")[-1] if getattr(m, "key", None) else "")
    if id(m) in dm_mods: name = "MTP:" + name
    f = m.forward
    def g(*a, _f=f, _n=name, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); stack.append(0.0)
        r = _f(*a, **k)
        torch.cuda.synchronize(); dt = time.perf_counter() - t; child = stack.pop()
        acc[_n] += dt - child; cnt[_n] += 1
        if stack: stack[-1] += dt
        return r
    m.forward = g; patched.append((m, f))
total = run(build(TOKENS, 4))
for m, f in patched: m.forward = f
print(f"RESULT synced cold {TOKENS}: {total:.2f}s, modules {sum(acc.values()):.2f}s, other {total - sum(acc.values()):.2f}s")
for k, v in acc.most_common(25):
    print(f"RESULT   {k:<40} {v:6.3f}s  calls={cnt[k]}", flush=True)

# KERNELS=N: one more cold prefill (unsynced) under torch.profiler; every GPU kernel is
# attributed to its innermost module (class name, TransformerBlock = the block's own code),
# then the top N kernels of the top scopes
KERNELS = int(os.environ.get("KERNELS", "0"))
if KERNELS:
    from torch.profiler import profile, ProfilerActivity, record_function
    from torch.autograd import DeviceType
    for m in allm:
        name = ("MTP:" if id(m) in dm_mods else "") + type(m).__name__
        f = m.forward
        def g2(*a, _f=f, _n=name, **k):
            with record_function("MOD::" + _n):
                return _f(*a, **k)
        m.forward = g2
    ids = build(TOKENS, 5); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter(); run(ids); wall = time.perf_counter() - t
    for m, f in patched: m.forward = f
    def scope(e):
        p = e.cpu_parent
        while p is not None:
            if p.name.startswith("MOD::"): return p.name[5:]
            p = p.cpu_parent
        return "(untagged)"
    def short(n):
        n = n.replace("void ", "").replace("at::native::", "").replace("(anonymous namespace)::", "")
        return n.split("(")[0][:90]
    sgpu = collections.Counter(); kern = collections.defaultdict(collections.Counter)
    kcnt = collections.defaultdict(collections.Counter)
    # GPU-side spans of the MOD:: ranges nest like the modules; each kernel goes to the
    # innermost span that contains its start (sweep with a stack)
    spans, kernels = [], []
    for e in prof.events():
        if e.device_type != DeviceType.CUDA: continue
        tr = e.time_range
        (spans if e.name.startswith("MOD::") else kernels).append((tr.start, tr.end, e.name))
    spans.sort(key=lambda x: (x[0], -x[1])); kernels.sort()
    stack, i = [], 0
    for ks, ke, kn in kernels:
        while i < len(spans) and spans[i][0] <= ks:
            while stack and stack[-1][1] <= spans[i][0]: stack.pop()
            stack.append(spans[i]); i += 1
        while stack and stack[-1][1] <= ks: stack.pop()
        s_ = stack[-1][2][5:] if stack else "(untagged)"
        sgpu[s_] += ke - ks; kern[s_][short(kn)] += ke - ks; kcnt[s_][short(kn)] += 1
    print(f"KERNEL profiled cold {TOKENS}: wall {wall:.2f}s, kernels {sum(sgpu.values())/1e6:.2f}s")
    for s, v in sgpu.most_common(12):
        print(f"KERNEL {s:<32} {v/1e6:6.3f}s")
        for k, d in kern[s].most_common(KERNELS):
            print(f"KERNEL     {d/1e6:6.3f}s n={kcnt[s][k]:<6} {k}")
    print("DONE", flush=True)
else:
    print("DONE", flush=True)
