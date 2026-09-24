#!/usr/bin/env python3
"""Which MoE path runs during prefill, where the time goes, and what one MoE layer costs.

Runs inside the qwen38-exl3-tabby image like engine_bench.py (SCRIPT=moe_trace.py run_engine_bench.sh).
TabbyAPI must be stopped. Four parts:

  FLAGS   per MoE layer: is_quantized, uniform_expert_q, support_quant_paths,
          support_fused, fused buffers, fused_rows, bc
  TIER    one cold prefill of TRACE_TOKENS (default 600): how many experts each tier
          handles (fused exl3_moe launches, batched reconstruct, per-expert graph,
          per-expert dequant, bszN)
  SCOPE   a torch profile of the same prefill with ops tagged by module class:
          wall/GPU time and the index_add_ / index_select / gemv counts per class
  LAYER   one MoE layer alone on real hidden states at 16, 64, 600, 2048 rows:
          time with the fused tier on and off (when it can be on), parity between
          the two, and the floor for reading the experts that routing touched

Env: TRACE_TOKENS=600, CHUNK=8192, LAYER=24 (index among the MoE layers),
     BW_GBS=230 (memory bandwidth used for the floor), EXL3_* as usual.
"""
import os, time, random, collections
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.block_sparse_mlp as B

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
CHUNK = int(os.environ.get("CHUNK", "8192"))
NTOK = int(os.environ.get("TRACE_TOKENS", "600"))
LAYER = int(os.environ.get("LAYER", "24"))
BW = float(os.environ.get("BW_GBS", "230")) * 1e9
print("CONFIG", f"CHUNK={CHUNK} TRACE_TOKENS={NTOK} EXL3_MOE_FUSED_UNIFORM={os.environ.get('EXL3_MOE_FUSED_UNIFORM', '(unset)')}", flush=True)

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768, max_batch_size=1, layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
model.load(progressbar=False, max_chunk_size=CHUNK, max_batch_size=1)
gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=1, max_chunk_size=CHUNK)

def walk(m, out):
    for s in getattr(m, "modules", []):
        out.append(s); walk(s, out)
    return out
mods = walk(model, [])
moes = [m for m in mods if isinstance(m, B.BlockSparseMLP)]

# ---- FLAGS
rows = collections.Counter()
for m in moes:
    rows[(m.is_quantized, m.uniform_expert_q, m.support_quant_paths, m.support_fused,
          m.fused_mode_buffers is not None, m.fused_rows, m.bc is not None,
          getattr(m, "mixedk_unified", False), m.f_threshold)] += 1
print("FLAGS (is_quantized, uniform_expert_q, support_quant_paths, support_fused, fused_buffers,"
      " fused_rows, bc, mixedk_unified, f_threshold) -> layers")
for k, v in rows.items():
    print(f"FLAGS {k} -> {v}")
m0 = moes[0]
print(f"FLAGS codebooks gate/up/down: {m0.multi_gate.q_cb() if m0.multi_gate else None} "
      f"{m0.multi_up.q_cb() if m0.multi_up else None} {m0.multi_down.q_cb() if m0.multi_down else None}"
      f" activation={m0.activation_fn} experts={m0.num_experts} top_k={m0.num_experts_per_tok}", flush=True)

# ---- instrumentation
tier = collections.Counter()
class BCProxy:
    def __init__(self, bc): self._bc = bc
    def run_single_expert(self, x, e):
        tier["per-expert graph (experts)"] += 1; tier["per-expert graph (rows)"] += x.shape[0]
        return self._bc.run_single_expert(x, e)
    def run_single_expert_dq(self, x, e, *a):
        tier["per-expert dq (experts)"] += 1; tier["per-expert dq (rows)"] += x.shape[0]
        return self._bc.run_single_expert_dq(x, e, *a)
    def run_bszN(self, y, *a):
        tier["bszN (calls)"] += 1
        return self._bc.run_bszN(y, *a)
    def __getattr__(self, k): return getattr(self._bc, k)
_ext = B.ext
class ExtProxy:
    def exl3_moe(self, *a):
        tier["fused exl3_moe (launches)"] += 1
        n = a[29]  # num_active
        tier["fused exl3_moe (experts, -1 = all)"] += n if n >= 0 else 0
        if n < 0: tier["fused exl3_moe all-fused calls"] += 1
        return _ext.exl3_moe(*a)
    def __getattr__(self, k): return getattr(_ext, k)
_rbr = B.BlockSparseMLP._run_batch_recon
def rbr(self, recon, y, fhs, ts, ws, ecl, groups, *a, **k):
    tier["batched recon (groups)"] += len(groups); tier["batched recon (experts)"] += sum(map(len, groups))
    return _rbr(self, recon, y, fhs, ts, ws, ecl, groups, *a, **k)
def instrument(on):
    B.ext = ExtProxy() if on else _ext
    B.BlockSparseMLP._run_batch_recon = rbr if on else _rbr
    for m in moes:
        if on and not isinstance(m.bc, BCProxy): m.bc = BCProxy(m.bc)
        if not on and isinstance(m.bc, BCProxy): m.bc = m.bc._bc

SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()
src_ids = tok.encode(SRC, add_bos=False)[0]
END = tok.encode("\nSummarize the above in one line.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_bos=False)[0]
def build(n, seed):
    random.seed(seed); parts, k = [f"Session {seed} salt {random.random()}\n"], 0
    while k < n:
        a = random.randrange(0, len(src_ids) - 512)
        parts.append(tok.decode(src_ids[a:a + 512].unsqueeze(0))[0]); k += 512
    return torch.cat([tok.encode("<|im_start|>user\n" + "".join(parts), add_bos=False)[0][:n], END])
def prefill(ids):
    job = Job(input_ids=ids.unsqueeze(0), max_new_tokens=1, sampler=GreedySampler())
    t = time.time(); gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
    torch.cuda.synchronize()
    return time.time() - t

prefill(build(1024, 1)); prefill(build(1024, 2))  # warm-up (kernel setup)
print(f"TIME cold {NTOK} (uninstrumented): {prefill(build(NTOK, 11)):.2f}s", flush=True)
instrument(True)
dt = prefill(build(NTOK, 12))
instrument(False)
print(f"TIME cold {NTOK} again (uninstrumented): {prefill(build(NTOK, 14)):.2f}s", flush=True)
print(f"TIER one cold prefill of {NTOK}+{END.shape[0]} tokens over {len(moes)} MoE layers ({dt:.2f}s instrumented):")
for k in sorted(tier): print(f"TIER   {k:<40} {tier[k]}")

# ---- SCOPE
from torch.profiler import profile, ProfilerActivity, record_function
wrapped = {}
for m in mods:
    c = type(m)
    if c in wrapped or not hasattr(c, "forward"): continue
    if c.__name__ not in ("BlockSparseMLP", "GatedDeltaNet", "Attention", "PLE", "NgramEmbedding",
                          "Embedding", "RMSNorm", "GatedMLP", "Linear", "MLP") and "Ngram" not in c.__name__ \
            and "PLE" not in c.__name__ and "Attention" not in c.__name__ and "DeltaNet" not in c.__name__:
        continue
    f = c.forward
    def mk(f, name):
        def g(self, *a, **k):
            with record_function("MOD::" + name):
                return f(self, *a, **k)
        return g
    wrapped[c] = f; c.forward = mk(f, c.__name__)
print("SCOPE tagged classes:", sorted(c.__name__ for c in wrapped), flush=True)
ids = build(NTOK, 13)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    prefill(ids)
for c, f in wrapped.items(): c.forward = f
evs = prof.events()
def scope(e):
    # innermost-but-one tagged ancestor would double count nested Linear inside MoE; take outermost
    s = None; p = e.cpu_parent
    while p is not None:
        if p.name.startswith("MOD::"): s = p.name[5:]
        p = p.cpu_parent
    return s or "(untagged)"
wall = collections.Counter(); gpu = collections.Counter(); ncalls = collections.Counter()
for e in evs:
    if e.name.startswith("MOD::"):
        # top-level only
        p = e.cpu_parent; top = True
        while p is not None:
            if p.name.startswith("MOD::"): top = False; break
            p = p.cpu_parent
        if top:
            wall[e.name[5:]] += e.cpu_time_total; gpu[e.name[5:]] += e.device_time_total; ncalls[e.name[5:]] += 1
ops = collections.defaultdict(collections.Counter)
for e in evs:
    for key in ("aten::index_add_", "aten::index_select", "aten::mul_", "cudaGraphLaunch", "cudaLaunchKernel"):
        if e.name == key: ops[scope(e)][key] += 1
    if e.device_type is not None and "gemv" in e.name.lower():
        ops[scope(e)]["gemv kernels"] += 1
print("SCOPE outermost-module scopes: calls, CPU wall (s), GPU (s) attributed")
for k in sorted(wall, key=lambda k: -wall[k]):
    print(f"SCOPE   {k:<22} calls={ncalls[k]:>5} wall={wall[k]/1e6:6.2f}s gpu={gpu[k]/1e6:6.2f}s")
print("SCOPE op counts by outermost module scope")
for k in sorted(ops, key=lambda k: -sum(ops[k].values())):
    print(f"SCOPE   {k:<22} " + " ".join(f"{n}={v}" for n, v in sorted(ops[k].items())))
ka = prof.key_averages()
tot = sum(e.self_device_time_total for e in ka)
print(f"SCOPE total self GPU {tot/1e6:.2f}s", flush=True)

# ---- LAYER
lay = moes[LAYER]
cap = {}
_f = B.BlockSparseMLP.forward
def capf(self, x, params, out_dtype=None):
    if self is lay and "x" not in cap: cap["x"] = x.clone(); cap["params"] = params
    return _f(self, x, params, out_dtype)
B.BlockSparseMLP.forward = capf
prefill(build(2048, 21))
B.BlockSparseMLP.forward = _f
x = cap["x"]; params = cap["params"]
xs = x.view(-1, x.shape[-1])
print(f"LAYER {lay.key}: captured {xs.shape[0]} rows", flush=True)
ebytes = sum(l.inner.trellis.numel() * l.inner.trellis.element_size()
             for l in (lay.gates[:1] + lay.ups[:1] + lay.downs[:1]))
fb = lay.fused_mode_buffers
@torch.inference_mode()
def run(n, fused):
    lay.fused_mode_buffers = fb if fused else None
    return lay.forward(xs[:n].view(1, n, -1).clone(), params)
def timeit(n, fused, it=10):
    for _ in range(2): run(n, fused)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): run(n, fused)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it
@torch.inference_mode()
def route(z):
    if lay.routing_gate is None: return None
    if lay.router_pre_norm: z = lay.router_pre_norm.forward(z, params, out_dtype=torch.half)
    return lay.routing_fn(z.shape[0], lay.routing_cfg, z, params)[0]
for n in (16, 64, 600, 2048):
    if n > xs.shape[0]: continue
    # experts touched by these rows
    z = xs[:n]
    sel = route(z)
    touched = int(torch.unique(sel).numel()) if sel is not None else lay.num_experts
    floor = touched * ebytes / BW
    t_off = timeit(n, False)
    line = f"LAYER rows={n:>5} experts touched={touched:>3} floor={floor*1e3:6.2f}ms  per-expert tiers={t_off*1e3:7.2f}ms"
    if fb is not None:
        t_on = timeit(n, True)
        a = run(n, False).float(); b = run(n, True).float()
        rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-9)
        line += f"  fused tier={t_on*1e3:7.2f}ms  x{t_off/t_on:4.1f}  parity max rel {rel:.2e} mean|d| {(a-b).abs().mean().item():.2e}"
    print(line, flush=True)
lay.fused_mode_buffers = fb
print(f"LAYER expert bytes (gate+up+down) {ebytes/1e6:.2f} MB, bandwidth assumed {BW/1e9:.0f} GB/s")
print("DONE", flush=True)
