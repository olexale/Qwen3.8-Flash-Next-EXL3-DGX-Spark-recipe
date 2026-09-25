#!/usr/bin/env python3
"""A/B launch variants of the QSA sparse attention kernel on inputs captured by qsa_capture.py.
No model load, so it can run while TabbyAPI serves (timings are then only indicative).

Each variant runs _qsa_sparse_split_kernel + the combine kernel exactly like
qsa_sparse_attend_rows, with its own (num_warps, num_stages); a variant with DEV=1 takes the
kernels from /tmp/qsa_dev.py (mounted dev copy of qsa_triton.py) instead. Output is compared
bit for bit with the shipped path.

  docker run --rm --gpus all -v ~/scratch/out:/out -v $PWD/docker/tabbyapi/tools/qsa_ab.py:/tmp/ab.py:ro \
    --entrypoint python3 qwen38-exl3-tabby:latest /tmp/ab.py
Env: VARIANTS="4:2,1:2,2:2"  (warps:stages[:dev])  REPS=5
"""
import os, glob, time, importlib.util
import torch, triton
import exllamav3.modules.attention_fn.qsa_triton as QT
from exllamav3.modules.attention_fn.triton_paged import _paged_attn_decode_combine_kernel
from exllamav3.modules.attention_fn.triton_paged import _get_h32

DEV = None
if os.path.exists("/tmp/qsa_dev.py"):
    spec = importlib.util.spec_from_file_location("exllamav3.modules.attention_fn.qsa_dev", "/tmp/qsa_dev.py")
    DEV = importlib.util.module_from_spec(spec); spec.loader.exec_module(DEV)

def launch(c, warps, stages, dev=False, bn=32, stg=False):
    q, k, v, idx, bt = c["q"], c["k"], c["v"], c["indices"], c["block_table"]
    ks, vs, kb, vb = c["qc"]; kvh = c["n_kv_heads"]
    R, H, hd = q.shape; group = H // kvh
    BLOCK_H, BLOCK_N = 16, bn
    programs = R * kvh * triton.cdiv(group, BLOCK_H); K_pad = idx.shape[1]
    if stg:
        # staged: dequantize the sequence's K/V once (dev qsa_stage_kv), then the fp16 gather
        n_tok = c["n_tok"]; kvh_ = c["n_kv_heads"]; ps = c["page_size"]
        k16 = DEV.qsa_stage_kv(k, ks, kb, bt[0].contiguous(), n_tok, kvh_, hd, ps)
        v16 = DEV.qsa_stage_kv(v, vs, vb, bt[0].contiguous(), n_tok, kvh_, hd, ps)
        return DEV.qsa_sparse_attend_rows(q, k, v, idx, c["sm_scale"], bt, ps, c["qc"], kvh_,
                                          staged=(k16, v16))
    if dev and hasattr(DEV, "qsa_sparse_attend_rows_cfg"):
        return DEV.qsa_sparse_attend_rows_cfg(q, k, v, idx, c["sm_scale"], bt, c["page_size"],
                                              c["qc"], kvh, num_warps=warps, num_stages=stages)
    kern = (DEV if dev else QT)._qsa_sparse_split_kernel
    splits = max(1, min(2 * torch.cuda.get_device_properties(0).multi_processor_count // programs,
                        -(-K_pad // (4 * BLOCK_N)), 128))
    split_len = -(-(-(-K_pad // splits)) // BLOCK_N) * BLOCK_N
    po = torch.empty((programs * splits * BLOCK_H * hd,), dtype=torch.float, device=q.device)
    pml = torch.empty((programs * splits * BLOCK_H * 2,), dtype=torch.float, device=q.device)
    o = torch.empty((R, H, hd), dtype=torch.half, device=q.device)
    h32 = _get_h32(q.device)
    kern[(programs, splits)](
        q, k, v, bt, idx, po, pml, K_pad, bt.shape[1], splits, split_len, ks, vs, h32,
        n_q_heads=H, n_kv_heads=kvh, page_size=c["page_size"], head_dim=hd, K_pad=K_pad,
        scale=float(c["sm_scale"]), BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, PAGED=1, QCK=kb, QCV=vb,
        num_warps=warps, num_stages=stages)
    _paged_attn_decode_combine_kernel[(programs,)](
        po, pml, o, h32, splits, pml, QCV=vb, HAS_SINKS=False, q_len=1, n_q_heads=H,
        n_kv_heads=kvh, head_dim=hd, HD_PAD=hd, BLOCK_M=1, BLOCK_H=BLOCK_H, BLOCK_ROWS=BLOCK_H,
        num_warps=4, num_stages=1)
    return o

variants = []
for s in os.environ.get("VARIANTS", "4:2,1:2,2:2,1:3,2:3").split(","):
    p = s.split(":")   # warps:stages[:dev][:bnN]
    variants.append((int(p[0]), int(p[1]), "dev" in p[2:],
                     next((int(x[2:]) for x in p[2:] if x.startswith("bn")), 32), "stg" in p[2:]))
REPS = int(os.environ.get("REPS", "5"))
tot = {v: 0.0 for v in variants}; worst = {v: 0.0 for v in variants}; same = {v: [0, 0] for v in variants}
for f in sorted(glob.glob("/out/qsa_caps/*.pt")):
    c = torch.load(f)
    for key in ("q", "k", "v", "indices", "block_table"): c[key] = c[key].cuda()
    c["qc"] = (c["qc"][0].cuda(), c["qc"][1].cuda(), c["qc"][2], c["qc"][3])
    c["n_tok"] = int(c["indices"].max().item()) + 1
    ref = QT.qsa_sparse_attend_rows(c["q"], c["k"], c["v"], c["indices"], c["sm_scale"], c["block_table"],
                                    c["page_size"], c["qc"], c["n_kv_heads"])
    assert torch.equal(ref.cpu(), c["out"]), f"{f}: replay differs from the captured output"
    line = [f"{os.path.basename(f)} rows {c['q'].shape[0]:>5}:"]
    for v in variants:
        o = launch(c, *v)
        d = int((o != ref).sum().item()); same[v][0] += d; same[v][1] += ref.numel()
        worst[v] = max(worst[v], ((o.float() - ref.float()).abs().max() / ref.float().abs().max()).item())
        for _ in range(2): launch(c, *v)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(REPS): launch(c, *v)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / REPS; tot[v] += dt
        line.append(f"{v[0]}w{v[1]}s{'D' if v[2] else ''}{'S' if v[4] else ''}/{v[3]} {dt*1e3:7.2f} ms diff {d}")
    print("RESULT " + "  ".join(line), flush=True)
    del c, ref; torch.cuda.empty_cache()
base = tot[variants[0]]
for v in variants:
    mx = worst[v]
    print(f"RESULT total {v[0]} warps {v[1]} stages BLOCK_N {v[3]}{' dev' if v[2] else ''}{' staged' if v[4] else ''}: {tot[v]*1e3:8.2f} ms "
          f"(x{base/tot[v]:.2f})  differing elements {same[v][0]}/{same[v][1]}, max rel diff {mx:.2e}", flush=True)
print("DONE", flush=True)
