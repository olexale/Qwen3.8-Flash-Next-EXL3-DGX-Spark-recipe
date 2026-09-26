#!/usr/bin/env python3
"""Kernel-level parity for patch_exllamav3_fla_capture.py (Plan C, C3 step 1). No model.

The model's GatedDeltaNet shapes (16 key heads, 48 value heads, head dim 128), random bf16
inputs, an fp32 initial state (a resumed prefill). For several chunk lengths T and capture
points b (page boundaries, as the fork's last-page split uses):

  one      one forward over T tokens with the fp32 state captured at b
  split    a forward over the first b tokens (its final state: what the fork stashes today)
  plain    one forward over T tokens without capture (outputs must be unchanged by it)

Checks capture == split state bit for bit, and plain vs one outputs/final state bit for bit.
Also prints how far the bf16 per-chunk state h (what the plan first suggested using) is from it.

  docker run --rm --gpus all -v $PWD/docker/tabbyapi:/t:ro --entrypoint bash qwen38-exl3-tabby:latest \
    -c "python3 /t/patch_exllamav3_fla_capture.py && python3 /t/tools/fla_capture_parity.py"
"""
import torch
from exllamav3.vendor.fla import chunk_gated_delta_rule

torch.manual_seed(0)
dev = "cuda"
H, HV, K, V = 16, 48, 128, 128
fails = 0
for T, b in ((600, 512), (1000, 768), (2400, 2304), (8192, 7936), (300, 256), (257, 256)):
    q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, HV, device=dev, dtype=torch.float) * 0.1
    beta = torch.rand(1, T, HV, device=dev, dtype=torch.bfloat16)
    s0 = torch.randn(1, HV, K, V, device=dev, dtype=torch.float) * 0.1
    kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
    cap = torch.full((1, HV, K, V), float("nan"), device=dev, dtype=torch.float)
    o1, f1 = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=s0.clone(),
                                    capture_state=cap, capture_chunk=b // 64, **kw)
    o0, f0 = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=s0.clone(), **kw)
    os_, fs = chunk_gated_delta_rule(q[:, :b], k[:, :b], v[:, :b], g=g[:, :b], beta=beta[:, :b],
                                     initial_state=s0.clone(), **kw)
    ok_cap = torch.equal(cap, fs)
    ok_plain = torch.equal(o1, o0) and torch.equal(f1, f0)
    ok_rows = torch.equal(o1[:, :b], os_)
    fails += (not ok_cap) + (not ok_plain)
    md = (cap - fs).abs().max().item()
    bf = (fs.to(torch.bfloat16).float() - fs).abs().max().item()
    print(f"RESULT T={T:>5} b={b:>5}: capture vs split state {'bit-identical' if ok_cap else f'max|d| {md:.3e}'}; "
          f"outputs/final with vs without capture {'bit-identical' if ok_plain else 'DIFFER'}; "
          f"rows < b one vs split forward {'bit-identical' if ok_rows else 'differ'}; "
          f"bf16 state would be off by up to {bf:.3e}", flush=True)
print("PASS" if not fails else f"FAIL {fails}", flush=True)
