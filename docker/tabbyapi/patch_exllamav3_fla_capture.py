"""FLA state capture (Plan C, C3 groundwork): let the chunked gated-delta-rule prefill also
write the fp32 recurrent state at one chunk boundary inside the forward.

The fork runs the partial last page of a prompt as its own prefill forward only to get a
recurrent checkpoint at the page boundary; a forward costs ~0.2 s before it does any work.
FLA's `chunk_gated_delta_rule_fwd_h` already walks the state chunk by chunk (64 tokens) in fp32
registers, but stores the per-chunk states `h` in the input dtype (bf16) and only the final
state in fp32. This adds an optional fp32 store of the state at the start of chunk
`capture_chunk`, i.e. after `64 * capture_chunk` tokens of this call, into a caller-supplied
`capture_state` buffer ([B, HV, K, V] fp32, or [B, HV, V, K] with state_v_first). The same
registers at the same point are what a forward over only those tokens would store as its final
state, so on identical inputs the capture equals the split forward's state bit for bit.

Nothing changes unless a caller passes `capture_state` (`capture_chunk` < the number of chunks):
`chunk_gated_delta_rule(..., capture_state=buf, capture_chunk=n)`.

Usage: python3 patch_exllamav3_fla_capture.py [exllamav3 package dir]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, re, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

p = root / "vendor/fla/chunk_delta_h.py"
s = p.read_text()

# The capture store: the kernel's final-state store, aimed at hb, inside the chunk loop
final = s[s.index("    if STORE_FINAL_STATE:\n        if STATE_V_FIRST:\n            p_ht"):]
final = final[:final.index("tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)\n") +
              len("tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)\n")]
capture = final.replace("STORE_FINAL_STATE", "STORE_B").replace("p_ht", "p_hb").replace("m_ht", "m_hb") \
               .replace("ht +", "hb +")
body = capture[capture.index("\n") + 1:]
capture = "        if STORE_B:\n            if i_t == TB:\n" + \
    "\n".join(("        " + l if l else l) for l in body.split("\n"))

edits = [
    ("    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,\n",
     "    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,\n"
     "    'STORE_B': lambda args: args['hb'] is not None,\n"),
    ("@triton.jit(do_not_specialize=['T'])\ndef chunk_gated_delta_rule_fwd_kernel_h_blockdim64(",
     "@triton.jit(do_not_specialize=['T', 'TB'])\ndef chunk_gated_delta_rule_fwd_kernel_h_blockdim64("),
    ("    h0,\n    ht,\n    cu_seqlens,\n    chunk_offsets,\n    T,\n",
     "    h0,\n    ht,\n    hb,\n    cu_seqlens,\n    chunk_offsets,\n    T,\n    TB,\n"),
    ("    STORE_FINAL_STATE: tl.constexpr,\n    SAVE_NEW_VALUE: tl.constexpr,\n",
     "    STORE_FINAL_STATE: tl.constexpr,\n    STORE_B: tl.constexpr,\n    SAVE_NEW_VALUE: tl.constexpr,\n"),
    ("    if STORE_FINAL_STATE:\n        ht = ht + i_nh * K*V\n",
     "    if STORE_FINAL_STATE:\n        ht = ht + i_nh * K*V\n    if STORE_B:\n        hb = hb + i_nh * K*V\n"),
    ("        o_t = i_t * BT + tl.arange(0, BT)\n        m_t = o_t < T\n",
     "        o_t = i_t * BT + tl.arange(0, BT)\n        m_t = o_t < T\n" + capture),
    # host wrapper
    ("    chunk_indices: torch.LongTensor | None = None,\n) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n"
     "    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]\n",
     "    chunk_indices: torch.LongTensor | None = None,\n"
     "    capture_state: torch.Tensor | None = None,\n"
     "    capture_chunk: int = 0,\n"
     ") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:\n"
     "    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]\n"),
    ("        ht=final_state,\n        cu_seqlens=cu_seqlens,\n        chunk_offsets=chunk_offsets,\n        T=T,\n",
     "        ht=final_state,\n        hb=capture_state,\n        cu_seqlens=cu_seqlens,\n        chunk_offsets=chunk_offsets,\n"
     "        T=T,\n        TB=capture_chunk,\n"),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"patch_exllamav3_fla_capture: anchor not found exactly once in chunk_delta_h.py:\n{old}")
    s = s.replace(old, new)
p.write_text(s)
print("patched vendor/fla/chunk_delta_h.py")

p = root / "vendor/fla/__init__.py"
s = p.read_text()
edits = [
    ("    use_qk_l2norm_in_kernel: bool = False,\n    chunk_size: int = 64,\n) -> tuple[torch.Tensor, torch.Tensor | None]:\n"
     "    \"\"\"\n    Chunked gated delta rule (Gated DeltaNet prefill).\n",
     "    use_qk_l2norm_in_kernel: bool = False,\n    chunk_size: int = 64,\n"
     "    capture_state: torch.Tensor | None = None,\n    capture_chunk: int = 0,\n"
     ") -> tuple[torch.Tensor, torch.Tensor | None]:\n"
     "    \"\"\"\n    Chunked gated delta rule (Gated DeltaNet prefill).\n"),
    ("    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n        k = k, w = w, u = u, g = g,\n"
     "        initial_state = initial_state,\n        output_final_state = output_final_state,\n"
     "        chunk_size = chunk_size,\n    )\n    o = chunk_fwd_o(",
     "    if capture_state is not None:\n"
     "        assert chunk_size == 64 and 0 <= capture_chunk < (q.shape[1] + 63) // 64\n"
     "    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n        k = k, w = w, u = u, g = g,\n"
     "        initial_state = initial_state,\n        output_final_state = output_final_state,\n"
     "        chunk_size = chunk_size,\n"
     "        capture_state = capture_state,\n        capture_chunk = capture_chunk,\n    )\n    o = chunk_fwd_o("),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"patch_exllamav3_fla_capture: anchor not found exactly once in __init__.py:\n{old}")
    s = s.replace(old, new)
p.write_text(s)
print("patched vendor/fla/__init__.py")
