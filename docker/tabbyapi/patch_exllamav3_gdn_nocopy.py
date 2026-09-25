"""Drop the full-tensor copies around GatedDeltaNet's chunked prefill.

For a prefill chunk (split projections, the chunked delta rule) the fork materializes the
same data four times on its way into and out of the FLA kernels:

  1. qkv (bsz, seq, dim) fp16 -> transpose(1, 2).to(bf16).contiguous() for the conv kernel;
  2-3. the conv output (bsz, seq, dim) is split into q, k, v views, which FLA's input_guard
     copies to contiguous tensors (three copies);
  4. torch.cat() of the single sequence's output (a copy for bsz 1).

Here the conv kernel reads the fp16 projection output in place and rounds it to bf16 on
load (the same round-to-nearest-even as the torch .to()), writes q, k and v as three
contiguous tensors (so input_guard has nothing to copy), and a single sequence's output is
used as is. Every value the kernels see is unchanged, so the output is bit-identical
(tools/prefill_parity.py TOGGLE=EXL3_GDN_NOCOPY).

Only the chunked prefill path changes (seq >= num_v_heads, no SD history, split
projections); decode and MTP keep the fused C++ path. EXL3_GDN_NOCOPY=0 at run time
restores the copies.

Usage: python3 patch_exllamav3_gdn_nocopy.py [exllamav3 package dir]
Run once at image build time, AFTER patch_exllamav3_checkpoints.py (both touch
gated_delta_net.py); exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

X_LOAD_OLD = """x_vals = tl.load(
                x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],"""
X_LOAD_NEW = """x_vals = _load_x(
                x, pid_b, offs_d, x_t, dim, seq_len, X_BSD,"""
X_LOAD2_OLD = """    x_vals = tl.load(
        x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],"""
X_LOAD2_NEW = """    x_vals = _load_x(
        x, pid_b, offs_d, x_t, dim, seq_len, X_BSD,"""
STORE_OLD = """    acc = acc * tl.sigmoid(acc)
    if transpose_output:"""
STORE_NEW = """    acc = acc * tl.sigmoid(acc)
    if SPLIT_K > 0:
        _store_split(out, out_k, out_v, acc, pid_b, pid_d * BLOCK_D, offs_s, offs_d, dim, seq_len,
                     SPLIT_K, mask_d[:, None] & mask_s[None, :])
    elif transpose_output:"""

EDITS = {
    "modules/gated_delta_net_fn/conv1d.py": [
        # helpers, ahead of the first kernel
        ("""@triton.jit
def _causal_conv1d_update_slotted_kernel(
""", '''@triton.jit
def _load_x(x, pid_b, offs_d, x_t, dim, seq_len, X_BSD: tl.constexpr, mask, other):
    """X_BSD: x is the (bsz, seq_len, dim) fp16 projection output, rounded to bf16 on load
    exactly as the torch .to(torch.bfloat16) did (patch_exllamav3_gdn_nocopy.py)"""
    if X_BSD:
        v = tl.load(x + (pid_b * seq_len + x_t[None, :]) * dim + offs_d[:, None], mask = mask, other = other)
        return v.to(tl.bfloat16)
    else:
        return tl.load(x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :], mask = mask, other = other)


@triton.jit
def _store_split(out_q, out_k, out_v, acc, pid_b, d0, offs_s, offs_d, dim, seq_len,
                 SPLIT_K: tl.constexpr, mask):
    """(bsz, seq_len, *) q / k / v outputs instead of one (bsz, seq_len, dim) tensor; a
    BLOCK_D tile never straddles two of them (SPLIT_K is a multiple of BLOCK_D)"""
    if d0 < SPLIT_K:
        tl.store(out_q + (pid_b * seq_len + offs_s[None, :]) * SPLIT_K + offs_d[:, None], acc, mask = mask)
    elif d0 < 2 * SPLIT_K:
        tl.store(out_k + (pid_b * seq_len + offs_s[None, :]) * SPLIT_K + (offs_d[:, None] - SPLIT_K),
                 acc, mask = mask)
    else:
        tl.store(out_v + (pid_b * seq_len + offs_s[None, :]) * (dim - 2 * SPLIT_K) + (offs_d[:, None] - 2 * SPLIT_K),
                 acc, mask = mask)


@triton.jit
def _causal_conv1d_update_slotted_kernel(
'''),
        # new kernel parameters (defaults keep every existing launch unchanged)
        ("""    BLOCK_K: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)""", """    BLOCK_K: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    out_k = None,
    out_v = None,
    X_BSD: tl.constexpr = 0,
    SPLIT_K: tl.constexpr = 0,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)"""),
        ("""    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_s = tl.program_id(2)""", """    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    out_k = None,
    out_v = None,
    X_BSD: tl.constexpr = 0,
    SPLIT_K: tl.constexpr = 0,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_s = tl.program_id(2)"""),
        ("""    BLOCK_D: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_state = tl.arange(0, BLOCK_STATE)""", """    BLOCK_D: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    X_BSD: tl.constexpr = 0,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_state = tl.arange(0, BLOCK_STATE)"""),
        # x loads (two in the loop bodies, two in the state writers) and the output stores
        (X_LOAD_OLD, X_LOAD_NEW, 2),
        (X_LOAD2_OLD, X_LOAD2_NEW, 2),
        (STORE_OLD, STORE_NEW, 2),
        # wrapper
        ("""    transpose_output: bool = False,
    history: bool = False,
) -> torch.Tensor:
    if not x.is_cuda:""", """    transpose_output: bool = False,
    history: bool = False,
    x_bsd: bool = False,
    split_k: int = 0,
) -> torch.Tensor:
    if not x.is_cuda:"""),
        ("""        raise RuntimeError(f"bias is on {bias.device}, expected {x.device}")

    bsz, dim, seq_len = x.shape
    state_size = conv_state.shape[-1]""", """        raise RuntimeError(f"bias is on {bias.device}, expected {x.device}")

    if x_bsd:
        bsz, seq_len, dim = x.shape
    else:
        bsz, dim, seq_len = x.shape
    state_size = conv_state.shape[-1]"""),
        ("""    out_shape = (bsz, seq_len, dim) if transpose_output else (bsz, dim, seq_len)
    out = torch.empty(out_shape, dtype = x.dtype, device = x.device)
    block_d = 32""", """    out_shape = (bsz, seq_len, dim) if transpose_output else (bsz, dim, seq_len)
    block_d = 32
    if split_k:
        assert transpose_output and split_k % block_d == 0 and dim > 2 * split_k
        out = torch.empty((bsz, seq_len, split_k), dtype = torch.bfloat16, device = x.device)
        out_k = torch.empty((bsz, seq_len, split_k), dtype = torch.bfloat16, device = x.device)
        out_v = torch.empty((bsz, seq_len, dim - 2 * split_k), dtype = torch.bfloat16, device = x.device)
    else:
        out = torch.empty(out_shape, dtype = torch.bfloat16 if x_bsd else x.dtype, device = x.device)
        out_k = out_v = out"""),
        ("""                BLOCK_K = block_k,
                BLOCK_STATE = block_state,
                num_warps = 4,
            )""", """                BLOCK_K = block_k,
                BLOCK_STATE = block_state,
                out_k = out_k,
                out_v = out_v,
                X_BSD = 1 if x_bsd else 0,
                SPLIT_K = split_k,
                num_warps = 4,
            )"""),
        ("""                BLOCK_S = block_s,
                BLOCK_K = block_k,
                num_warps = 4,
            )""", """                BLOCK_S = block_s,
                BLOCK_K = block_k,
                out_k = out_k,
                out_v = out_v,
                X_BSD = 1 if x_bsd else 0,
                SPLIT_K = split_k,
                num_warps = 4,
            )"""),
        ("""                BLOCK_D = block_d,
                BLOCK_STATE = block_state,
                num_warps = 4,
            )
    return out
""", """                BLOCK_D = block_d,
                BLOCK_STATE = block_state,
                X_BSD = 1 if x_bsd else 0,
                num_warps = 4,
            )
    if split_k:
        return out, out_k, out_v
    return out


def causal_conv1d_update_split(qkv, conv_state, recurrent_slots, conv1d_weight, conv1d_bias, k_dim):
    \"\"\"Prefill form of causal_conv1d_update (patch_exllamav3_gdn_nocopy.py): qkv is the
    (bsz, seq, dim) fp16 projection output, read in place; returns contiguous bf16 q, k, v
    (bsz, seq, k_dim / k_dim / v_dim) with the values causal_conv1d_update's output holds.\"\"\"
    return causal_conv1d_update_slotted_triton(
        qkv, conv_state, recurrent_slots, conv1d_weight, conv1d_bias,
        transpose_output = True, history = False, x_bsd = True, split_k = k_dim,
    )
"""),
    ],
    "modules/gated_delta_net_fn/gated_delta_rule.py": [
        ("""    if params is None:
        params = {}

    bsz, seqlen, _ = mixed_qkv.shape
""", """    if params is None:
        params = {}

    # patch_exllamav3_gdn_nocopy.py: mixed_qkv may arrive as contiguous (q, k, v) for the
    # chunked rule
    bsz, seqlen, _ = (mixed_qkv[0] if isinstance(mixed_qkv, tuple) else mixed_qkv).shape
"""),
        ("""        from ...vendor.fla import chunk_gated_delta_rule

        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
""", """        from ...vendor.fla import chunk_gated_delta_rule

        if isinstance(mixed_qkv, tuple):
            q, k, v = mixed_qkv
        else:
            q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
"""),
        ("""        core_attn_out = torch.cat(core_attn_out, dim = 0)
""", """        if len(core_attn_out) == 1 and isinstance(mixed_qkv, tuple):
            core_attn_out = core_attn_out[0]
        else:
            core_attn_out = torch.cat(core_attn_out, dim = 0)
"""),
    ],
    "modules/gated_delta_net.py": [
        ("""        # Torch path
""", """        # Torch path
        nocopy = False  # patch_exllamav3_gdn_nocopy.py
"""),
        ("""            b = self.b_proj.forward(x, params)
            a = self.a_proj.forward(x, params)

            mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
""", """            b = self.b_proj.forward(x, params)
            a = self.a_proj.forward(x, params)

            # Chunked prefill: the conv reads the fp16 projection in place and emits q, k, v
            # contiguous (EXL3_GDN_NOCOPY=0 keeps the copies)
            nocopy = (
                seqlen >= self.num_v_heads and not save_history and conv_state is not None and
                qkv.dtype == torch.half and qkv.is_contiguous() and self.k_dim % 32 == 0 and
                __import__("os").environ.get("EXL3_GDN_NOCOPY", "1") != "0"
            )
            mixed_qkv = qkv if nocopy else qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
"""),
        ("""        # Convolution
        mixed_qkv = causal_conv1d_update(
            mixed_qkv = mixed_qkv,
            conv_state = conv_state,
            recurrent_slots = recurrent_slots,
            conv1d_weight = self.conv1d_weight_flat,
            conv1d_bias = self.conv1d_bias,
            history = save_history,
            params = params,
        )
""", """        # Convolution
        if nocopy:
            from .gated_delta_net_fn.conv1d import causal_conv1d_update_split
            mixed_qkv = causal_conv1d_update_split(
                mixed_qkv, conv_state, recurrent_slots, self.conv1d_weight_flat, self.conv1d_bias,
                self.k_dim,
            )
        else:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv = mixed_qkv,
                conv_state = conv_state,
                recurrent_slots = recurrent_slots,
                conv1d_weight = self.conv1d_weight_flat,
                conv1d_bias = self.conv1d_bias,
                history = save_history,
                params = params,
            )
"""),
    ],
}

for rel, edits in EDITS.items():
    p = root / rel
    s = p.read_text()
    for e in edits:
        old, new = e[0], e[1]
        n = e[2] if len(e) > 2 else 1
        if s.count(old) != n:
            sys.exit(f"patch_exllamav3_gdn_nocopy: anchor found {s.count(old)}x (want {n}) in {rel}:\n{old[:300]}")
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)
