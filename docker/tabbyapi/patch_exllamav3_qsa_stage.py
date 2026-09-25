"""Stage the 8-bit K/V once per layer for sparse (QSA) attention in prefill.

With the quantized cache, the gathered-GQA kernel (_qsa_sparse_split_kernel) dequantizes
every K/V tile it gathers. In a prefill chunk every query row selects ~2,048 cache
positions, so each cached token is dequantized by thousands of rows. This adds a staging
kernel that dequantizes the sequence's positions [0, n_tok) once into fp16 buffers, still
in the H32-rotated domain, with the same expression (_qc_load_v), and runs the gather
kernel on them as plain fp16 K/V with q rotated as before (ROTQ). The kernel sees the same
fp16 values in the same tiles, so the output is bit-identical (checked with
tools/qsa_ab.py on captured prefill inputs: 0 of 123M elements differ); the sparse
attention kernel runs ~2x faster.

Staging runs only for one sequence (bsz 1) and when the chunk gathers at least 4x as many
positions as it would stage (rows * K_pad >= 4 * n_tok), so decode fallbacks and MTP
verify batches keep the direct path. It costs n_tok x 2 KiB of transient memory per
layer (K and V, fp16): ~235 MB at 115k tokens.

EXL3_QSA_STAGE=0 at run time restores the direct path.

Usage: python3 patch_exllamav3_qsa_stage.py [exllamav3 package dir]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

EDITS = {
    "modules/attention_fn/qsa_triton.py": [
        # gather kernel: fp16 K/V that are still rotated -> rotate q as for packed K
        ("""    QCV: tl.constexpr = 0,   # packed quantized V pages (bits), 0 = fp16
):
""", """    QCV: tl.constexpr = 0,   # packed quantized V pages (bits), 0 = fp16
    ROTQ: tl.constexpr = 0,  # fp16 K/V still in the H32-rotated domain (staged, see below)
):
"""),
        ("""    if QCK > 0:
        # Packed keys live in the H32-rotated domain: rotate q once, scores stay exact
        q_tile = _rot_h32(q_tile, h32, BLOCK_H, head_dim)
""", """    if QCK > 0 or ROTQ:
        # Packed keys live in the H32-rotated domain: rotate q once, scores stay exact
        q_tile = _rot_h32(q_tile, h32, BLOCK_H, head_dim)
"""),
        # staging kernel
        ("""_sm_counts = {}
""", '''@triton.jit(do_not_specialize = ["n_tok"])
def _qsa_stage_kv_kernel(
    qwords,              # packed int32 pages (CacheLayer_quant)
    scales,
    block_table,         # (num_pages,) int32, one sequence
    out,                 # (n_tok, n_kv_heads, head_dim) fp16, logical position order
    n_tok,
    page_size: tl.constexpr,
    BITS: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dequantize one sequence's cache positions [0, n_tok) to fp16, still in the rotated
    domain, with the expression the gather kernel evaluates per tile (_qc_load_v): once per
    position instead of once per (query row, selected position). patch_exllamav3_qsa_stage.py"""
    t = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    kv_head = tl.program_id(1)
    m = t < n_tok
    tc = tl.where(m, t, 0)
    phys = tl.load(block_table + tc // page_size, mask = m, other = 0)
    rows = phys * page_size + tc % page_size
    offs_d = tl.arange(0, head_dim)
    tile = _qc_load_v(qwords, scales, rows, kv_head, offs_d, m, BITS, n_kv_heads, head_dim, head_dim)
    tl.store(out + (t[:, None] * n_kv_heads + kv_head) * head_dim + offs_d[None, :], tile,
             mask = m[:, None])


def qsa_stage_kv(qwords, scales, bits, block_table_row, n_tok, n_kv_heads, head_dim, page_size):
    """(n_tok, n_kv_heads, head_dim) fp16 rotated-domain copy of one sequence's packed K or V."""
    out = torch.empty((n_tok, n_kv_heads, head_dim), dtype = torch.half, device = qwords.device)
    BLOCK_N = 32
    with torch.cuda.device(qwords.device):
        _qsa_stage_kv_kernel[(triton.cdiv(n_tok, BLOCK_N), n_kv_heads)](
            qwords, scales, block_table_row, out, n_tok,
            page_size = page_size, BITS = bits, n_kv_heads = n_kv_heads, head_dim = head_dim,
            BLOCK_N = BLOCK_N, num_warps = 4, num_stages = 1,
        )
    return out


_sm_counts = {}
'''),
        # wrapper: staged (k16, v16) -> flat fp16 gather, rotated q, combine still rotates back
        ("""    n_kv_heads: int | None = None, # required with qc (the packed rows carry no head axis)
) -> torch.Tensor:
""", """    n_kv_heads: int | None = None, # required with qc (the packed rows carry no head axis)
    staged: tuple | None = None,   # (k16, v16) from qsa_stage_kv of qc's K/V; indices are then
                                   # positions of that one sequence (flat gather, no pages)
) -> torch.Tensor:
"""),
        ("""    paged = block_table is not None
    assert q.is_contiguous()""", """    paged = block_table is not None
    rotq, v_bits_combine = 0, v_bits
    if staged is not None:
        k, v = staged
        paged, rotq, block_table = False, 1, None
        k_scales, v_scales, k_bits, v_bits = q, q, 0, 0
    assert q.is_contiguous()"""),
        ("""            QCK = k_bits, QCV = v_bits,
            num_warps = 4, num_stages = 2,
""", """            QCK = k_bits, QCV = v_bits, ROTQ = rotq,
            num_warps = 4, num_stages = 2,
"""),
        ("""            QCV = v_bits, HAS_SINKS = False, q_len = 1,""",
         """            QCV = v_bits_combine, HAS_SINKS = False, q_len = 1,"""),
    ],
    # the decode graph compiles the same kernel ahead of time: declare the new constexpr
    "modules/attention_fn/bc_attn.py": [
        ("""                    "BLOCK_H", "BLOCK_N", "PAGED", "QCK", "QCV")},""",
         """                    "BLOCK_H", "BLOCK_N", "PAGED", "QCK", "QCV", "ROTQ")},"""),
        ("""                     PAGED = 1, QCK = self.k_bits, QCV = self.v_bits),""",
         """                     PAGED = 1, QCK = self.k_bits, QCV = self.v_bits, ROTQ = 0),"""),
    ],
    "modules/qsa_indexer.py": [
        ("""        o = qsa_sparse_attend_rows(
            q.reshape(bsz * seq, attn.num_q_heads, attn.head_dim).contiguous(),
            k_arg, v_arg, indices, attn.sm_scale,
            block_table = bt_rows, page_size = page_size,
            qc = qc, n_kv_heads = attn.num_kv_heads,
        )
""", """        # patch_exllamav3_qsa_stage.py: dequantize this sequence's K/V once when the chunk
        # gathers far more positions than it holds (prefill); EXL3_QSA_STAGE=0 disables
        staged = None
        if qc is not None and bsz == 1 and __import__("os").environ.get("EXL3_QSA_STAGE", "1") != "0":
            n_tok = int(cache_seqlens_cpu[0]) + seq
            if seq * indices.shape[1] >= 4 * n_tok:
                from .attention_fn.qsa_triton import qsa_stage_kv
                bt0 = block_table[0].int().contiguous()
                staged = tuple(
                    qsa_stage_kv(w, s, b, bt0, n_tok, attn.num_kv_heads, attn.head_dim, page_size)
                    for w, s, b in ((qk, sk, kb), (qv, sv, vb))
                )
        o = qsa_sparse_attend_rows(
            q.reshape(bsz * seq, attn.num_q_heads, attn.head_dim).contiguous(),
            k_arg, v_arg, indices, attn.sm_scale,
            block_table = bt_rows, page_size = page_size,
            qc = qc, n_kv_heads = attn.num_kv_heads, staged = staged,
        )
"""),
    ],
}

for rel, edits in EDITS.items():
    p = root / rel
    s = p.read_text()
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_qsa_stage: anchor not found exactly once in {rel}:\n{old[:200]}")
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)
