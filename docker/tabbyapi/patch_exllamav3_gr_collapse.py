"""One fused kernel for the hyper-connection stream collapse in prefill.

Every transformer block of this model mixes its 4 residual streams twice
(GatedResidual.mix, before attention and before the MoE). For prefill-sized
inputs (more than 32 rows) the fork does the final step in torch:

    mixed = (sigmoid(g.float()).view(R, H, D) * normed.float().view(R, H, D)).mean(-2).half()

which materialises four fp32 tensors of R x 4 x 2560 (~335 MB each for an
8,192-token chunk) per call. Measured on an 8,192-token prompt, the blocks' own
code (mostly this) took ~2.2 s of 8.4 s. ext.gr_collapse computes the same thing
in one pass: reads g and normed once (fp16) and computes what torch computes, in
the same order and with the same expf, so the output is bit-identical.

EXL3_GR_COLLAPSE=0 at run time restores the torch expression.

Usage: python3 patch_exllamav3_gr_collapse.py <exllamav3 source dir, the one with setup.py>
Run at image build time, before the extension is compiled; exits non-zero if an
anchor is missing.
"""
import pathlib, sys

src = pathlib.Path(sys.argv[1]) / "exllamav3"
ext = src / "exllamav3_ext"

def edit(path, reps, append = None):
    s = path.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_gr_collapse: anchor not found exactly once in {path.name}: {old[:60]!r}")
        s = s.replace(old, new)
    if append:
        s += append
    path.write_text(s)
    print("patched", path.relative_to(src.parent))

edit(ext / "hc_mix.cuh", [], append = """
// GatedResidual prefill collapse (patch_exllamav3_gr_collapse.py):
// mixed[r, d] = mean_h sigmoid(g[r, h * D + d]) * normed[r * H + h, d]
void gr_collapse
(
    const at::Tensor& g,
    const at::Tensor& normed,
    at::Tensor mixed
);
""")

edit(ext / "hc_mix.cu", [], append = """

// GatedResidual prefill collapse (patch_exllamav3_gr_collapse.py). One thread per 8 columns
// of one row: reads the H gate and normed values once (fp16), sums the H products in fp32 in
// stream order (the order torch's mean over dim -2 uses), divides by H, writes fp16.
// Bit-identical to the torch expression it replaces: torch's sigmoid is 1 / (1 + expf(-x))
// with libdevice's expf, which the --use_fast_math build would turn into __expf, so the
// libdevice function is called by name; the _rn intrinsics keep the multiply-add unfused and
// the division exact.

extern "C" __device__ float __nv_expf(float);

template <int H>
__global__ __launch_bounds__(256)
void gr_collapse_kernel
(
    const half* __restrict__ g,          // (R, H * D)
    const half* __restrict__ normed,     // (R * H, D)
    half* __restrict__ mixed,            // (R, D)
    const int R,
    const int D
)
{
    const int D8 = D / 8;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) R * D8) return;
    const int r = (int) (i / D8);
    const int c = (int) (i % D8);

    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;

    auto sig = [] (float x) -> float
    {
        return __fdiv_rn(1.0f, __fadd_rn(1.0f, __nv_expf(-x)));
    };

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        int4 gv = ((const int4*) (g + ((size_t) r * H + h) * D))[c];
        int4 nv = ((const int4*) (normed + ((size_t) r * H + h) * D))[c];
        const half2* g2 = (const half2*) &gv;
        const half2* n2 = (const half2*) &nv;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            float2 gf = __half22float2(g2[j]);
            float2 nf = __half22float2(n2[j]);
            acc[2 * j]     = __fadd_rn(acc[2 * j],     __fmul_rn(sig(gf.x), nf.x));
            acc[2 * j + 1] = __fadd_rn(acc[2 * j + 1], __fmul_rn(sig(gf.y), nf.y));
        }
    }

    int4 out;
    half2* o2 = (half2*) &out;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        o2[j] = __floats2half2_rn(__fdiv_rn(acc[2 * j], (float) H), __fdiv_rn(acc[2 * j + 1], (float) H));
    ((int4*) (mixed + (size_t) r * D))[c] = out;
}

void gr_collapse
(
    const at::Tensor& g,                 // (R, H * D) half
    const at::Tensor& normed,            // (R * H, D) half
    at::Tensor mixed                     // (R, D) half
)
{
    const at::cuda::OptionalCUDAGuard device_guard(g.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(g, kHalf);
    TORCH_CHECK_DTYPE(normed, kHalf);
    TORCH_CHECK_DTYPE(mixed, kHalf);
    TORCH_CHECK(g.is_contiguous() && normed.is_contiguous() && mixed.is_contiguous(), "gr_collapse: contiguous tensors only");
    const int R = mixed.size(0);
    const int D = mixed.size(1);
    const int H = g.size(1) / D;
    TORCH_CHECK(g.size(0) == R && g.size(1) == H * D, "gr_collapse: g shape");
    TORCH_CHECK(normed.size(0) == R * H && normed.size(1) == D, "gr_collapse: normed shape");
    TORCH_CHECK(D % 8 == 0, "gr_collapse: D must be a multiple of 8");

    const int64_t n = (int64_t) R * (D / 8);
    const int threads = 256;
    const int blocks = (int) ((n + threads - 1) / threads);
    #define LAUNCH(HH) gr_collapse_kernel<HH><<<blocks, threads, 0, stream>>>( \\
        (const half*) g.data_ptr(), (const half*) normed.data_ptr(), (half*) mixed.data_ptr(), R, D)
    switch (H)
    {
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
        default: TORCH_CHECK(false, "gr_collapse: unsupported stream count ", H);
    }
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}
""")

edit(ext / "bindings.cpp", [
    ("""    m.def("hc_apply", &hc_apply, "hc_apply");
""",
     """    m.def("hc_apply", &hc_apply, "hc_apply");
    m.def("gr_collapse", &gr_collapse, "gr_collapse");
"""),
])

edit(src / "modules" / "hyperconnections.py", [
    ("""_GR_INT8 = _os.environ.get("EXL3_GR_INT8", "1") != "0"
""",
     """_GR_INT8 = _os.environ.get("EXL3_GR_INT8", "1") != "0"
# EXL3_GR_COLLAPSE (default on; =0 disables): the prefill stream collapse as one fused kernel
# (ext.gr_collapse, patch_exllamav3_gr_collapse.py) instead of four fp32 temporaries in torch
_GR_COLLAPSE = _os.environ.get("EXL3_GR_COLLAPSE", "1") != "0"
"""),
    ("""            g = torch.matmul(t, self.up_h.t())                             # (R, H * Dh)
            mixed = (torch.sigmoid(g.float()).view(R, H, Dh)
                     * normed.float().view(R, H, Dh)).mean(dim = -2).half()
""",
     """            g = torch.matmul(t, self.up_h.t())                             # (R, H * Dh)
            if _GR_COLLAPSE and g.dtype == torch.half and g.is_contiguous():
                mixed = torch.empty((R, Dh), dtype = torch.half, device = dev)
                ext.gr_collapse(g, normed, mixed)
            else:
                mixed = (torch.sigmoid(g.float()).view(R, H, Dh)
                         * normed.float().view(R, H, Dh)).mean(dim = -2).half()
"""),
])
