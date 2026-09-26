"""Speculative sampling for MTP drafts (EXL3_SPEC_SAMPLE=1) and the per-request A/B switch
for the live trial (EXL3_TRIAL_AB=1). Plan D, D1 and D4.

The fork drafts the MTP head's argmax d and keeps it only if the target's own sample equals
it, so a draft is accepted with probability p(d). Standard speculative sampling (Leviathan et
al. 2023; Chen et al. 2023) draws d from the draft distribution q instead, accepts it with
probability min(1, p(d) / q(d)) and, on rejection, samples the position from the normalized
residual max(0, p - q). The emitted tokens are distributed exactly as p, and a draft is
accepted with probability sum_x min(p(x), q(x)) = 1 - TV(p, q). Thinking (sampled at
temperature 1.0, top-k 20, top-p 0.95) has flat distributions, where that is much larger:
offline sizing (tools/spec_sample_size.py) measured +10.5% tokens per verify round.

Draft side (MTP head, qwen4_exp_mtp.sample_from_state): for a job this applies to, d is drawn
from q = the head's logits at the job's temperature, cut to its top-k (at most 64 tokens) and
top-p, and q is kept per position. Any q keeps the output exact; this one is close to p.
Greedy jobs and jobs this does not apply to keep the argmax draft.

Verify side (Generator.iterate_gen): the job's sampler (the fused kernel, SS_Fused) samples
all positions of the verify window in one call. Its histogram workspace then holds, per row,
the max logit and the kept-set bound, and the kept set is recomputed here with the kernel's
own fp32 binning expression, so p is exactly the distribution the kernel samples from
(softmax at the job's temperature over that set). Per drafted position i, with u ~ U(0, 1):
accept d_i if u * q_i(d_i) < p_i(d_i); at the first rejection emit a sample of
max(0, p_i - q_i) (normalized); if every draft is accepted, the kernel's own sample of the
last position is the bonus token. One device-to-host copy per round.

Prompt-lookup drafts (patch_exllamav3_pld.py) are point masses (q = 1 at the drafted token):
accept with p(d), residual p without d, i.e. the current rule. In a gated lookup round the
first position was drawn from the MTP head's q and keeps it.

Applies to a job when its sampler is the fused kernel alone (no active penalties, logit bias,
bans or min-p; TabbyAPI's no-op penalty steps are simplified away) and it has no filters,
forced tokens, logit masks or probability/logit outputs. Anything else runs the fork's path
unchanged; with sampled drafts that path is still exact, because its output is always the
target's own sample. Dynamic drafting keeps working on the head's max logit; rounds with
speculative sampling label their own calibrator, since their acceptance per score differs.

EXL3_TRIAL_AB=1: requests alternate by generator serial number between arm A (this off) and
arm B (on, if EXL3_SPEC_SAMPLE=1; with EXL3_SPEC_SAMPLE=0 both arms run the same code, an A/A
check). The arm is logged in [decode-stats] (patch_exllamav3_decode_stats.py).

Usage: python3 patch_exllamav3_specsample.py [exllamav3 package dir]
Run once at image build time, after patch_exllamav3_pld.py and patch_exllamav3_decode_stats.py;
exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

MODULE = r'''"""Speculative sampling for MTP drafts (patch_exllamav3_specsample.py; see there)."""
import os
import torch
from .sampler.custom import SamplingState, SS_Fused
from .draft_confidence import DraftConfidenceCalibrator

SPEC = os.environ.get("EXL3_SPEC_SAMPLE", "0") == "1"
TRIAL_AB = os.environ.get("EXL3_TRIAL_AB", "0") == "1"
ACTIVE = SPEC or TRIAL_AB
Q_MAX_K = 64                     # the draft distribution keeps at most this many tokens

# The fused kernel's histogram workspace (sampling_fused.cuh): per row a coarse and a refinement
# histogram (1024 buckets x (u64 mass + u32 count)), then the control block
# { float m; u32 status, ref_bucket, k_rem, keep_b, keep_s; u64 target }
HIST_BUCKETS = 1024
HIST_RANGE = 32.0
CTRL_OFFSET = 2 * HIST_BUCKETS * 12


def arm(job):
    """'A' / 'B' by serial number with EXL3_TRIAL_AB, else '-'; kept on the job (and in its
    [decode-stats] line)"""
    a = getattr(job, "_trial_arm", None)
    if a is None:
        a = ("B" if job.serial_number % 2 else "A") if TRIAL_AB else "-"
        job._trial_arm = a
    return a


def job_step(job):
    """The job's fused sampling step if speculative sampling applies to it, else None"""
    s = job.sampler
    if not getattr(s, "fused_only", False) or not getattr(s, "steps", None):
        return None
    st = s.steps[-1]
    if not isinstance(st, SS_Fused):
        return None
    if st.mode not in (SS_Fused.MODE_SAMPLE, SS_Fused.MODE_SAMPLE_FILTERS) or st.filters & SS_Fused.F_MINP:
        return None
    if (job.filters or job.forced_ids is not None or job.return_probs or job.return_top_tokens or
            getattr(job, "return_logits", False) or len(job.sequences) != 1):
        return None
    return st


def enabled_for(job):
    return SPEC and (not TRIAL_AB or arm(job) == "B")


def job_generator(job, device):
    g = getattr(job, "_spec_gen", None)
    if g is None or g.device != device:
        g = torch.Generator(device = device)
        g.manual_seed(job.rng.randint(0, (1 << 62) - 1))
        job._spec_gen = g
    return g


def draft_rows(jobs):
    """Per-row draft settings for this round's MTP chain, (cfgs, jobs), or None if no row
    samples its drafts. cfg = (inv_temp, top_k, top_p, job) or None (argmax draft)"""
    cfgs = []
    for job in jobs:
        arm(job)
        job._spec_q = None
        st = job_step(job) if enabled_for(job) else None
        if st is None or job.new_tokens < 0:
            cfgs.append(None)
            continue
        job._spec_q = []
        job._spec_step = st
        top_k = st.top_k if st.filters & SS_Fused.F_TOPK else 0
        top_p = st.top_p if st.filters & SS_Fused.F_TOPP else 1.0
        cfgs.append((st.inv_temp, top_k, top_p, job))
    if all(c is None for c in cfgs):
        return None
    return cfgs, list(jobs)


def q_dist(logits_row, inv_temp, top_k, top_p):
    """(ids, probs) of the draft distribution q for a row (or rows) of head logits"""
    x = logits_row.float() * inv_temp
    k = top_k if 0 < top_k <= Q_MAX_K else Q_MAX_K
    k = min(k, x.shape[-1])
    v, i = torch.topk(x, k)
    pr = torch.softmax(v, dim = -1)
    if top_p < 1.0:
        c = torch.cumsum(pr, dim = -1)
        pr = pr * ((c - pr) < top_p)
        pr = pr / pr.sum(dim = -1, keepdim = True)
    return i, pr


def draft_sample(y, ids, params):
    """Replace the argmax draft ids of sampling rows by a sample of q; keep q per row in
    params["spec_q"] (None for argmax rows). y: (rows, vocab') head logits, ids: (rows,)"""
    cfgs, _ = params["spec_cfg"]
    out = []
    ids = ids.clone()
    for row, cfg in enumerate(cfgs):
        if cfg is None:
            out.append(None)
            continue
        inv_temp, top_k, top_p, job = cfg
        qi, qp = q_dist(y[row], inv_temp, top_k, top_p)
        g = job_generator(job, y.device)
        ids[row] = qi[torch.multinomial(qp, 1, generator = g)[0]]
        out.append((qi, qp))
    params["spec_q"] = out
    return ids


def collect(params):
    """After a draft step: append each sampling row's q to its job"""
    cfgs, jobs = params["spec_cfg"]
    for job, q in zip(jobs, params.get("spec_q") or [None] * len(jobs)):
        if job._spec_q is not None:
            job._spec_q.append(q)


def calibrator(gen, spec):
    """The dynamic-drafting calibrator for this round: a separate one for rounds that sample
    their drafts (their acceptance per draft score differs)"""
    cal = gen.draft_calibrator
    if cal is None or spec is None:
        return cal
    sc = getattr(gen, "_spec_calibrator", None)
    if sc is None:
        sc = gen._spec_calibrator = DraftConfidenceCalibrator(
            cal.confidence, bin_width = cal.bin_width, decay = cal.decay,
            min_count = cal.min_count, burn_in = cal.burn_in)
    return sc


def fused_p(logits, step, size, rand_u32):
    """Sample every row with the fused step and return (samples (R,), p (R, V) float32), p being
    exactly the distribution the kernel sampled from. logits: (R, V) half or float"""
    R, V = logits.shape
    if logits.dtype not in (torch.half, torch.float):
        logits = logits.float()
    logits = logits.contiguous()
    st = SamplingState(rand_u32 = rand_u32, bsz = R, dim = V, in_logits = logits, fused_dim = size)
    step.run(st)
    sample = st.sample.view(R)
    x = logits.float()
    col = torch.arange(V, device = x.device)
    keep = (x != -float("inf")) & (col < size)
    if step.mode == SS_Fused.MODE_SAMPLE_FILTERS:
        hist = step.histograms[(logits.device, R)].view(R, -1)
        ctrl = hist[:, CTRL_OFFSET:CTRL_OFFSET + 24].contiguous().view(torch.int32)
        m = ctrl[:, 0:1].contiguous().view(torch.float32)
        kb = ctrl[:, 4:5]
        ks = ctrl[:, 5:6]
        scale = (torch.tensor(float(HIST_BUCKETS) / HIST_RANGE, dtype = torch.float32) *
                 torch.tensor(step.inv_temp_filter, dtype = torch.float32)).to(x.device)
        db = (m - x) * scale
        b = torch.clamp(db, max = float(HIST_BUCKETS - 1)).to(torch.int32)
        s = torch.clamp(torch.trunc((db - b.float()) * float(HIST_BUCKETS)), 0.0, float(HIST_BUCKETS - 1)).to(torch.int32)
        keep &= (x == m) | (b < kb) | ((b == kb) & (s <= ks))
    else:
        m = torch.where(keep, x, torch.full_like(x, -float("inf"))).max(dim = -1, keepdim = True).values
    inv_t = torch.tensor(step.inv_temp, dtype = torch.float32, device = x.device)
    w = torch.where(keep, torch.exp((x - m) * inv_t), torch.zeros_like(x))
    return sample, w / w.sum(dim = -1, keepdim = True)


def accept(p, q, d, u, resid, bonus):
    """The ratio test for one window (tensors, batched over a leading dim B):
    p (B, w+1, V) target distributions, q (B, w, V) draft distributions, d (B, w) drafts,
    u (B, w) uniforms, resid (B, w) samples of the normalized max(0, p - q) at each position,
    bonus (B,) a sample of p at the last position.
    Returns tokens (B, w+1) (valid up to n_acc inclusive) and n_acc (B,)"""
    B, w = d.shape
    pd = torch.gather(p[:, :w], 2, d.unsqueeze(-1)).squeeze(-1)
    qd = torch.gather(q, 2, d.unsqueeze(-1)).squeeze(-1)
    ok = u * qd < pd
    n_acc = torch.cumprod(ok.to(torch.int32), dim = 1).sum(dim = 1)
    pos = torch.arange(w + 1, device = d.device).unsqueeze(0)
    fill = torch.cat([resid, bonus.unsqueeze(1)], dim = 1)
    tok = torch.where(pos < n_acc.unsqueeze(1), torch.cat([d, d[:, -1:]], dim = 1), fill)
    return tok, n_acc


def residual_samples(p, q, g):
    """(B, w) samples of the normalized max(0, p - q) per position (p where that is empty,
    which only happens when p == q and so no rejection can occur)"""
    r = torch.clamp(p - q, min = 0.0)
    z = r.sum(dim = -1, keepdim = True)
    r = torch.where(z > 0, r, p)
    B, w, V = r.shape
    return torch.multinomial(r.view(B * w, V), 1, generator = g).view(B, w)


def dense_q(qs, d, V):
    """(w, V) draft distributions from the per-position (ids, probs), None = point mass at d"""
    w = d.shape[0]
    q = torch.zeros((w, V), dtype = torch.float32, device = d.device)
    for i in range(w):
        qi = qs[i] if i < len(qs) else None
        if qi is None:
            q[i, d[i]] = 1.0
        else:
            q[i].scatter_(0, qi[0], qi[1].float())
    return q


def verify(gen, job, job_logits, drafts):
    """Speculative-sampling verify of one job's window. job_logits (1, w+1, V), drafts (w,) cpu.
    Returns (tokens cpu (w+1,), accepted flags list (w,)) for the fork's batched-verify loop,
    or None to use the fork's path"""
    qs = getattr(job, "_spec_q", None)
    job._spec_q = None
    if qs is None or job.new_tokens < 0 or job.device_logit_mask is not None:
        return None
    step = getattr(job, "_spec_step", None)
    if step is None:
        return None
    q_rows = job_logits.shape[1]
    w = q_rows - 1
    L = job_logits.view(q_rows, -1)
    V = L.shape[-1]
    size = min(V, gen.tokenizer.actual_vocab_size)
    dev = L.device
    d = drafts.to(dev, non_blocking = True).view(w).long()
    sample, p = fused_p(L, step, size, job.rng.randint(0, (1 << 32) - 1))
    q = dense_q(qs, d, V)
    g = job_generator(job, dev)
    u = torch.rand((1, w), generator = g, device = dev)
    resid = residual_samples(p[:w].unsqueeze(0), q.unsqueeze(0), g)
    tok, n_acc = accept(p.unsqueeze(0), q.unsqueeze(0), d.unsqueeze(0), u, resid, sample[w:].view(1))
    pos = torch.arange(w, device = dev)
    match = (pos < n_acc).to(tok.dtype)
    packed = torch.cat([tok.view(-1), match]).cpu()   # single sync
    ds = getattr(job, "_ds", None)
    if ds:
        ds["extra"]["spec rounds"] = ds["extra"].get("spec rounds", 0) + 1
    return packed[:q_rows], packed[q_rows:].tolist()
'''

edits = {
    "architecture/qwen4_exp_mtp.py": [
        ("                if params.get(\"export_draft_conf\"):\n"
         "                    # -dds: conf is the raw max logit, identical to the full head's when the\n"
         "                    # argmax is in-slice; lower otherwise, so drafting stops earlier\n"
         "                    conf, ids = torch.max(y, dim = -1)\n"
         "                    params[\"draft_conf\"] = conf.view(b, q)\n"
         "                    return ids.view(b, q)\n"
         "                return torch.argmax(y, dim = -1).view(b, q)\n",
         "                if params.get(\"export_draft_conf\"):\n"
         "                    # -dds: conf is the raw max logit, identical to the full head's when the\n"
         "                    # argmax is in-slice; lower otherwise, so drafting stops earlier\n"
         "                    conf, ids = torch.max(y, dim = -1)\n"
         "                    params[\"draft_conf\"] = conf.view(b, q)\n"
         "                else:\n"
         "                    ids = torch.argmax(y, dim = -1)\n"
         "                # Speculative sampling (patch_exllamav3_specsample.py): sampled drafts\n"
         "                if params.get(\"spec_cfg\") is not None:\n"
         "                    from ..generator.spec_sample import draft_sample as _spec_draft\n"
         "                    ids = _spec_draft(y, ids, params)\n"
         "                return ids.view(b, q)\n"),
        ("        logits = lm.forward(state, params)\n"
         "        if params.get(\"export_draft_conf\"):\n"
         "            logits = logits[..., :self.attached_model().config.vocab_size]\n"
         "            conf, ids = torch.max(logits, dim = -1)\n"
         "            params[\"draft_conf\"] = conf\n"
         "            return ids\n"
         "        return torch.argmax(logits, dim = -1)\n",
         "        logits = lm.forward(state, params)\n"
         "        # Speculative sampling (patch_exllamav3_specsample.py): sampled drafts\n"
         "        if params.get(\"spec_cfg\") is not None:\n"
         "            from ..generator.spec_sample import draft_sample as _spec_draft\n"
         "            logits = logits[..., :self.attached_model().config.vocab_size]\n"
         "            conf, ids = torch.max(logits, dim = -1)\n"
         "            if params.get(\"export_draft_conf\"):\n"
         "                params[\"draft_conf\"] = conf\n"
         "            b, q = ids.shape\n"
         "            return _spec_draft(logits.reshape(b * q, -1), ids.reshape(b * q), params).view(b, q)\n"
         "        if params.get(\"export_draft_conf\"):\n"
         "            logits = logits[..., :self.attached_model().config.vocab_size]\n"
         "            conf, ids = torch.max(logits, dim = -1)\n"
         "            params[\"draft_conf\"] = conf\n"
         "            return ids\n"
         "        return torch.argmax(logits, dim = -1)\n"),
    ],
    "generator/generator.py": [
        # Draft rows for this round
        ("        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)\n"
         "        # Device-resident draft chain (EXL3_MTP_DEVICE_DRAFT)",
         "        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)\n"
         "        spec_rows = None\n"
         "        if _ss.ACTIVE:\n"
         "            spec_rows = _ss.draft_rows([job for job in self.active_jobs if job.is_prefill_done()])\n"
         "        # Device-resident draft chain (EXL3_MTP_DEVICE_DRAFT)"),
        ("        window = self.num_draft_tokens\n"
         "        cal = self.draft_calibrator\n"
         "        conf_cols = []\n"
         "        reach = None\n"
         "        for idx in range(window):\n"
         "            params = {\n"
         "                \"target_hidden\": temp_hidden,\n",
         "        window = self.num_draft_tokens\n"
         "        cal = _ss.calibrator(self, spec_rows) if _ss.ACTIVE else self.draft_calibrator\n"
         "        conf_cols = []\n"
         "        reach = None\n"
         "        for idx in range(window):\n"
         "            params = {\n"
         "                \"target_hidden\": temp_hidden,\n"),
        ("            if cal is not None:\n"
         "                params[\"export_draft_conf\"] = True\n"
         "            batch_state = self.draft_model.forward(batch_ids, params)\n"
         "            lm_head = self.model.modules[self.model.logit_layer_idx]\n"
         "            batch_state = lm_head.prepare_for_device(batch_state, params)\n"
         "            new_ids = self.draft_model.sample_from_state(batch_state, params)\n",
         "            if cal is not None:\n"
         "                params[\"export_draft_conf\"] = True\n"
         "            if spec_rows is not None:\n"
         "                params[\"spec_cfg\"] = spec_rows\n"
         "            batch_state = self.draft_model.forward(batch_ids, params)\n"
         "            lm_head = self.model.modules[self.model.logit_layer_idx]\n"
         "            batch_state = lm_head.prepare_for_device(batch_state, params)\n"
         "            new_ids = self.draft_model.sample_from_state(batch_state, params)\n"
         "            if spec_rows is not None:\n"
         "                _ss.collect(params)\n"),
        ("            self._draft_conf_round = {\n"
         "                \"ids\": self.draft_ids_pinned[:batch_size, :window],\n"
         "                \"conf\": torch.cat(conf_cols, dim = 1),\n"
         "                \"window\": window,\n"
         "            }\n"
         "\n"
         "        return self.draft_ids_pinned[:, :window]\n"
         "\n"
         "\n"
         "    def _pld_candidates(self):\n",
         "            self._draft_conf_round = {\n"
         "                \"ids\": self.draft_ids_pinned[:batch_size, :window],\n"
         "                \"conf\": torch.cat(conf_cols, dim = 1),\n"
         "                \"window\": window,\n"
         "                \"cal\": cal,\n"
         "            }\n"
         "\n"
         "        return self.draft_ids_pinned[:, :window]\n"
         "\n"
         "\n"
         "    def _pld_candidates(self):\n"),
        # Verify
        ("                pre_tokens = None\n"
         "                pre_match = None\n"
         "                if (\n"
         "                    _BATCH_VERIFY and draft_tokens is not None",
         "                pre_tokens = None\n"
         "                pre_match = None\n"
         "                # Speculative sampling (patch_exllamav3_specsample.py): ratio test on the whole\n"
         "                # window, one readback; None falls through to the fork's paths\n"
         "                if (_ss.ACTIVE and draft_tokens is not None and batch_logits.shape[1] > 1 and\n"
         "                        getattr(job, \"_spec_q\", None) is not None):\n"
         "                    _sr = _ss.verify(self, job, job_logits, draft_tokens[j, :batch_logits.shape[1] - 1])\n"
         "                    if _sr is not None:\n"
         "                        pre_tokens, pre_match = _sr\n"
         "                if pre_tokens is None and (\n"
         "                    _BATCH_VERIFY and draft_tokens is not None"),
        # Calibrator labels go to the calibrator that drafted the round
        ("            st = self._draft_conf_round\n"
         "            self._draft_conf_round = None\n"
         "            cal = self.draft_calibrator\n",
         "            st = self._draft_conf_round\n"
         "            self._draft_conf_round = None\n"
         "            cal = st.get(\"cal\") or self.draft_calibrator\n"),
    ],
}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_specsample: anchor not found exactly once in {rel}: {old[:60]!r}")
        s = s.replace(old, new)
    if rel == "generator/generator.py":
        s = s.replace("from .draft_confidence import DraftConfidenceCalibrator\n",
                      "from .draft_confidence import DraftConfidenceCalibrator\nfrom . import spec_sample as _ss\n", 1)
        if "from . import spec_sample as _ss" not in s:
            sys.exit("patch_exllamav3_specsample: import anchor not found in generator/generator.py")
    p.write_text(s)
    print("patched", rel)
(root / "generator" / "spec_sample.py").write_text(MODULE)
print("wrote generator/spec_sample.py")
