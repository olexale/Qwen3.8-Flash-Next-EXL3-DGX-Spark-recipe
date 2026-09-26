"""patch_exllamav3_specsample.py: the ratio test and residual step against a brute-force
reference, p equal to the fused kernel's sampling distribution, the emitted token
distribution over chains of 3 drafts (chi-square), and greedy/ineligible requests unchanged.
Needs a GPU (the fused sampling kernel); no model."""
import math, random, types
import torch
from exllamav3.generator import spec_sample as ss
from exllamav3.generator.sampler import (CustomSampler, ComboSampler, SS_RepP, SS_PresFreqP, SS_Temperature,
                                         SS_TopK, SS_TopP, SS_MinP, SS_Sample)
from exllamav3.generator.sampler.custom import SS_Fused

DEV = "cuda"


def _chi2_z(obs, exp):
    """Wilson-Hilferty z of the chi-square statistic over cells with expected count >= 5
    (smaller cells pooled)"""
    big = exp >= 5
    o = torch.cat([obs[big], obs[~big].sum().view(1)]) if (~big).any() else obs[big]
    e = torch.cat([exp[big], exp[~big].sum().view(1)]) if (~big).any() else exp[big]
    keep = e > 0
    o, e = o[keep].double(), e[keep].double()
    chi2 = float(((o - e) ** 2 / e).sum())
    k = int(o.numel()) - 1
    return ((chi2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k)), chi2, k


# --- the ratio test itself ---

def test_accept_bruteforce():
    g = torch.Generator().manual_seed(1)
    B, w, V = 3000, 3, 5
    p = torch.rand(B, w + 1, V, generator=g); p /= p.sum(-1, keepdim=True)
    q = torch.rand(B, w, V, generator=g) * (torch.rand(B, w, V, generator=g) < 0.7); q[..., 0] += 0.01
    q /= q.sum(-1, keepdim=True)
    d = torch.multinomial(q.view(-1, V), 1, generator=g).view(B, w)
    u = torch.rand(B, w, generator=g)
    resid = torch.randint(0, V, (B, w), generator=g)
    bonus = torch.randint(0, V, (B,), generator=g)
    tok, n = ss.accept(p, q, d, u, resid, bonus)
    for b in range(B):
        out = []
        for i in range(w):
            if u[b, i] * q[b, i, d[b, i]] < p[b, i, d[b, i]]:
                out.append(int(d[b, i]))
            else:
                out.append(int(resid[b, i]))
                break
        else:
            out.append(int(bonus[b]))
        assert int(n[b]) == len(out) - 1, (b, out, n[b])
        assert tok[b, :len(out)].tolist() == out, (b, out, tok[b])


def test_residual_and_dense_q():
    V = 7
    p = torch.tensor([[[0.5, 0.2, 0.3, 0, 0, 0, 0]]], device=DEV)
    q = torch.tensor([[[0.1, 0.6, 0.3, 0, 0, 0, 0]]], device=DEV)
    g = torch.Generator(device=DEV).manual_seed(3)
    r = torch.cat([ss.residual_samples(p, q, g) for _ in range(2000)])
    assert set(r.view(-1).tolist()) == {0}, "residual max(0, p - q) is only token 0 here"
    # p == q: no rejection is possible; the fallback samples p
    r = torch.cat([ss.residual_samples(p, p, g) for _ in range(500)]).view(-1).tolist()
    assert set(r) <= {0, 1, 2}
    d = torch.tensor([4, 2], device=DEV)
    dq = ss.dense_q([(torch.tensor([2, 5], device=DEV), torch.tensor([0.25, 0.75], device=DEV))], d, V)
    assert dq[0].tolist() == [0, 0, 0.25, 0, 0, 0.75, 0] and dq[1].tolist() == [0, 0, 1, 0, 0, 0, 0], dq


# --- p is the fused kernel's distribution ---

def _eager_keep(x, inv_temp_filter, top_k, top_p):
    """Tie-aware eager kept set: temperature, top-k (ties at the k-th logit kept), top-p over the
    top-k set (the token crossing the cumulative sum dropped, with its ties; the top always kept)"""
    R, V = x.shape
    keep = torch.ones_like(x, dtype=torch.bool)
    for r in range(R):
        xs, order = torch.sort(x[r], descending=True)
        n = V
        if top_k:
            c = xs[top_k - 1]
            n = int((xs >= c).sum())
        if top_p < 1.0:
            pr = torch.softmax(xs[:n].double() * inv_temp_filter, 0)
            cum = torch.cumsum(pr, 0)
            cross = int((cum > top_p).nonzero()[0]) if (cum > top_p).any() else n
            if cross < n:
                c = xs[cross]
                n2 = int((xs[:n] > c).sum())
                n = max(n2, int((xs == xs[0]).sum()))
        keep[r] = False
        keep[r, order[:n]] = True
    return keep


def _logits(R, V, seed):
    g = torch.Generator().manual_seed(seed)
    # fp16 logits in [2, 7.9): distinct values are > 1/32768 nat apart (the kernel's bin width)
    return (2 + 5.9 * torch.rand(R, V, generator=g)).half().to(DEV)


def test_fused_p_kept_set_matches_eager():
    for T, k, tp in ((1.0, 20, 0.95), (0.7, 20, 0.9), (1.3, 0, 0.8), (1.0, 50, 1.0), (0.6, 0, 1.0)):
        step = ComboSampler(temperature=T, top_k=k, top_p=tp).steps[-1]
        assert isinstance(step, SS_Fused), step
        L = _logits(64, 600, int(T * 10) + k)
        _, p = ss.fused_p(L, step, 600, 12345)
        ek = _eager_keep(L.float().cpu(), step.inv_temp_filter, k, tp)
        assert torch.equal(p.cpu() > 0, ek), (T, k, tp, int(((p.cpu() > 0) != ek).sum()))
        assert torch.allclose(p.sum(-1), torch.ones(64, device=DEV), atol=1e-5)


def test_fused_p_matches_kernel_samples():
    step = ComboSampler(temperature=1.0, top_k=20, top_p=0.95).steps[-1]
    V, R, N = 300, 4, 50000
    L = _logits(R, V, 7) * 0.6
    _, p = ss.fused_p(L, step, V, 1)
    # the kernel's own samples: every row repeated, in chunks (one random stream per call)
    counts = torch.zeros(R, V, device=DEV)
    rows = L.repeat_interleave(2000, dim=0)
    for c in range(N // 2000):
        s, _ = ss.fused_p(rows, step, V, 1000 + c)
        counts.view(-1).index_add_(0, (torch.arange(rows.shape[0], device=DEV) // 2000) * V + s,
                                   torch.ones_like(s, dtype=torch.float))
    assert (counts[p == 0] == 0).all(), "a sample outside p's support"
    for r in range(R):
        z, chi2, k = _chi2_z(counts[r].cpu(), (p[r] * N).cpu())
        assert z < 4.5, (r, z, chi2, k)


# --- the emitted token distribution over chains ---

def _chain_test(Vt, w, p_step, q_temp, q_topk, q_topp, point_mass_pos, B, seed):
    """Toy model over Vt tokens: logits depend on the whole prefix (tables for every prefix up
    to w + 2 tokens). Speculative rounds of w drafts until 3 tokens are emitted per trial; the
    joint distribution of the first 3 must be p(t0) p(t1|t0) p(t2|t0,t1)."""
    depth = w + 3
    sizes = [Vt ** n for n in range(depth)]
    offs = [sum(sizes[:n]) for n in range(depth)]
    total = sum(sizes)
    g = torch.Generator().manual_seed(seed)
    pl = (1.2 * torch.randn(total, Vt, generator=g) + 4).half().to(DEV)
    ql = pl.float() + 0.8 * torch.randn(total, Vt, generator=g).to(DEV)   # a different draft model
    P = torch.cat([ss.fused_p(pl[i:i + 4096], p_step, Vt, 99 + i)[1] for i in range(0, total, 4096)])
    qi, qp = ss.q_dist(ql, 1.0 / q_temp, q_topk, q_topp)
    Q = torch.zeros(total, Vt, device=DEV).scatter_(1, qi, qp)
    gt = torch.Generator(device=DEV).manual_seed(seed)

    def code(pref, n):   # pref (B, n) tokens -> table row
        c = torch.zeros(pref.shape[0], dtype=torch.long, device=DEV)
        for i in range(n):
            c = c * Vt + pref[:, i]
        return c + offs[n]

    emitted = torch.full((B, 3 + w + 1), -1, dtype=torch.long, device=DEV)
    count = torch.zeros(B, dtype=torch.long, device=DEV)
    while True:
        act = (count < 3).nonzero().view(-1)
        if act.numel() == 0:
            break
        for n in range(3):   # group by prefix length
            idx = act[count[act] == n]
            if idx.numel() == 0:
                continue
            pref = emitted[idx, :n]
            ds, ps, qs = [], [], []
            cur = pref
            for i in range(w):
                c = code(cur, n + i)
                ps.append(P[c])
                if i == point_mass_pos:   # a lookup-style deterministic draft
                    di = (cur.sum(1) * 3 + 1) % Vt if n + i > 0 else torch.ones_like(c)
                    qrow = torch.zeros(idx.numel(), Vt, device=DEV).scatter_(1, di.view(-1, 1), 1.0)
                else:
                    qrow = Q[c]
                    di = torch.multinomial(qrow, 1, generator=gt).view(-1)
                qs.append(qrow)
                ds.append(di)
                cur = torch.cat([cur, di.view(-1, 1)], 1)
            ps.append(P[code(cur, n + w)])
            p = torch.stack(ps, 1); q = torch.stack(qs, 1); d = torch.stack(ds, 1)
            u = torch.rand(d.shape, generator=gt, device=DEV)
            resid = ss.residual_samples(p[:, :w], q, gt)
            bonus = torch.multinomial(p[:, w], 1, generator=gt).view(-1)
            tok, na = ss.accept(p, q, d, u, resid, bonus)
            for j in range(w + 1):
                sel = na >= j
                emitted[idx[sel], n + j] = tok[sel, j]
            count[idx] += na + 1
    t = emitted[:, :3]
    cells = t[:, 0] * Vt * Vt + t[:, 1] * Vt + t[:, 2]
    obs = torch.bincount(cells, minlength=Vt ** 3).float().cpu()
    p0 = P[offs[0]]
    exp = torch.zeros(Vt ** 3)
    for a in range(Vt):
        pa = P[offs[1] + a]
        for b in range(Vt):
            pb = P[offs[2] + a * Vt + b]
            for c in range(Vt):
                exp[a * Vt * Vt + b * Vt + c] = float(p0[a] * pa[b] * pb[c]) * B
    z, chi2, k = _chi2_z(obs, exp)
    assert k >= 20, f"too few populated cells for a meaningful test: {k + 1}"
    return z, chi2, k


def test_chain_distribution_plain_sampling():
    step = ComboSampler(temperature=1.0).steps[-1]
    z, chi2, k = _chain_test(6, 3, step, q_temp=1.3, q_topk=4, q_topp=1.0, point_mass_pos=None, B=300000, seed=5)
    assert z < 4.5, (z, chi2, k)


def test_chain_distribution_topk_topp():
    step = ComboSampler(temperature=1.2, top_k=4, top_p=0.95).steps[-1]
    z, chi2, k = _chain_test(6, 3, step, q_temp=1.0, q_topk=3, q_topp=0.95, point_mass_pos=None, B=300000, seed=6)
    assert z < 4.5, (z, chi2, k)


def test_chain_distribution_lookup_position():
    step = ComboSampler(temperature=1.0, top_k=5, top_p=0.95).steps[-1]
    z, chi2, k = _chain_test(6, 3, step, q_temp=1.0, q_topk=0, q_topp=1.0, point_mass_pos=1, B=300000, seed=7)
    assert z < 4.5, (z, chi2, k)


def test_chain_test_detects_the_wrong_rule():
    """The same harness rejects a wrong acceptance (accept a sampled draft with p(d), no ratio)"""
    real = ss.accept
    def wrong(p, q, d, u, resid, bonus):
        return real(p, torch.ones_like(q), d, u, resid, bonus)
    ss.accept = wrong
    try:
        step = ComboSampler(temperature=1.0).steps[-1]
        z, _, _ = _chain_test(6, 3, step, q_temp=1.3, q_topk=4, q_topp=1.0, point_mass_pos=None, B=300000, seed=5)
    finally:
        ss.accept = real
    assert z > 10, z


# --- scope: greedy and ineligible requests keep the fork's path ---

def _job(sampler, **kw):
    j = types.SimpleNamespace(sampler=sampler, filters=[], forced_ids=None, return_probs=False, return_top_tokens=0,
                              return_logits=False, sequences=[0], new_tokens=3, serial_number=1,
                              rng=random.Random(0), device_logit_mask=None)
    j.__dict__.update(kw)
    return j


def test_scope():
    tabby_like = CustomSampler([SS_RepP(1.0, int(10e7), 0), SS_PresFreqP(0.0, 0.0, int(10e7), 0), SS_Temperature(1.0),
                                SS_TopK(20), SS_TopP(0.95), SS_Sample()])
    assert isinstance(ss.job_step(_job(tabby_like)), SS_Fused)
    assert ss.job_step(_job(ComboSampler(temperature=0.0))) is None                   # greedy
    assert ss.job_step(_job(ComboSampler(temperature=1.0, rep_p=1.1))) is None        # active penalty
    assert ss.job_step(_job(ComboSampler(temperature=1.0, min_p=0.05))) is None       # min-p
    assert ss.job_step(_job(tabby_like, filters=[object()])) is None                  # grammar filter
    assert ss.job_step(_job(tabby_like, return_probs=True)) is None


def test_greedy_rows_keep_argmax():
    y = torch.randn(2, 64, device=DEV)
    ids = torch.argmax(y, dim=-1)
    job = _job(ComboSampler(temperature=1.0, top_k=20))
    params = {"spec_cfg": ([None, (1.0, 20, 1.0, job)], [None, job])}
    out = torch.stack([ss.draft_sample(y, ids, params) for _ in range(200)])
    assert (out[:, 0] == ids[0]).all(), "a greedy row's draft must stay the argmax"
    assert params["spec_q"][0] is None and params["spec_q"][1] is not None
    top20 = set(torch.topk(y[1], 20).indices.tolist())
    assert set(out[:, 1].tolist()) <= top20 and len(set(out[:, 1].tolist())) > 1


def test_draft_rows_and_trial_arms():
    tabby_like = ComboSampler(temperature=1.0, top_k=20, top_p=0.95)
    greedy = _job(ComboSampler(temperature=0.0), serial_number=2)
    samp = _job(tabby_like, serial_number=3)
    old = (ss.SPEC, ss.TRIAL_AB)
    try:
        ss.SPEC, ss.TRIAL_AB = True, False
        cfgs, jobs = ss.draft_rows([greedy, samp])
        assert cfgs[0] is None and cfgs[1][:3] == (1.0, 20, 0.95) and samp._spec_q == [] and greedy._spec_q is None
        assert greedy._trial_arm == "-"
        ss.SPEC, ss.TRIAL_AB = True, True
        a, b = _job(tabby_like, serial_number=4), _job(tabby_like, serial_number=5)
        cfgs, _ = ss.draft_rows([a, b])
        assert (a._trial_arm, b._trial_arm) == ("A", "B") and cfgs[0] is None and cfgs[1] is not None
        ss.SPEC = False   # A/A: arms logged, nothing sampled
        assert ss.draft_rows([_job(tabby_like, serial_number=7)]) is None
    finally:
        ss.SPEC, ss.TRIAL_AB = old


def test_verify_window():
    V, w = 50, 4
    step = ComboSampler(temperature=1.0, top_k=20, top_p=0.95).steps[-1]
    L = _logits(w + 1, V, 11).view(1, w + 1, V)
    d = torch.tensor([3, 7, 1, 9])
    qs = [ss.q_dist(L[0, i].float(), 1.0, 20, 0.95) for i in range(w - 1)]   # last position: point mass
    gen = types.SimpleNamespace(tokenizer=types.SimpleNamespace(actual_vocab_size=V))
    for trial in range(50):
        job = _job(ComboSampler(temperature=1.0, top_k=20, top_p=0.95), rng=random.Random(trial))
        job._spec_q, job._spec_step = list(qs), step
        tok, match = ss.verify(gen, job, L, d)
        assert job._spec_q is None and tok.shape == (w + 1,) and len(match) == w
        n = match.index(0) if 0 in match else w
        assert all(m == 1 for m in match[:n]) and tok[:n].tolist() == d[:n].tolist(), (match, tok)
        # a rejected position emits a residual sample: never the drafted token
        if n < w:
            assert int(tok[n]) != int(d[n]), (n, tok, d)
