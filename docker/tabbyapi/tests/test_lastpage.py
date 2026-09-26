"""patch_exllamav3_lastpage.py: with MTP, a k*256+1-token prompt's last full page gets its
content hash when prefill completes it, so a follow-up (and, with level 2, the same prompt sent
again) finds its K/V and the checkpoint at k*256. Real Sequence / PageTable / RecurrentCache,
prefill simulated on the page table (no model)."""
from types import SimpleNamespace
import torch
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator import job as jm
from exllamav3.generator.pagetable import PageTable, Sequence, is_content_hash
from exllamav3.cache.recurrent import RecurrentCache

K = 3
HIDDEN = 8

class _State:
    def __init__(self, pos):
        self.pos = pos
    def stash(self):
        return {"position": self.pos, "checkpoint_size": 100}

def _env(level):
    jm._LP_LEVEL = level
    rc = RecurrentCache(SimpleNamespace(loaded_tp=False), max_size=10**6)
    gen = SimpleNamespace(mtp_draft=True, recurrent_cache=rc)
    pt = PageTable(gen, SimpleNamespace(max_num_tokens=64 * PAGE_SIZE))
    return pt, rc, gen

def _ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 50000, (1, n), generator=g)

def _job(pt, gen, ids):
    """Queue a job the way prepare_for_queue does with MTP (the cap under test), then allocate."""
    seq = Sequence(ids, ids.clone())
    seq.prepare(False, 16)
    job = SimpleNamespace(pagetable=pt, generator=gen, embeddings=[], prefix_token=None,
                          mtp_last_hidden=None)
    seq.max_cached_pages = jm._lp_mtp_max_cached(job, seq) if jm._LP_LEVEL >= 2 else \
        max(0, (len(seq.sequence_ids) - 2) // PAGE_SIZE)
    protected = set(seq.page_hashes[:seq.max_cached_pages])
    _, cached, _, stashed = seq.allocate_pages(pt, gen.recurrent_cache, protected)
    if jm._LP_LEVEL >= 2 and cached:
        jm._lp_restore_carry(job, seq, stashed, cached)
    return job, seq, cached

def _prefill(job, seq):
    """What Job.prefill does to the pages for [kv_position, len - 1): fill, chain, rehash (the
    patch), then the last-page checkpoint and the carry."""
    end = len(seq.sequence_ids) - 1
    for pi in range(seq.kv_position // PAGE_SIZE, (end + PAGE_SIZE - 1) // PAGE_SIZE + 1):
        if pi >= len(seq.allocated_pages):
            break
        page = seq.allocated_pages[pi]
        page.kv_position = min(max(end - pi * PAGE_SIZE, 0), PAGE_SIZE)
        page.prev_hash = None if pi == 0 else seq.allocated_pages[pi - 1].phash
        a, b = pi * PAGE_SIZE, min(pi * PAGE_SIZE + PAGE_SIZE, end)
        if b > a:
            page.sequence[:, : b - a].copy_(seq.sequence_ids.torch_slice(a, b))
        if jm._LP_LEVEL:
            jm._lp_rehash(job, seq, pi)
    seq.kv_position = end
    if end % PAGE_SIZE == 0 and end > 0:
        last = seq.allocated_pages[end // PAGE_SIZE - 1]
        job.generator.recurrent_cache.put(last.phash, _State(end))
        seq.mtp_carry_hidden = torch.full((1, 1, HIDDEN), 0.5)
        if jm._LP_LEVEL >= 2:
            jm._lp_save_carry(job, seq)

def _run(pt, gen, ids):
    job, seq, cached = _job(pt, gen, ids)
    if seq.kv_position < len(seq.sequence_ids) - 1:
        _prefill(job, seq)
    pt.deallocate_pages(seq.allocated_pages)
    return job, seq, cached

def _content_hashes(ids):
    s = Sequence(ids, ids.clone()); s.prepare(False, 0)
    return s.page_hashes

def test_bug_without_patch():
    """Level 0 reproduces the report: the checkpoint at k*256 sits under a placeholder hash."""
    pt, rc, gen = _env(0)
    p = _ids(K * PAGE_SIZE + 1)
    _, seq, _ = _run(pt, gen, p)
    assert not is_content_hash(seq.allocated_pages[K - 1].phash)
    assert _content_hashes(p)[K - 1] not in rc and len(rc) == 1
    _, _, cached = _run(pt, gen, torch.cat([p, _ids(300, 1)], -1))
    assert cached == 0

def test_follow_up_resumes_at_last_page():
    pt, rc, gen = _env(1)
    p = _ids(K * PAGE_SIZE + 1)
    _, seq, cached = _run(pt, gen, p)
    assert cached == 0
    h = _content_hashes(p)
    assert [pg.phash for pg in seq.allocated_pages[:K]] == h
    assert h[K - 1] in rc and rc[h[K - 1]]["position"] == K * PAGE_SIZE
    assert jm._lp_stats["rehashed"] >= 1
    # a follow-up that starts with the whole prompt (diverges right after it)
    _, seq2, cached = _run(pt, gen, torch.cat([p, _ids(300, 1)], -1))
    assert cached == K and seq2.allocated_pages[K - 1].prev_hash == h[K - 2]
    # level 1 leaves the exact re-send at the MTP cap
    _, _, cached = _run(pt, gen, p)
    assert cached == 0

def test_resend_uses_all_pages_with_carry():
    pt, rc, gen = _env(2)
    p = _ids(K * PAGE_SIZE + 1)
    _run(pt, gen, p)
    st = rc[_content_hashes(p)[K - 1]]
    assert "mtp_carry" in st and st["mtp_carry"][0].shape == (1, 1, HIDDEN)
    job, seq, cached = _run(pt, gen, p)
    assert cached == K and seq.kv_position == K * PAGE_SIZE == len(seq.sequence_ids) - 1
    assert torch.equal(job.mtp_last_hidden, torch.full((1, 1, HIDDEN), 0.5))
    assert seq.mtp_carry_hidden is job.mtp_last_hidden

def test_resend_without_carry_keeps_the_cap():
    """A checkpoint at k*256 without a carry (made by another job) does not lift the cap, since
    the job would have no MTP carry; the run that prefills the prompt then attaches its carry."""
    pt, rc, gen = _env(2)
    p = _ids(K * PAGE_SIZE + 1)
    _run(pt, gen, torch.cat([p, _ids(300, 1)], -1))  # longer prompt first: pages hashed normally
    h = _content_hashes(p)
    rc.put(h[K - 1], _State(K * PAGE_SIZE))
    job, seq, cached = _run(pt, gen, p)
    assert cached == 0 and job.mtp_last_hidden is None
    assert "mtp_carry" in rc[h[K - 1]]
    job, seq, cached = _run(pt, gen, p)
    assert cached == K and job.mtp_last_hidden is not None

def test_controls_unchanged():
    """k*256 and k*256+2 tokens: every full prompt page is hashed up front, nothing to fix."""
    for n in (K * PAGE_SIZE, K * PAGE_SIZE + 2):
        pt, rc, gen = _env(2)
        p = _ids(n, n)
        job = SimpleNamespace(pagetable=pt, generator=gen, embeddings=[], prefix_token=None)
        s = Sequence(p, p.clone()); s.prepare(False, 16)
        assert jm._lp_mtp_max_cached(job, s) == max(0, (n - 2) // PAGE_SIZE) == len(s.page_hashes)
        n0 = jm._lp_stats["rehashed"]
        _, seq, _ = _run(pt, gen, p)
        assert jm._lp_stats["rehashed"] == n0
        full = (n - 1) // PAGE_SIZE
        assert [pg.phash for pg in seq.allocated_pages[:full]] == _content_hashes(p)

def test_live_duplicate_page_stays_unique():
    """Another live job holds the same page: ours keeps its placeholder, the other is untouched."""
    pt, rc, gen = _env(1)
    p = _ids(K * PAGE_SIZE + 1)
    h = _content_hashes(p)
    other, oseq, _ = _job(pt, gen, torch.cat([p, _ids(300, 1)], -1))
    _prefill(other, oseq)                   # live, not deallocated
    assert h[K - 1] in pt.referenced_pages
    n0 = jm._lp_stats["shared_skip"]
    job, seq, _ = _job(pt, gen, p)
    # K-1 pages come from the live job (by hash), page K-1 is a fresh page
    assert seq.allocated_pages[K - 1] is not oseq.allocated_pages[K - 1]
    _prefill(job, seq)
    assert jm._lp_stats["shared_skip"] == n0 + 1
    assert not is_content_hash(seq.allocated_pages[K - 1].phash)
    assert pt.referenced_pages[h[K - 1]] is oseq.allocated_pages[K - 1]

def test_unreferenced_duplicate_is_replaced():
    pt, rc, gen = _env(1)
    p = _ids(K * PAGE_SIZE + 1)
    h = _content_hashes(p)
    _, oseq, _ = _run(pt, gen, torch.cat([p, _ids(300, 1)], -1))
    old = pt.unreferenced_pages[h[K - 1]]
    rc.clear()
    _, seq, _ = _run(pt, gen, p)
    new = pt.unreferenced_pages[h[K - 1]]
    assert new is seq.allocated_pages[K - 1] and new is not old
    assert not is_content_hash(old.phash) and old.kv_position == 0
