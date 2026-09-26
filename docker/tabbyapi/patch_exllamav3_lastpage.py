"""The last full page of a k*256+1-token prompt keeps its hash (EXL3_LASTPAGE).

With the MTP drafter, prepare_for_queue caps the cached prefix at (len - 2) // 256 pages so
that prefill always runs at least one prompt token (MTP takes its carry state, the target's
hidden state before the first generated position, from the last prefill row). For a prompt
of exactly k*256 + 1 tokens that leaves page k-1, the prompt's last full page, out of the
hashed pages: it is allocated as a fresh page with a placeholder (random) hash. Prefill fills
it and splits there for the last-page checkpoint (recurrent_last_page), and the checkpoint is
stored under the page's hash, but nothing gives the page its content hash afterwards (decode
only hashes pages it completes itself). So both the page's K/V and the checkpoint at k*256
can never be found again:

- a longer prompt that starts with this one (a follow-up) matches K/V up to page k-1 only if
  some other prompt left a hashed copy, and never finds the checkpoint at k*256: it resumes at
  the previous checkpoint, usually 0 (engine: 5,121 then 7,703 tokens, "kv-matched 7680,
  resumed 0" with a checkpoint at 5,120 in the cache);
- the same prompt sent again is capped at k-1 pages by the same rule, where no checkpoint
  exists, and prefills from 0 (real traffic: a 1,537-token prompt re-sent 5 times, each
  "kv-matched 1280, resumed 0, lost to checkpoints 1280").

Level 1: when prefill completes a page that still has a placeholder hash, the page gets its
content hash (the hash prepare() would have given it, chained on the previous page) before the
checkpoint is stored, as decode does for the pages it completes. If a live page of another job
already has that hash the page stays unique (as now); an unreferenced copy is cleared, as in
receive_sample. Bookkeeping only: the job itself computes exactly what it did before.

Level 2 adds the exact re-send: the last-page checkpoint of such a prompt also keeps the MTP
carry (one hidden-state row, [1, 1, hidden], on the host), and a prompt of k*256 + 1 tokens may
then use all k pages when that checkpoint is in the recurrent cache. The job starts decoding at
k*256 with the restored state, K/V and carry: the same inputs its first cold run's first
decode round had, so no prefill forward at all.

Env: EXL3_LASTPAGE=0|1|2 (default 0 in exllamav3, nothing changes when 0; the image sets 1,
approved 2026-09-26). Level 2 waits for the owner: in the engine check the re-send's first-token
logits were bit-identical to its cold run at 5,121 tokens but not at 1,537 (KL 1.9e-2, greedy
32 tokens identical), not yet explained; see PLAN_C_SESSION_START.md. Counters: job._lp_stats.

Usage: python3 patch_exllamav3_lastpage.py [exllamav3 package dir]
Run once at image build time, after patch_exllamav3_conv_ckpt.py; exits non-zero if an anchor
is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

JOB_HELPERS = '''

# --- the last full prompt page keeps its hash (patch_exllamav3_lastpage.py) ---
import os as _lp_os
from .pagetable import is_content_hash as _lp_is_content_hash
_LP_LEVEL = int(_lp_os.environ.get("EXL3_LASTPAGE", "0") or 0)
_lp_stats = {"rehashed": 0, "shared_skip": 0, "carry_saved": 0, "carry_resumed": 0}

def _lp_rehash(job, seq, pi):
    """Give a page that prefill just completed its content hash if it still has a placeholder."""
    page = seq.allocated_pages[pi]
    if page.kv_position != PAGE_SIZE or _lp_is_content_hash(page.phash):
        return
    pt = job.pagetable
    prev = seq.allocated_pages[pi - 1].phash if pi > 0 else None
    h = tensor_hash_checksum(seq.sequence_ids.torch_slice(pi * PAGE_SIZE, (pi + 1) * PAGE_SIZE), prev)
    if h in pt.referenced_pages:
        # A live job holds a page with the same content: keep ours unique (switching the block
        # table to that page mid-prefill would change which K/V this job reads)
        _lp_stats["shared_skip"] += 1
        return
    up = pt.unreferenced_pages.get(h)
    if up is not None:
        up.clear()
    page.update_hash(h)
    _lp_stats["rehashed"] += 1

def _lp_save_carry(job, seq):
    """Keep the MTP carry with the checkpoint at the end of a k*256+1-token prompt's prefill."""
    gen = job.generator
    pos = seq.kv_position
    if not gen.mtp_draft or seq.mtp_carry_hidden is None or job.embeddings:
        return
    if pos % PAGE_SIZE or pos != len(seq.sequence_ids) - 1:
        return
    page = seq.allocated_pages[pos // PAGE_SIZE - 1]
    if not _lp_is_content_hash(page.phash):
        return
    st = dict.get(gen.recurrent_cache, page.phash)
    if st is None or st.get("position") != pos or "mtp_carry" in st:
        return
    c = seq.mtp_carry_hidden
    st["mtp_carry"] = (c.to("cpu"), c.device)
    _lp_stats["carry_saved"] += 1

def _lp_mtp_max_cached(job, seq):
    """MTP cap on cached pages; with level 2 a k*256+1-token prompt may use all k pages when
    its last-page checkpoint carries the MTP state."""
    n = max(0, (len(seq.sequence_ids) - 2) // PAGE_SIZE)
    rc = job.generator.recurrent_cache
    if (
        _LP_LEVEL >= 2 and rc is not None and not job.embeddings and job.prefix_token is None and
        (len(seq.sequence_ids) - 1) % PAGE_SIZE == 0 and len(seq.page_hashes) == n + 1
    ):
        st = dict.get(rc, seq.page_hashes[-1])
        if st is not None and "mtp_carry" in st and st.get("position") == len(seq.sequence_ids) - 1:
            return n + 1
    return n

def _lp_restore_carry(job, seq, stashed, cached_pages):
    if stashed is None or not job.generator.mtp_draft:
        return
    if cached_pages * PAGE_SIZE != len(seq.sequence_ids) - 1:
        return
    c = stashed.get("mtp_carry")
    if c is None:
        return
    seq.mtp_carry_hidden = c[0].to(c[1])
    job.mtp_last_hidden = seq.mtp_carry_hidden
    _lp_stats["carry_resumed"] += 1
'''

edits = {
    "generator/job.py": [
        # Level 2: a k*256+1 prompt may use all k pages when the carry is stashed
        ("            if self.generator.mtp_draft:\n"
         "                seq.max_cached_pages = max(0, (len(seq.sequence_ids) - 2) // PAGE_SIZE)\n",
         "            if self.generator.mtp_draft:\n"
         "                seq.max_cached_pages = _lp_mtp_max_cached(self, seq) if _LP_LEVEL >= 2 else \\\n"
         "                    max(0, (len(seq.sequence_ids) - 2) // PAGE_SIZE)\n"),
        # Level 1: pages completed by prefill get their content hash, before the checkpoint is stored
        ("                    if pfp_b > pfp_a:\n"
         "                        page.sequence[:, pfp_a:pfp_b].copy_(seq.sequence_ids.torch_slice(pf_a, pf_b))\n"
         "                    page.can_revert = False\n",
         "                    if pfp_b > pfp_a:\n"
         "                        page.sequence[:, pfp_a:pfp_b].copy_(seq.sequence_ids.torch_slice(pf_a, pf_b))\n"
         "                    page.can_revert = False\n"
         "                    if _LP_LEVEL and not self.embeddings:\n"
         "                        _lp_rehash(self, seq, local_idx)\n"),
        ("                if recurrent_last_page:\n"
         "                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)\n",
         "                if recurrent_last_page:\n"
         "                    self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)\n"
         "                    if _LP_LEVEL >= 2:\n"
         "                        _lp_save_carry(self, seq)\n"),
        ("                    self.last_recurrent_checkpoint_pos = self.recurrent_state.position\n"
         "\n"
         "            if _CC_LEVEL:\n",
         "                    self.last_recurrent_checkpoint_pos = self.recurrent_state.position\n"
         "                    if _LP_LEVEL >= 2:\n"
         "                        _lp_restore_carry(self, seq, stashed_recurrent_state, cached_pages)\n"
         "\n"
         "            if _CC_LEVEL:\n"),
    ],
}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    if "patch_exllamav3_lastpage.py" in s:
        sys.exit(f"patch_exllamav3_lastpage: {rel} is already patched")
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_lastpage: anchor not found exactly once in {rel}:\n{old}")
        s = s.replace(old, new)
    s += JOB_HELPERS
    p.write_text(s)
    print("patched", rel)
