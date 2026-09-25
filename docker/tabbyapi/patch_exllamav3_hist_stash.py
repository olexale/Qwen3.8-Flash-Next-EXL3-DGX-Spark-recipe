"""Recurrent checkpoints at every page boundary of the output, from the rollback history
(EXL3_HIST_STASH=1).

A follow-up turn resumes from the longest cached page prefix that also has a stashed
recurrent state. During generation the fork stashes one only every 2,048 tokens (and cuts
the verify round short at that boundary to land on it exactly), so a pi turn, whose prompt
is the previous prompt + the previous answer + a tool result, resumes at the previous
prompt's last page and prefills the whole previous answer again: in a recorded pi session
~330 tokens per turn (~0.25 s of a ~1.2 s time to first token), 11k of 32k prefilled tokens.

With speculative decoding the recurrent layers already keep per-token history for rolling
back rejected drafts: after a verify forward over T rows, GatedDeltaNet slot k (1 <= k < T)
holds the state after k rows, slot 0 the state after all T, and the conv windows sit
right-aligned in the trailing columns (PLE layers: the same convention). So when a round's
kept rows cross a page boundary b, the state at b is copied out of the history just before
the round's rewind (rejection rewind or the rewind(0) after a fully accepted draft; the
rewind overwrites the live conv window, which can overlap the history columns), and added
to the recurrent cache under the page's hash once the hash is checked against the page's
tokens. The copy is exactly what the fork's own boundary truncation would have stashed (the
same forward, rewound to b), so nothing changes in what the model computes; the round is no
longer cut short at the 2,048-token boundaries either (the history provides that state).

Per job only the latest of these checkpoints is kept, plus the ones on the usual 2,048-token
grid, so a long answer does not push other sessions' checkpoints out of the LRU cache. A
boundary crossed in the round that ends the job (EOS) is not stashed. Tensor-parallel
loading and recurrent layer types other than GatedDeltaNet / PLE: off (logged once).

Env: EXL3_HIST_STASH=0|1 (default 0 in exllamav3; the image sets 1)

Usage: python3 patch_exllamav3_hist_stash.py [exllamav3 package dir]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

HELPERS = '''

# --- recurrent checkpoints from the rollback history (patch_exllamav3_hist_stash.py) ---
import os as _hs_os
_HS_ON = _hs_os.environ.get("EXL3_HIST_STASH", "0") == "1"
_hs_metrics = {"captured": 0, "committed": 0, "dropped_superseded": 0, "hash_mismatch": 0, "unsupported": 0}
_hs_warned = set()

class _HSStashed:
    """Adapter for RecurrentCache.put, which calls state.stash()."""
    def __init__(self, stashed):
        self.stashed = stashed
    def stash(self):
        return self.stashed

def _hs_layer(l, slot, T, k):
    n = T - k
    name = type(l).__name__
    if name == "GDNLayerState":
        cd = l.module.conv_kernel_size
        ri = k if k < T else 0
        p = l.conv_state.shape[-1] - n
        return (l.recurrent_state[slot, ri : ri + 1].clone(), l.conv_state[slot, :, p - cd : p].clone())
    if name == "PLELayerState":
        p = l.conv_state.shape[-1] - n
        q = l.id_state.shape[-1] - n
        return (l.conv_state[slot, :, p - l.win : p].clone(), l.id_state[slot, q - l.ctx : q].clone())
    if name not in _hs_warned:
        _hs_warned.add(name)
        print(f"[hist-stash] recurrent layer type {name} not supported, no history checkpoints", flush = True)
    return None

def _hs_capture(gen, job, state, num_rejected):
    """Called just before a verify round's rewind: stash the state at a page boundary inside
    the kept rows, taken from the per-token history."""
    if not _HS_ON or state is None or gen.recurrent_cache is None or gen.model.loaded_tp:
        return
    if state.last_history <= 0 or len(job.sequences) != 1:
        return
    T = state.last_history + 1
    p0 = state.position - T
    keep = T - num_rejected
    b = ((p0 + keep) // PAGE_SIZE) * PAGE_SIZE
    if b <= p0:
        return
    k = b - p0
    stashed = {"position": b, "checkpoint_size": state.checkpoint_size}
    for key, l in state.cache.get_all_recurrent_layers().items():
        s = _hs_layer(l, state.slot, T, k)
        if s is None:
            _hs_metrics["unsupported"] += 1
            return
        stashed[key] = s
    _hs_metrics["captured"] += 1
    job._hs_pending = (b, stashed)
    _hs_commit(gen, job)

def _hs_commit(gen, job):
    pend = getattr(job, "_hs_pending", None)
    if not pend:
        return
    b, stashed = pend
    seq = job.sequences[0]
    pi = b // PAGE_SIZE - 1
    if pi >= len(seq.allocated_pages) or len(seq.sequence_ids) < b:
        job._hs_pending = None
        return
    page = seq.allocated_pages[pi]
    if page.kv_position != PAGE_SIZE:
        return  # page not complete yet: retried at the next recurrent_checkpoint()
    job._hs_pending = None
    prev_hash = seq.allocated_pages[pi - 1].phash if pi > 0 else None
    h = tensor_hash_checksum(seq.sequence_ids.torch_slice(b - PAGE_SIZE, b), prev_hash)
    if h != page.phash:
        _hs_metrics["hash_mismatch"] += 1
        return
    rc = gen.recurrent_cache
    if h in rc:
        rc.move_to_end(h)
        return
    rc.put(h, _HSStashed(stashed))
    _hs_metrics["committed"] += 1
    last = getattr(job, "_hs_last", None)
    if last is not None:
        lk, lb, ld = last
        on_grid = (lb - job.cached_pages * PAGE_SIZE) % gen.recurrent_checkpoint_interval == 0
        if not on_grid and dict.get(rc, lk) is ld:
            popped = rc.pop(lk)
            _hs_note_freed(popped["checkpoint_size"])
            rc.update_total_size()
            _hs_metrics["dropped_superseded"] += 1
    job._hs_last = (h, b, dict.get(rc, h))

def _hs_note_freed(nbytes):
    try:
        from ..cache.recurrent import note_freed
        note_freed(nbytes)
    except Exception:
        pass
'''

edits = [
    # Before the rejection rewind
    ("            # Rewind recurrent states\n"
     "            if batch_states_ is not None:\n"
     "                batch_states_[j_].rewind(num_rejected)\n",
     "            # Rewind recurrent states\n"
     "            if batch_states_ is not None:\n"
     "                _hs_capture(self, job_, batch_states_[j_], num_rejected)\n"
     "                batch_states_[j_].rewind(num_rejected)\n"),
    # Before the rewind(0) after a fully accepted draft
    ("                if batch_states and draft_tokens is not None and rejected == 0:\n"
     "                    batch_states[j].rewind(0)\n",
     "                if batch_states and draft_tokens is not None and rejected == 0:\n"
     "                    _hs_capture(self, job, batch_states[j], 0)\n"
     "                    batch_states[j].rewind(0)\n"),
    # No round truncation at checkpoint boundaries: the history provides that state
    ("                        cp_boundary = batch_states is not None and job.is_checkpoint_boundary()\n",
     "                        cp_boundary = batch_states is not None and not _HS_ON and job.is_checkpoint_boundary()\n"),
    # Commit checkpoints whose page completed after the capture
    ("    def recurrent_checkpoint(self):\n"
     "        for job in self.active_jobs:\n"
     "            job.maybe_stash_recurrent(self.recurrent_cache)\n",
     "    def recurrent_checkpoint(self):\n"
     "        for job in self.active_jobs:\n"
     "            job.maybe_stash_recurrent(self.recurrent_cache)\n"
     "            if _HS_ON:\n"
     "                _hs_commit(self, job)\n"),
]
p = root / "generator/generator.py"
s = p.read_text()
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"patch_exllamav3_hist_stash: anchor not found exactly once in generator.py: {old[:70]!r}")
    s = s.replace(old, new)
if "tensor_hash_checksum" not in s.split("class Generator")[0]:
    s = s.replace("\nclass Generator", "\nfrom .pagetable import tensor_hash_checksum\n\nclass Generator", 1)
if "PAGE_SIZE" not in s.split("class Generator")[0]:
    sys.exit("patch_exllamav3_hist_stash: PAGE_SIZE not imported in generator.py")
s += HELPERS
p.write_text(s)
print("patched generator/generator.py")
