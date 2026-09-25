"""Prompt-lookup drafting alongside the MTP head (EXL3_PLD=1, off by default).

When the text being generated repeats text already in the context (pi's `edit` tool calls
quote `oldText` verbatim, file rewrites copy most of a file), the continuation of that
match is a better draft than the MTP head's <= 5 tokens: it can be long and it costs no
draft forwards. Per round, for a single active job: look up the last EXL3_PLD_NGRAM tokens
of the sequence (prompt + output) in an n-gram index of the earlier sequence; if the match
extends to at least EXL3_PLD_MIN_MATCH tokens, draft the tokens that followed it instead of
the rest of the MTP chain for that round, provided the MTP head's first draft token (step 0
of the chain, already on the host in the dynamic-drafting path) agrees with the lookup's
(EXL3_PLD_GATE=1, default; 0 skips the MTP chain entirely on a match). Draft length: EXL3_PLD_START of them, or up to EXL3_PLD_MAX right after a
lookup round whose drafts were all accepted (a copy in progress). Otherwise draft with MTP as before (the
device-resident chain, dynamic drafting and the calibrator are untouched).

Verification is unchanged: the target samples every position and a drafted token is kept
only if it equals that sample, so the output distribution is the same whatever the draft.
In a lookup round the MTP head's KV for the round's first position was not written by a
draft step, so the post-verify MTP prefill covers positions K..K+A-1 (with the carried
hidden state for K) instead of K+1..K+A-1.

The index is a dict of n-gram -> last few end positions, built once per job (numpy-free,
~20 ms per 100k prompt tokens) and extended by the new tokens each round; the lookup is
O(1) per round plus a bounded backward extension of the match.

The GatedDeltaNet layers keep per-token state history for rolling back rejected drafts,
sized by the cache's max_history; with EXL3_PLD=1 a Cache built with max_history > 0 gets
max(max_history, EXL3_PLD_MAX). That is (EXL3_PLD_MAX - 5) x ~113 MB more per batch slot
on this model. Drafts are also capped at the cache's max_history, at 15, and at the fused
decode MoE's row limit minus one (7 by default; 15 with patch_exllamav3_bszn16.py and
EXL3_MOE_BSZN_MAX=16): verify rows beyond it take the prefill-style MoE kernel, ~45 ms
more per round.

Env: EXL3_PLD=0|1  EXL3_PLD_GATE=1  EXL3_PLD_START=7  EXL3_PLD_MAX=15  EXL3_PLD_MIN_MATCH=8  EXL3_PLD_NGRAM=3
     EXL3_PLD_BATCH=1 (largest number of active jobs for which lookup is tried)

Usage: python3 patch_exllamav3_pld.py [exllamav3 package dir]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])

GEN = "generator/generator.py"
CACHE = "cache/cache.py"

EDITS = {
    GEN: [
        # flags
        ("""_MTP_DEVICE_DRAFT = _os.environ.get("EXL3_MTP_DEVICE_DRAFT", "1") != "0"
""", """_MTP_DEVICE_DRAFT = _os.environ.get("EXL3_MTP_DEVICE_DRAFT", "1") != "0"
# Prompt-lookup drafting alongside MTP (patch_exllamav3_pld.py)
_PLD = _os.environ.get("EXL3_PLD", "0") == "1"
_PLD_MAX = int(_os.environ.get("EXL3_PLD_MAX", "15"))
_PLD_START = int(_os.environ.get("EXL3_PLD_START", "7"))
_PLD_MIN_MATCH = int(_os.environ.get("EXL3_PLD_MIN_MATCH", "8"))
_PLD_NGRAM = int(_os.environ.get("EXL3_PLD_NGRAM", "3"))
_PLD_BATCH = int(_os.environ.get("EXL3_PLD_BATCH", "1"))
_PLD_GATE = _os.environ.get("EXL3_PLD_GATE", "1") != "0"
_PLD_KEEP = 4          # end positions kept per n-gram (most recent)
_PLD_EXTEND = 64       # backward match extension limit


def _pld_draft(job, max_len):
    \"\"\"
    Continuation of the longest recent earlier occurrence of the sequence's tail, or None.
    State lives on the job; a shrunk sequence (rewind) rebuilds it.
    \"\"\"
    seq = job.sequences[0]
    n = len(seq.sequence_ids)
    st = getattr(job, "_pld_state", None)
    if st is None or n < len(st[0]):
        st = job._pld_state = [seq.sequence_ids.torch().view(-1).tolist(), {}, 0]
    ids, index, done = st
    if n > len(ids):
        ids += seq.sequence_ids.torch_slice(len(ids), n).view(-1).tolist()
    ng = _PLD_NGRAM
    # index n-grams ending at positions done .. n-2 (the tail's own n-gram is the query)
    for p in range(max(done, ng - 1), n - 1):
        key = tuple(ids[p - ng + 1:p + 1])
        lst = index.get(key)
        if lst is None:
            index[key] = [p]
        else:
            lst.append(p)
            if len(lst) > _PLD_KEEP: del lst[0]
    st[2] = max(done, n - 1)
    if n < ng + 1:
        return None
    lst = index.get(tuple(ids[n - ng:n]))
    if not lst:
        return None
    best_p, best_m = -1, 0
    for p in reversed(lst):
        m = ng
        while m < _PLD_EXTEND and p - m >= 0 and ids[p - m] == ids[n - 1 - m]:
            m += 1
        if m > best_m:
            best_p, best_m = p, m
    if best_m < _PLD_MIN_MATCH:
        return None
    d = ids[best_p + 1:best_p + 1 + max_len]
    return d if d else None
"""),
        # wider draft buffer
        ("""            self.draft_ids_pinned = torch.empty(
                (max_batch_size, self.num_draft_tokens),""",
         """            self.draft_ids_pinned = torch.empty(
                (max_batch_size, max(self.num_draft_tokens, _PLD_MAX if _PLD else 0)),"""),
        # lookup round in the MTP draft function
        ("""        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)
        # Device-resident draft chain""",
         """        # Prompt-lookup round: a long enough match replaces the MTP chain for this round. With
        # EXL3_PLD_GATE (default) only if the MTP head's first draft token agrees with the
        # lookup's, checked after draft step 0; otherwise the chain simply continues
        self._pld_round = False
        pld_gate = None
        if _PLD and batch_size <= _PLD_BATCH:
            pld_gate = self._pld_candidates()
            if pld_gate is not None and not _PLD_GATE:
                return self._pld_commit(pld_gate, "lookup")
        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)
        # Device-resident draft chain"""),
        ("""            cache_seqlens += 1
            temp_hidden = batch_state
            draft_conf = params.get("draft_conf")
            if cal is not None and draft_conf is not None:
                c = draft_conf.float().cpu()""",
         """            cache_seqlens += 1
            temp_hidden = batch_state
            if pld_gate is not None:
                first = (dev_draft_ids[:, 0] if dev_draft else self.draft_ids_pinned[:batch_size, 0]).tolist()
                if all(d[0] == f for d, f in zip(pld_gate[0], first)):
                    # step 0 wrote position K into the MTP cache: the usual post-verify prefill
                    return self._pld_commit(pld_gate, "gated")
                pld_gate = None
            draft_conf = params.get("draft_conf")
            if cal is not None and draft_conf is not None:
                c = draft_conf.float().cpu()"""),
        ("""    # TODO: Refactor, share code with other draft fns
    def iterate_draftmodel_dflash_gen(self, results: list):""",
         """    def _pld_candidates(self):
        \"\"\"
        Lookup drafts for every active job as (drafts, jobs), or None if any job has no usable
        match (the batch must be rectangular; MTP drafts that round).
        \"\"\"
        # Verify rows beyond the fused decode MoE's row limit take a much slower path
        from ..modules import block_sparse_mlp as _bsm
        cap = min(_PLD_MAX, 15, _bsm.MAX_BSZN - 1, getattr(self.cache, "max_history", 0),
                  self.draft_ids_pinned.shape[1])
        drafts = []
        jobs = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if len(job.sequences) != 1: return None
            budget = min(job.max_new_tokens, job.max_rq_tokens) - max(job.new_tokens, 0) - 1
            # Adaptive length: EXL3_PLD_START tokens, the cap after a fully accepted lookup round
            n = min(getattr(job, "_pld_len", _PLD_START), cap, budget)
            d = _pld_draft(job, n) if n > 1 else None
            if d is None: return None
            drafts.append(d)
            jobs.append(job)
            job._pld_cap = cap
        return (drafts, jobs) if drafts else None


    def _pld_commit(self, cand, kind):
        \"\"\"
        Use lookup drafts this round. Rows shorter than the longest are padded with their own last
        token (padding is verified and rejected like any draft).
        \"\"\"
        drafts, jobs = cand
        for job, d in zip(jobs, drafts):
            job._pld_drafted = len(d)
        w = max(len(d) for d in drafts)
        for row, d in enumerate(drafts):
            d = d + [d[-1]] * (w - len(d))
            self.draft_ids_pinned[row, :w].copy_(torch.tensor(d, dtype = torch.long))
        self._pld_round = kind
        return self.draft_ids_pinned[:, :w]


    # TODO: Refactor, share code with other draft fns
    def iterate_draftmodel_dflash_gen(self, results: list):"""),
        # post-verify MTP prefill covers the round's first position in a lookup round
        ("""        if self.mtp_draft:
            target_hidden = p_export_states[-1]
            accepted_idx = 0""",
         """        if self.mtp_draft:
            target_hidden = p_export_states[-1]
            accepted_idx = 0
            pld_round = getattr(self, "_pld_round", False) if draft_tokens is not None else False
            self._pld_round = False"""),
        ("""                # Position K was drafted from the last target state already. Replace accepted
                # speculative positions K+1..K+A-1 with the corresponding target-state inputs.
                if accepted_length > 1:""",
         """                if pld_round:
                    full = accepted_length - 1 >= getattr(job, "_pld_drafted", 1 << 30)
                    job._pld_len = getattr(job, "_pld_cap", _PLD_MAX) if full else _PLD_START

                # Lookup round: no draft step wrote position K into the MTP cache, so prefill
                # K..K+A-1, pairing K with the carried target state
                if pld_round == "lookup" and job.mtp_last_hidden is not None:
                    self.draft_model.prefill(
                        batch_ids[a_idx:b_idx, 0:accepted_length],
                        {
                            "attn_mode": "flash_attn",
                            "block_table": block_index[a_idx:b_idx],
                            "cache": self.draft_cache,
                            "cache_seqlens": p_cache_seqlens[a_idx:b_idx],
                            "target_hidden": torch.cat((
                                job.mtp_last_hidden.to(target_hidden.device),
                                target_hidden[a_idx:b_idx, :accepted_length - 1, :]
                            ), dim = 1),
                        },
                    )

                # Position K was drafted from the last target state already. Replace accepted
                # speculative positions K+1..K+A-1 with the corresponding target-state inputs.
                elif accepted_length > 1:"""),
    ],
    CACHE: [
        ("""        self.max_history = max_history
""", """        # Prompt-lookup drafting (patch_exllamav3_pld.py) verifies longer drafts
        import os as _os
        if max_history > 0 and _os.environ.get("EXL3_PLD", "0") == "1":
            max_history = max(max_history, min(int(_os.environ.get("EXL3_PLD_MAX", "15")), 15))
        self.max_history = max_history
"""),
    ],
}

for rel, edits in EDITS.items():
    p = root / rel
    s = p.read_text()
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_pld: anchor found {s.count(old)}x (want 1) in {rel}:\n{old[:300]}")
        s = s.replace(old, new)
    p.write_text(s)
print("patch_exllamav3_pld: applied")
