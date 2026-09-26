"""patch_exllamav3_prefix_diag.py: the longest-common-prefix helper."""
import torch
from exllamav3.generator import job as jm

def test_lcp():
    a = torch.tensor([1, 2, 3, 4]); b = torch.tensor([1, 2, 9, 4, 5])
    assert jm._pd_lcp(a, b) == 2
    assert jm._pd_lcp(a, a) == 4
    assert jm._pd_lcp(a, torch.tensor([1, 2])) == 2
    assert jm._pd_lcp(a, torch.tensor([], dtype=torch.long)) == 0
    assert jm._pd_lcp(torch.tensor([7]), torch.tensor([8])) == 0


# EXL3_PREFIX_DIAG=2 helpers. Token ids: 1 <|im_start|>, 2 <|im_end|>, 3 <tool_response>
# (special); 10 system, 11 user, 12 assistant; 20 "\n", 21 " ", 30+ words
import contextlib, io, types

_ST = {"special": {1, 2, 3}, "im_start": 1, "tool_response": 3,
       "roles": {10: "system", 11: "user", 12: "assistant"}}

class _Tok:
    tokenizer = types.SimpleNamespace(decode=lambda t: {20: "\n", 21: " "}.get(t[0], "w"))

def _set_struct():
    jm._pd_struct = _ST

def _t(*x):
    return torch.tensor(x, dtype=torch.long)

def test_find():
    assert jm._pd_find(_t(5, 6, 7, 8), _t(7, 8)) == 2
    assert jm._pd_find(_t(5, 6, 7, 8), _t(8, 9)) == -1
    assert jm._pd_find(_t(5), _t(5, 6)) == -1
    assert jm._pd_find(_t(5, 6), _t()) == -1

def test_msgs_users():
    _set_struct()
    ids = _t(1, 10, 30, 2, 20, 1, 11, 31, 2, 20, 1, 12, 32, 2, 20, 1, 11, 20, 3, 33, 2, 1, 11, 34)
    msgs = jm._pd_msgs(ids, _ST)
    assert msgs == [(0, "system", False), (5, "user", False), (10, "assistant", False),
                    (15, "user", True), (21, "user", False)]
    assert jm._pd_users(msgs) == (5, 21)
    assert jm._pd_users([(0, "system", False)]) == (None, None)

def test_classes():
    _set_struct()
    ids = _t(1, 20, 21, 30)
    assert jm._pd_classes(_Tok, ids, -2, 10, _ST) == "SNWO"

def test_diverge_line_moved_tail():
    # prev: system, user(31 32 33), gen prompt; new: same system, user gets 2 tokens inserted
    # before its text, and an assistant turn follows
    _set_struct()
    prev = _t(1, 10, 30, 2, 20, 1, 11, 20, 31, 32, 33, 2, 20, 1, 12, 20)
    new = _t(1, 10, 30, 2, 20, 1, 11, 20, 40, 41, 31, 32, 33, 2, 20, 1, 12, 20, 50, 2, 20, 1, 12, 20)
    l = jm._pd_lcp(new, prev)
    assert l == 8
    line = jm._pd_diverge_line(_Tok, new, prev, prev.shape[0], l)
    assert "diverge at 8: msg #2 role user (tool response no) +3" in line, line
    assert "first user prev 5 new 5, last user prev 5 new 5" in line, line
    assert "prev tail 8 at new +2 (matched 8)" in line, line
    assert "classes prev SOOSNSON|OOOSNSON new SOOSNSON|OOOOOSNS" in line, line

def test_session_start():
    _set_struct()
    jm._pd_starts.clear()
    jm._pd_start_count = 0
    gen = types.SimpleNamespace(tokenizer=_Tok)
    job = types.SimpleNamespace(generator=gen)
    first = _t(1, 10, 30, 2, 20, 1, 11, 31, 2, 20, 1, 12, 20)
    follow = _t(1, 10, 30, 2, 20, 1, 11, 31, 2, 20, 1, 12, 32, 2, 20, 1, 11, 33, 2, 20, 1, 12, 20)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        jm._pd_session_start(job, types.SimpleNamespace(page_hashes=[b"a", b"b", b"c"]), first)
        jm._pd_session_start(job, types.SimpleNamespace(page_hashes=[b"a", b"x"]), follow)  # not a start
        jm._pd_session_start(job, types.SimpleNamespace(page_hashes=[b"a", b"b", b"z"]), first)
    lines = out.getvalue().splitlines()
    assert len(lines) == 2, lines
    assert "session start #1 prompt 13, first user 5 | no earlier session start" in lines[0], lines
    assert "session start #2 prompt 13, first user 5 | shared with #1 (" in lines[1], lines
    assert f") {2 * jm.PAGE_SIZE} tokens, first user there 5" in lines[1], lines
