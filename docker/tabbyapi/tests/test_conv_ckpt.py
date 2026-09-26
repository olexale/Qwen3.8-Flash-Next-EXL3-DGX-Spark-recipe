"""patch_exllamav3_conv_ckpt.py: anchor positions and the separate anchor LRU."""
import types
import torch
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator import job as jm
from exllamav3.cache.recurrent import RecurrentCache

# Token ids: 1 <|im_start|>, 3 <tool_response>, 4 </tool_response>, 11 user, 12 assistant, 30+ words
_IDS = {"im_start": 1, "tool_response": 3, "tool_response_end": 4, "user": 11}

def _t(*x):
    return torch.tensor(x, dtype=torch.long)

def test_last_user_skips_tool_responses():
    ids = _t(1, 11, 30, 1, 12, 31, 1, 11, 20, 3, 32, 4)
    assert jm._cc_last_user(ids, _IDS) == 0
    ids = _t(1, 11, 30, 1, 12, 31, 1, 11, 20, 3, 32, 4, 1, 11, 33, 1, 12)
    assert jm._cc_last_user(ids, _IDS) == 12
    assert jm._cc_last_user(_t(30, 31), _IDS) is None

def test_last_user_skips_markers_quoted_in_tool_responses():
    # a tool response quoting a chat template: its <|im_start|>user is not a message start
    ids = _t(1, 11, 30, 1, 12, 31, 1, 11, 20, 3, 32, 1, 11, 33, 4, 1, 12)
    assert jm._cc_last_user(ids, _IDS) == 0
    # after the tool response closes, a real user message counts again
    ids = _t(1, 11, 30, 1, 12, 31, 1, 11, 20, 3, 1, 11, 4, 1, 11, 34, 1, 12)
    assert jm._cc_last_user(ids, _IDS) == 13

def test_shared_pages_ignores_prompts_it_extends():
    recent = [[b"a", b"b", b"c"], [b"a", b"x"]]
    # extends the first prompt entirely: not a different conversation; shares 1 page with the second
    assert jm._cc_shared_pages([b"a", b"b", b"c", b"d"], recent) == 1
    assert jm._cc_shared_pages([b"a", b"b", b"z"], recent) == 2
    assert jm._cc_shared_pages([b"q"], recent) == 0
    assert jm._cc_shared_pages([], recent) == 0

class _State:
    def __init__(self, size=100):
        self.size = size
    def stash(self):
        return {"checkpoint_size": self.size, "position": 0}

def _rc(max_size=250):
    return RecurrentCache(types.SimpleNamespace(loaded_tp=False), max_size=max_size)

def test_anchor_lru_is_separate_and_capped():
    rc = _rc()
    rc.put(b"m1", _State()); rc.put(b"m2", _State())
    for k in (b"c1", b"c2", b"c3"):
        rc.put_conv(k, _State(), 2)
    assert list(rc.conv_keys) == [b"c2", b"c3"] and b"c1" not in rc
    # anchor entries do not count against max_size ...
    assert rc.update_total_size() == 200
    # ... and the ordinary LRU evicts its own oldest entry, not the (older) anchors
    rc.put(b"m3", _State())
    assert b"m1" not in rc and b"m2" in rc and b"m3" in rc
    assert b"c2" in rc and b"c3" in rc

def test_same_key_twice_dedups():
    rc = _rc()
    rc.put_conv(b"k", _State(), 4)
    first = rc[b"k"]
    rc.put_conv(b"k", _State(), 4)  # a parallel session reaching the same anchor
    rc.put(b"k", _State())
    assert rc[b"k"] is first and len(rc) == 1 and len(rc.conv_keys) == 1

def test_clear_drops_anchor_keys():
    rc = _rc()
    rc.put_conv(b"k", _State(), 4); rc.put(b"m", _State())
    rc.clear()
    assert not rc and not rc.conv_keys

def test_prune_drops_anchor_keys_too():
    rc = _rc()
    rc.put_conv(b"k", _State(), 4)
    rc.pagetable = types.SimpleNamespace(is_resumable=lambda k: False)
    assert rc.prune_stranded() == 1
    assert b"k" not in rc and not rc.conv_keys

class _Seq:
    def __init__(self, ids, hashes):
        self.ids = torch.tensor([ids], dtype=torch.long)
        self.page_hashes = hashes
    @property
    def sequence_ids(self):
        return _Ids(self.ids)

class _Ids:
    def __init__(self, t):
        self.t = t
    def torch(self):
        return self.t
    def __len__(self):
        return self.t.shape[-1]

def _job(rc):
    gen = types.SimpleNamespace(recurrent_cache=rc, tokenizer=None)
    return types.SimpleNamespace(generator=gen)

def test_anchors_and_pick():
    old = (jm._CC_LEVEL, jm._CC_MIN, jm._cc_ids)
    jm._CC_LEVEL, jm._CC_MIN, jm._cc_ids = 2, 2 * PAGE_SIZE, _IDS
    jm._cc_recent.clear()
    try:
        rc = _rc()
        n = 5 * PAGE_SIZE + 40
        # latest user message starts at 4 pages + 10 -> C2a anchor at 4 pages
        ids = [30] * n
        ids[4 * PAGE_SIZE + 10] = 1; ids[4 * PAGE_SIZE + 11] = 11
        h1 = [b"p0", b"p1", b"p2", b"p3", b"p4"]
        j1 = _job(rc)
        assert jm._cc_anchors(j1, _Seq(ids, h1)) == [(4 * PAGE_SIZE, "a")]
        # a different prompt sharing 3 pages with the first -> C2b anchor at 3 pages
        h2 = [b"p0", b"p1", b"p2", b"q3", b"q4"]
        j2 = _job(rc)
        a2 = jm._cc_anchors(j2, _Seq(ids, h2))
        assert a2 == [(3 * PAGE_SIZE, "b"), (4 * PAGE_SIZE, "a")], a2
        s2 = _Seq(ids, h2)
        assert jm._cc_pick(j2, s2, 0, 5 * PAGE_SIZE) == (3 * PAGE_SIZE, "b")
        assert jm._cc_pick(j2, s2, 3 * PAGE_SIZE, 5 * PAGE_SIZE) == (4 * PAGE_SIZE, "a")
        assert jm._cc_pick(j2, s2, 4 * PAGE_SIZE, 5 * PAGE_SIZE) is None
        # an anchor already stashed is not made again; below the minimum no C2b anchor
        rc.put(b"p3", _State())
        rc.put(b"q3", _State())
        j3 = _job(rc)
        assert jm._cc_anchors(j3, _Seq(ids, [b"p0", b"p1", b"p2", b"p3", b"r4"])) == []
        j4 = _job(rc)
        assert jm._cc_anchors(j4, _Seq([30] * n, [b"p0", b"s1", b"s2", b"s3", b"s4"])) == []
    finally:
        jm._CC_LEVEL, jm._CC_MIN, jm._cc_ids = old
        jm._cc_recent.clear()
