"""patch_exllamav3_hist_stash.py: the state copied from the rollback history at k rows into a
verify round equals what the layer's own rewind to that point + stash() gives, for every
round length and position, for both recurrent layer types this model has (CPU tensors)."""
from types import SimpleNamespace
import torch
from exllamav3.generator import generator as gm
from exllamav3.modules.gated_delta_net import GDNLayerState
from exllamav3.modules.ple import PLELayerState

MAX_HISTORY = 11  # EXL3_PLD_MAX in the image

def _gdn():
    m = SimpleNamespace(fdim_qkv=6, conv_kernel_size=4, num_v_heads=2, k_head_dim=3, v_head_dim=3)
    l = GDNLayerState(m, max_batch_size=3, max_history=MAX_HISTORY, cache_id=0)
    l.alloc("cpu")
    return l

def _ple():
    m = SimpleNamespace(conv_state_len=3, hc_mult=2, hidden_size=4,
                        ple_embedding=SimpleNamespace(context_len=2, eos_token_id=0))
    l = PLELayerState(m, max_batch_size=3, max_history=MAX_HISTORY, cache_id=0)
    l.alloc("cpu")
    return l

def _randomize(l, seed):
    g = torch.Generator().manual_seed(seed)
    for t in l.get_state_tensors():
        if t.dtype.is_floating_point:
            t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
        else:
            t.copy_(torch.randint(1, 1000, t.shape, generator=g))

def _check(make):
    slot = 1
    n = 0
    for T in range(2, MAX_HISTORY + 2):          # rows in the verify forward; last_history = T - 1
        for k in range(1, T + 1):                # rows kept up to the page boundary
            l = make(); _randomize(l, T * 100 + k)
            ours = gm._hs_layer(l, slot, T, k)
            l.rewind(slot, T - 1, T - k)
            ref = l.stash(slot)
            assert len(ours) == len(ref)
            for a, b in zip(ours, ref):
                assert a.shape == b.shape and torch.equal(a, b), (T, k, a.shape, b.shape)
            n += 1
    return n

def test_gdn_history_copy_equals_rewind_then_stash():
    assert _check(_gdn) > 50

def test_ple_history_copy_equals_rewind_then_stash():
    assert _check(_ple) > 50

def test_unknown_layer_type_is_refused():
    assert gm._hs_layer(SimpleNamespace(), 0, 3, 1) is None
