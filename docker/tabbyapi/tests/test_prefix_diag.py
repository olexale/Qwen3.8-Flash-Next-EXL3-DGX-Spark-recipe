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
