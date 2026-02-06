import torch

from models import Policy


def test_policy_forward_with_mask_shapes():
    policy = Policy(g_dim=64, n_actions=6)
    x = torch.randn(4, 64)
    logits = policy(x)
    logits2, mask_logits = policy.forward_with_mask(x)

    assert logits.shape == (4, 6)
    assert logits2.shape == (4, 6)
    assert mask_logits.shape == (4, 6)
