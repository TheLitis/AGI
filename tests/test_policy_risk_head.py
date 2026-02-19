from __future__ import annotations

import torch

from models import Policy


def test_policy_forward_with_mask_and_risk_shapes():
    policy = Policy(g_dim=64, n_actions=6)
    x = torch.randn(4, 64)
    logits, mask_logits, risk_v, risk_c = policy.forward_with_mask_and_risk(x)
    assert tuple(logits.shape) == (4, 6)
    assert tuple(mask_logits.shape) == (4, 6)
    assert tuple(risk_v.shape) == (4, 6)
    assert tuple(risk_c.shape) == (4, 6)
    assert torch.isfinite(risk_v).all()
    assert torch.isfinite(risk_c).all()

