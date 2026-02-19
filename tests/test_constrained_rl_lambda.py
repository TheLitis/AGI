from __future__ import annotations

from trainer import Trainer


def test_update_lagrange_multiplier_increases_above_budget():
    lam = Trainer._update_lagrange_multiplier(0.1, observed_rate=0.3, budget=0.1, lr=0.5, cap=10.0)
    assert lam > 0.1


def test_update_lagrange_multiplier_decreases_below_budget_and_clamps():
    lam = Trainer._update_lagrange_multiplier(0.2, observed_rate=0.0, budget=0.5, lr=1.0, cap=10.0)
    assert lam == 0.0


def test_update_lagrange_multiplier_respects_cap():
    lam = Trainer._update_lagrange_multiplier(9.9, observed_rate=1.0, budget=0.0, lr=1.0, cap=10.0)
    assert lam == 10.0

