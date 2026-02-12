from trainer import Trainer


def _trainer_stub(conflict: float, uncertainty: float, horizon: int) -> Trainer:
    t = Trainer.__new__(Trainer)
    t.meta_conflict_ma = float(conflict)
    t.meta_uncertainty_ma = float(uncertainty)
    t.planning_horizon = int(horizon)
    t.gae_lambda_min = 0.90
    t.gae_lambda_max = 0.99
    t.gae_lambda_adapt_strength = 0.04
    t.gae_lambda_horizon_scale = 24.0
    return t


def test_adaptive_gae_lambda_increases_for_stable_long_horizon():
    trainer = _trainer_stub(conflict=0.0, uncertainty=0.0, horizon=24)
    lam = trainer.get_adaptive_gae_lambda(0.95)
    assert lam > 0.95
    assert lam <= 0.99


def test_adaptive_gae_lambda_decreases_under_high_conflict_and_uncertainty():
    trainer = _trainer_stub(conflict=2.0, uncertainty=2.0, horizon=24)
    lam = trainer.get_adaptive_gae_lambda(0.95)
    assert lam < 0.95
    assert lam >= 0.90


def test_adaptive_gae_lambda_horizon_scales_effect():
    short_h = _trainer_stub(conflict=0.0, uncertainty=0.0, horizon=6)
    long_h = _trainer_stub(conflict=0.0, uncertainty=0.0, horizon=24)
    lam_short = short_h.get_adaptive_gae_lambda(0.95)
    lam_long = long_h.get_adaptive_gae_lambda(0.95)
    assert lam_long > lam_short


def test_adaptive_gae_lambda_respects_bounds_for_out_of_range_base():
    trainer = _trainer_stub(conflict=0.0, uncertainty=0.0, horizon=24)
    assert trainer.get_adaptive_gae_lambda(1.5) <= 0.99
    assert trainer.get_adaptive_gae_lambda(-1.0) >= 0.90
