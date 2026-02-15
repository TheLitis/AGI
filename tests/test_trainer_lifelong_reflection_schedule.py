from trainer import Trainer


def _trainer_stub() -> Trainer:
    t = Trainer.__new__(Trainer)
    t.lifelong_reflect_early_step_boost = 2
    t.lifelong_reflect_early_step_size_scale = 1.2
    t.lifelong_reflect_early_lambda_prior_scale = 0.5
    t.lifelong_reflect_late_step_delta = -1
    t.lifelong_reflect_late_step_size_scale = 0.8
    t.lifelong_reflect_late_lambda_prior_scale = 1.5
    t.lifelong_reflect_safety_step_size_scale = 0.8
    t.lifelong_reflect_safety_lambda_prior_scale = 1.5
    return t


def test_lifelong_reflection_schedule_early_mid_late_phases():
    trainer = _trainer_stub()
    early = trainer._lifelong_reflection_schedule(
        episode_idx=0,
        episodes_per_chapter=9,
        base_steps=4,
        base_step_size=0.02,
        base_lambda_prior=0.01,
        high_safety_risk=False,
    )
    mid = trainer._lifelong_reflection_schedule(
        episode_idx=4,
        episodes_per_chapter=9,
        base_steps=4,
        base_step_size=0.02,
        base_lambda_prior=0.01,
        high_safety_risk=False,
    )
    late = trainer._lifelong_reflection_schedule(
        episode_idx=8,
        episodes_per_chapter=9,
        base_steps=4,
        base_step_size=0.02,
        base_lambda_prior=0.01,
        high_safety_risk=False,
    )

    assert early["phase"] == "early"
    assert early["steps"] == 6
    assert abs(early["step_size"] - 0.024) < 1.0e-9
    assert abs(early["lambda_prior"] - 0.005) < 1.0e-9

    assert mid["phase"] == "mid"
    assert mid["steps"] == 4
    assert abs(mid["step_size"] - 0.02) < 1.0e-9
    assert abs(mid["lambda_prior"] - 0.01) < 1.0e-9

    assert late["phase"] == "late"
    assert late["steps"] == 3
    assert abs(late["step_size"] - 0.016) < 1.0e-9
    assert abs(late["lambda_prior"] - 0.015) < 1.0e-9


def test_lifelong_reflection_schedule_applies_safety_and_clamps():
    trainer = _trainer_stub()
    cfg = trainer._lifelong_reflection_schedule(
        episode_idx=0,
        episodes_per_chapter=6,
        base_steps=20,
        base_step_size=0.2,
        base_lambda_prior=0.03,
        high_safety_risk=True,
    )

    assert cfg["phase"] == "early"
    assert cfg["high_safety_risk"] is True
    assert 1 <= cfg["steps"] <= 12
    assert 0.005 <= cfg["step_size"] <= 0.08
    assert 0.001 <= cfg["lambda_prior"] <= 0.02
