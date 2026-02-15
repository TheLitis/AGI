import torch

from trainer import Trainer


def _trainer_stub() -> Trainer:
    return Trainer.__new__(Trainer)


def test_blend_with_planner_reduces_alpha_under_uncertainty_and_conflict():
    trainer = _trainer_stub()
    policy_logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32)
    planner_logits = torch.tensor([[0.0, 4.0, -1.0]], dtype=torch.float32)

    _blended_low, info_low = trainer._blend_with_planner(
        policy_logits=policy_logits,
        planner_logits=planner_logits,
        base_planning_coef=0.6,
        uncertainty=None,
        r_self=None,
        v_pi=None,
    )
    _blended_high, info_high = trainer._blend_with_planner(
        policy_logits=policy_logits,
        planner_logits=planner_logits,
        base_planning_coef=0.6,
        uncertainty=torch.tensor([[3.0]], dtype=torch.float32),
        r_self=torch.tensor([[4.0]], dtype=torch.float32),
        v_pi=torch.tensor([[-4.0]], dtype=torch.float32),
    )

    assert info_low["planner_alpha"] > info_high["planner_alpha"]
    assert info_low["planner_alpha"] > 0.0


def test_blend_with_planner_falls_back_for_non_finite_planner_logits():
    trainer = _trainer_stub()
    policy_logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
    planner_logits = torch.tensor([[float("inf"), 0.0, -1.0]], dtype=torch.float32)

    blended, info = trainer._blend_with_planner(
        policy_logits=policy_logits,
        planner_logits=planner_logits,
        base_planning_coef=0.7,
    )

    assert torch.allclose(blended, policy_logits)
    assert info["planner_alpha"] == 0.0
    assert info["planner_override"] == 0.0


def test_blend_with_planner_reports_override_when_top_action_changes():
    trainer = _trainer_stub()
    policy_logits = torch.tensor([[4.0, 0.0, -1.0]], dtype=torch.float32)
    planner_logits = torch.tensor([[-1.0, 5.0, -2.0]], dtype=torch.float32)

    _blended, info = trainer._blend_with_planner(
        policy_logits=policy_logits,
        planner_logits=planner_logits,
        base_planning_coef=0.8,
    )

    assert info["planner_alpha"] > 0.0
    assert info["planner_override"] > 0.0


def test_planner_debug_summary_has_expected_keys():
    trainer = _trainer_stub()
    summary = trainer._planner_debug_summary(
        alpha_values=[0.1, 0.2, 0.3],
        js_values=[0.4, 0.6],
        margin_values=[1.0, 2.0, 3.0],
        override_values=[0.0, 1.0, 1.0],
    )
    assert set(summary.keys()) == {
        "planner_alpha_mean",
        "planner_alpha_p90",
        "planner_js_mean",
        "planner_margin_mean",
        "planner_override_rate",
    }
    assert summary["planner_alpha_mean"] > 0.0
    assert 0.0 <= summary["planner_override_rate"] <= 1.0
