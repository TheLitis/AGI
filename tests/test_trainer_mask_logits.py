import torch

from trainer import Trainer


def _trainer_with_mask_coef(coef: float) -> Trainer:
    t = Trainer.__new__(Trainer)
    t.action_mask_prediction_coef = float(coef)
    t.unmasked_mask_bias_mix = 0.2
    t.unmasked_mask_confidence_threshold = 0.9
    return t


def test_compose_policy_logits_hard_mask_takes_priority():
    trainer = _trainer_with_mask_coef(0.1)
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    mask_logits_pred = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    out = trainer._compose_policy_logits_with_masks(
        logits,
        mask,
        mask_logits_pred,
        apply_hard_mask=True,
    )
    assert torch.isfinite(out[0, 0])
    assert out[0, 1] < -1.0e8


def test_compose_policy_logits_without_oracle_mask_keeps_logits_when_prediction_uncertain():
    trainer = _trainer_with_mask_coef(0.1)
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mask_logits_pred = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    out = trainer._compose_policy_logits_with_masks(
        logits,
        mask=None,
        mask_logits_pred=mask_logits_pred,
        apply_hard_mask=True,
    )
    assert torch.allclose(out, logits)


def test_compose_policy_logits_without_oracle_mask_uses_confident_soft_bias():
    trainer = _trainer_with_mask_coef(0.1)
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mask_logits_pred = torch.tensor([[5.0, -5.0]], dtype=torch.float32)

    out = trainer._compose_policy_logits_with_masks(
        logits,
        mask=None,
        mask_logits_pred=mask_logits_pred,
        apply_hard_mask=True,
    )
    assert out[0, 0] > out[0, 1]
    # Soft mix keeps unmasked bias conservative.
    assert (out[0, 0] - out[0, 1]).item() < 2.0


def test_compose_policy_logits_uses_predicted_bias_with_oracle_mask():
    trainer = _trainer_with_mask_coef(0.1)
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    mask_logits_pred = torch.tensor([[5.0, -5.0]], dtype=torch.float32)

    out = trainer._compose_policy_logits_with_masks(
        logits,
        mask=mask,
        mask_logits_pred=mask_logits_pred,
        apply_hard_mask=False,
    )
    assert out[0, 0] > out[0, 1]
    assert (out[0, 0] - out[0, 1]).item() > 4.0


def test_compose_policy_logits_ignores_predicted_bias_when_disabled():
    trainer = _trainer_with_mask_coef(0.0)
    logits = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    mask_logits_pred = torch.tensor([[5.0, -5.0]], dtype=torch.float32)

    out = trainer._compose_policy_logits_with_masks(
        logits,
        mask=None,
        mask_logits_pred=mask_logits_pred,
        apply_hard_mask=True,
    )
    assert torch.allclose(out, logits)
