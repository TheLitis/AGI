from trainer import Trainer


def _make_trainer_stub(target: float, warmup: int, updates: int) -> Trainer:
    tr = Trainer.__new__(Trainer)
    tr.action_mask_dropout_target = float(target)
    tr.action_mask_dropout_warmup_updates = int(warmup)
    tr.policy_update_count = int(updates)
    return tr


def test_mask_dropout_curriculum_warmup_progresses_linearly():
    tr = _make_trainer_stub(target=0.2, warmup=4, updates=0)
    assert tr._current_action_mask_dropout_prob() == 0.0

    tr.policy_update_count = 1
    assert abs(tr._current_action_mask_dropout_prob() - 0.05) < 1e-9

    tr.policy_update_count = 2
    assert abs(tr._current_action_mask_dropout_prob() - 0.10) < 1e-9

    tr.policy_update_count = 4
    assert abs(tr._current_action_mask_dropout_prob() - 0.20) < 1e-9

    tr.policy_update_count = 99
    assert abs(tr._current_action_mask_dropout_prob() - 0.20) < 1e-9


def test_mask_dropout_curriculum_disabled_returns_target():
    tr = _make_trainer_stub(target=0.15, warmup=0, updates=0)
    assert abs(tr._current_action_mask_dropout_prob() - 0.15) < 1e-9
