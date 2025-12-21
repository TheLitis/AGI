import numpy as np

from memory import ReplayBuffer, Transition


def _make_tr(regime: str, val: int) -> Transition:
    return Transition(
        obs_patch=np.zeros((2, 2), dtype=np.int64) + val,
        energy=1.0,
        action=0,
        reward=float(val),
        done=False,
        next_obs_patch=np.zeros((2, 2), dtype=np.int64),
        next_energy=1.0,
        death_flag=0.0,
        got_food=0.0,
        took_damage=0.0,
        moved=0.0,
        alive=1.0,
        scenario_id=0,
        env_id=0,
        regime_name=regime,
    )


def test_sample_mixed_uses_both_regimes():
    buf = ReplayBuffer(capacity=10)
    for i in range(4):
        buf.push(_make_tr("current", val=1))
    for i in range(4):
        buf.push(_make_tr("past", val=2))

    batch = buf.sample_mixed(batch_size=4, seq_len=1, mix_config={"current_regime": "current", "frac_current": 0.5})
    rewards = batch[3].reshape(-1)  # r_seq
    assert any(r == 1.0 for r in rewards)
    assert any(r == 2.0 for r in rewards)


def test_sample_by_regime_filters():
    buf = ReplayBuffer(capacity=5)
    buf.push(_make_tr("A", val=3))
    buf.push(_make_tr("A", val=3))
    buf.push(_make_tr("B", val=4))
    batch = buf.sample_by_regime("A", batch_size=2, seq_len=1, with_events=False)
    rewards = batch[3].reshape(-1)
    assert all(r == 3.0 for r in rewards)


def test_sample_mixed_always_returns_full_batch():
    buf = ReplayBuffer(capacity=50)
    for _ in range(30):
        buf.push(_make_tr("current", val=1))

    batch_size = 8
    seq_len = 2
    batch = buf.sample_mixed(
        batch_size=batch_size,
        seq_len=seq_len,
        mix_config={"current_regime": "current", "frac_current": 0.5},
        with_events=True,
    )
    assert batch[0].shape[0] == batch_size  # obs_seq
    assert batch[3].shape[0] == batch_size  # r_seq
