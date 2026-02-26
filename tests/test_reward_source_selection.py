import torch

from trainer import Trainer, _should_use_raw_env_reward


class _DummyRewardHost:
    def _combined_preference_weights(self):
        # [survive, food, danger, move]
        return torch.tensor([[1.0, 2.0, -3.0, 0.5]], dtype=torch.float32)


def test_should_use_raw_env_reward_for_instruction_family():
    info = {"env_family": "instruction", "reward_env": 1.25}
    assert _should_use_raw_env_reward(info) is True


def test_should_not_use_raw_env_reward_for_grid_event_world():
    info = {
        "env_family": "gridworld",
        "reward_env": 0.0,
        "events": {"got_food": 1.0, "took_damage": 0.0, "moved": 1.0, "alive": 1.0},
    }
    assert _should_use_raw_env_reward(info) is False


def test_compute_preference_reward_uses_trait_events_for_gridworld():
    dummy = _DummyRewardHost()
    info = {
        "env_family": "gridworld",
        "reward_env": 0.0,
        "events": {"alive": 1.0, "got_food": 1.0, "took_damage": 0.0, "moved": 1.0},
    }
    reward = Trainer.compute_preference_reward(dummy, info, reward_profile=None)
    # 1*1 + 2*1 + (-3)*0 + 0.5*1 = 3.5
    assert reward == 3.5


def test_compute_preference_reward_uses_raw_reward_for_instruction():
    dummy = _DummyRewardHost()
    info = {"env_family": "instruction", "reward_env": 0.75}
    reward = Trainer.compute_preference_reward(dummy, info, reward_profile=None)
    assert reward == 0.75

