import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="gymnasium not installed")
pytest.importorskip("minigrid", reason="minigrid not installed")

from minigrid_env import MiniGridEnvPool


def test_minigrid_multi_reset_step():
    scenarios = ["minigrid-empty", "minigrid-doorkey", "test:minigrid-lavacrossing"]
    pool = MiniGridEnvPool(seed=123, schedule_mode="iid", scenario_names=scenarios)

    obs = pool.reset()
    assert "patch" in obs and obs["patch"].shape == (5, 5)
    assert "scenario_id" in obs

    seen = set()
    for _ in range(6):
        obs = pool.reset()
        seen.add(int(obs.get("scenario_id", -1)))
    assert len(seen) >= 2, "Pool should cycle through multiple MiniGrid scenarios"

    obs2, reward, done, info = pool.step(pool.sample_random_action())
    assert "env_id" in info and "scenario_id" in info
    assert obs2["patch"].shape == (5, 5)
    assert isinstance(reward, float)


def test_minigrid_descriptor_enriched():
    pool = MiniGridEnvPool(seed=0, schedule_mode="iid", scenario_names=["minigrid-empty"])
    desc = pool.get_env_descriptor(env_id=0)
    assert isinstance(desc, np.ndarray)
    assert desc.shape[0] >= 9
    # env_family id normalized should stay within [0, 1]
    assert 0.0 <= desc[0] <= 1.0
