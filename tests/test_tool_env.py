import numpy as np

from env import EnvPool
from tool_env import ToolEnv, ToolEnvConfig


def make_tools_basic_env(seed: int = 0) -> EnvPool:
    cfg = ToolEnvConfig()
    train_env = ToolEnv(cfg, env_id=0, env_name="tools_basic_train", seed=seed)
    test_env = ToolEnv(cfg, env_id=1, env_name="tools_basic_test", seed=seed + 101)
    return EnvPool(
        envs=[train_env, test_env],
        schedule_mode="iid",
        seed=seed,
        train_env_ids=[0],
        test_env_ids=[1],
    )


def oracle_policy(env, obs):
    """
    Cheat-policy that directly inspects the env/observation to move memory toward target.
    """
    memory = obs.get("memory")
    target = obs.get("target")
    if memory is None or target is None:
        try:
            active_env = env.envs[env.active_env_idx]  # type: ignore[attr-defined]
        except Exception:
            active_env = env
        memory = getattr(active_env, "memory", memory)
        target = getattr(active_env, "target", target)
    if memory is None or target is None:
        return 0
    if memory < target:
        return 1  # INC
    if memory > target:
        return 2  # DEC
    return 0  # NOOP when aligned


def test_reset_sets_memory_and_target_range():
    cfg = ToolEnvConfig(target_min=0, target_max=5, max_steps=10)
    env = ToolEnv(cfg, env_id=0, seed=123)
    obs = env.reset()
    assert obs["memory"] == 0
    assert cfg.target_min <= obs["target"] <= cfg.target_max
    assert obs["steps_left"] == cfg.max_steps


def test_inc_and_dec_actions_update_memory():
    cfg = ToolEnvConfig(target_min=0, target_max=0, max_steps=5)
    env = ToolEnv(cfg, env_id=0)
    env.reset()
    obs, reward, done, info = env.step(1)  # INC
    assert obs["memory"] == 1
    obs, reward, done, info = env.step(2)  # DEC
    assert obs["memory"] == 0


def test_reaching_target_ends_episode_with_bonus():
    cfg = ToolEnvConfig(target_min=1, target_max=1, max_steps=5, step_penalty=-0.5, success_reward=2.0)
    env = ToolEnv(cfg, env_id=0)
    env.reset()
    obs, reward, done, info = env.step(1)  # move memory -> target
    assert done is True
    assert np.isclose(reward, cfg.step_penalty + cfg.success_reward)
    assert info.get("reason") == "reached_target"


def test_out_of_range_actions_are_noop():
    cfg = ToolEnvConfig(target_min=1, target_max=1, max_steps=3)
    env = ToolEnv(cfg, env_id=0)
    env.reset()
    obs, reward, done, info = env.step(5)  # invalid -> NOOP
    assert obs["memory"] == 0
    assert done is False


def test_tools_basic_oracle_can_get_positive_return():
    env = make_tools_basic_env(seed=3)
    total_reward = 0.0
    obs = env.reset()
    info = {}

    while True:
        action = oracle_policy(env, obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    assert info.get("reason") == "reached_target"
    assert total_reward > 0.0
