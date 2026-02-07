from social_env import SocialEnv, SocialEnvConfig


def _move_to(env: SocialEnv, x: int, y: int) -> None:
    while env.agent_pos[0] > x:
        env.step(0)  # UP
    while env.agent_pos[0] < x:
        env.step(1)  # DOWN
    while env.agent_pos[1] > y:
        env.step(2)  # LEFT
    while env.agent_pos[1] < y:
        env.step(3)  # RIGHT


def test_social_env_coop_vs_compete():
    cfg = SocialEnvConfig(size=7, view_size=5, max_steps=60, step_penalty=-0.01, success_reward=1.0, fail_reward=-1.0)
    env = SocialEnv(config=cfg, env_id=0, env_name="social_test", seed=0)

    # Compete: if we do nothing, the other agent should eventually collect first => negative reward.
    env.reset(scenario_id=1)
    done = False
    info = None
    for _ in range(40):
        _, _, done, info = env.step(4)  # STAY
        if done:
            break
    assert done is True
    assert info is not None
    assert info["reason"] == "other_got_food"
    assert info["reward_env"] == cfg.fail_reward

    # Coop: same behavior should still yield success when the other collects.
    env.reset(scenario_id=0)
    done = False
    info = None
    for _ in range(40):
        _, _, done, info = env.step(4)  # STAY
        if done:
            break
    assert done is True
    assert info is not None
    assert info["reason"] == "food_collected"
    assert info["reward_env"] == cfg.success_reward


def test_social_env_you_can_win_race_in_compete():
    cfg = SocialEnvConfig(size=7, view_size=5, max_steps=60, step_penalty=-0.01, success_reward=1.0, fail_reward=-1.0)
    env = SocialEnv(config=cfg, env_id=0, env_name="social_race", seed=0)

    env.reset(scenario_id=1)  # compete
    fx, fy = env.food_pos
    _move_to(env, fx, fy)
    _, _, done, info = env.step(5)  # TAKE
    assert done is True
    assert info["reason"] == "you_got_food"
    assert info["reward_env"] == cfg.success_reward


def test_social_env_expert_collects_food_in_compete():
    cfg = SocialEnvConfig(size=7, view_size=5, max_steps=60, step_penalty=-0.01, success_reward=1.0, fail_reward=-1.0)
    env = SocialEnv(config=cfg, env_id=0, env_name="social_expert", seed=1)
    env.reset(scenario_id=1)  # compete

    done = False
    info = {}
    for _ in range(32):
        action = env.get_expert_action()
        assert action is not None
        _, _, done, info = env.step(int(action))
        if done:
            break

    assert done is True
    assert info.get("reason") == "you_got_food"


def test_social_env_progress_shaping_sign():
    cfg = SocialEnvConfig(size=7, view_size=5, max_steps=60, step_penalty=-0.01, progress_reward=0.05, success_reward=1.0, fail_reward=-1.0)
    env = SocialEnv(config=cfg, env_id=0, env_name="social_shape", seed=3)

    env.reset(scenario_id=1)
    _, _, _, info_toward = env.step(0)  # UP toward food at (1, size-2)
    assert info_toward["reward_env"] > cfg.step_penalty

    env.reset(scenario_id=1)
    _, _, _, info_away = env.step(1)  # DOWN away from food
    assert info_away["reward_env"] < cfg.step_penalty
