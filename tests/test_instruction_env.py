from instruction_env import InstructionEnv, InstructionEnvConfig


def _move_to(env: InstructionEnv, x: int, y: int) -> None:
    while env.agent_pos[0] > x:
        env.step(0)  # UP
    while env.agent_pos[0] < x:
        env.step(1)  # DOWN
    while env.agent_pos[1] > y:
        env.step(2)  # LEFT
    while env.agent_pos[1] < y:
        env.step(3)  # RIGHT


def test_instruction_env_correct_vs_wrong_goal():
    cfg = InstructionEnvConfig(size=7, view_size=5, max_steps=50, step_penalty=-0.01, success_reward=1.0, wrong_reward=-1.0)
    env = InstructionEnv(config=cfg, env_id=0, env_name="instr_test", seed=0)

    assert env.scenario_configs[0]["description"] != env.scenario_configs[1]["description"]

    # Scenario 0: goal A is correct at (1,1)
    env.reset(scenario_id=0)
    _move_to(env, 1, 1)
    _, _, done, info = env.step(5)  # TAKE
    assert done is True
    assert info["reason"] == "took_correct_goal"
    assert info["reward_env"] == cfg.success_reward
    assert info["instruction_success"] is True
    assert info["at_target"] is True

    # Scenario 0 but take goal B (wrong): (size-2, size-2)
    env.reset(scenario_id=0)
    _move_to(env, cfg.size - 2, cfg.size - 2)
    _, _, done, info = env.step(5)  # TAKE
    assert done is True
    assert info["reason"] == "took_wrong_goal"
    assert info["reward_env"] == cfg.wrong_reward
    assert info["instruction_success"] is False
    assert info["at_target"] is False


def test_instruction_env_expert_reaches_correct_goal():
    cfg = InstructionEnvConfig(size=7, view_size=5, max_steps=50, step_penalty=-0.01, success_reward=1.0, wrong_reward=-1.0)
    env = InstructionEnv(config=cfg, env_id=0, env_name="instr_expert", seed=1)
    env.reset(scenario_id=1)  # goal B

    done = False
    info = {}
    for _ in range(32):
        action = env.get_expert_action()
        assert action is not None
        _, _, done, info = env.step(int(action))
        if done:
            break

    assert done is True
    assert info.get("reason") == "took_correct_goal"


def test_instruction_env_progress_shaping_sign():
    cfg = InstructionEnvConfig(size=7, view_size=5, max_steps=50, step_penalty=-0.01, progress_reward=0.05, success_reward=1.0, wrong_reward=-1.0)
    env = InstructionEnv(config=cfg, env_id=0, env_name="instr_shape", seed=2)

    env.reset(scenario_id=0)  # target A at (1,1), start at center
    _, _, _, info_toward = env.step(0)  # UP (toward target)
    assert info_toward["reward_env"] > cfg.step_penalty

    env.reset(scenario_id=0)
    _, _, _, info_away = env.step(1)  # DOWN (away from target)
    assert info_away["reward_env"] < cfg.step_penalty


def test_instruction_env_single_goal_mode_spawns_only_target():
    cfg = InstructionEnvConfig(size=7, view_size=5, max_steps=50, spawn_both_goals=False)
    env = InstructionEnv(config=cfg, env_id=0, env_name="instr_single_goal", seed=4)

    env.reset(scenario_id=0)  # target A
    assert int(env.grid[1, 1]) == env.GOAL_A
    assert int(env.grid[cfg.size - 2, cfg.size - 2]) != env.GOAL_B

    env.reset(scenario_id=1)  # target B
    assert int(env.grid[cfg.size - 2, cfg.size - 2]) == env.GOAL_B
    assert int(env.grid[1, 1]) != env.GOAL_A


def test_instruction_env_marks_max_steps_success_when_already_at_target():
    cfg = InstructionEnvConfig(size=7, view_size=5, max_steps=5)
    env = InstructionEnv(config=cfg, env_id=0, env_name="instr_timeout_success", seed=7)
    env.reset(scenario_id=0)
    env.agent_pos = [1, 1]
    env.steps = cfg.max_steps - 1
    _, _, done, info = env.step(0)  # UP blocked by wall, stays on target and times out.
    assert done is True
    assert info["reason"] == "max_steps"
    assert info["at_target"] is True
    assert info["instruction_success"] is True


def test_instruction_env_marks_timeout_success_when_adjacent_to_target():
    cfg = InstructionEnvConfig(size=7, view_size=5, max_steps=5)
    env = InstructionEnv(config=cfg, env_id=0, env_name="instr_timeout_adjacent", seed=8)
    env.reset(scenario_id=0)
    env.agent_pos = [1, 2]  # Manhattan distance 1 from target A at (1,1)
    env.steps = cfg.max_steps - 1
    _, _, done, info = env.step(0)  # UP blocked by wall, remains adjacent, then max_steps triggers.
    assert done is True
    assert info["reason"] == "max_steps"
    assert info["distance_to_target"] == 1
    assert info["instruction_success"] is True

