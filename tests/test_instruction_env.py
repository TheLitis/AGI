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

    # Scenario 0 but take goal B (wrong): (size-2, size-2)
    env.reset(scenario_id=0)
    _move_to(env, cfg.size - 2, cfg.size - 2)
    _, _, done, info = env.step(5)  # TAKE
    assert done is True
    assert info["reason"] == "took_wrong_goal"
    assert info["reward_env"] == cfg.wrong_reward

