from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_open_task_keeps_full_action_mask(tmp_path):
    tasks = build_repo_taskset(["proc_mixed_open"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=64,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_open", seed=0)

    env.reset(scenario_id=0)
    mask = env.get_action_mask()
    assert mask.sum() == float(env.n_actions)

    env.step(1)
    mask_after_apply = env.get_action_mask()
    assert mask_after_apply.sum() == float(env.n_actions)

    env.step(3)
    mask_after_test = env.get_action_mask()
    assert mask_after_test.sum() == float(env.n_actions)
