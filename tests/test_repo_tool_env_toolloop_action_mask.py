from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_toolloop_action_mask_transitions(tmp_path):
    tasks = build_repo_taskset(["proc_bundle_loop"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=64,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_toolloop", seed=0)

    env.reset(scenario_id=0)
    mask = env.get_action_mask()
    assert mask[1] == 1.0 and mask[2] == 1.0
    assert mask[3] == 0.0

    env.step(1)
    mask_after_apply = env.get_action_mask()
    assert mask_after_apply[3] == 1.0
    assert mask_after_apply[1] == 0.0 and mask_after_apply[2] == 0.0

    env.step(3)
    mask_after_test = env.get_action_mask()
    if env.last_test_passed is True:
        assert mask_after_test.sum() == float(env.n_actions)
    else:
        assert mask_after_test[1] == 1.0 and mask_after_test[2] == 1.0
