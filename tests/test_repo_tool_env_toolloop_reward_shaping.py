from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_toolloop_reward_shaping_discourages_apply_spam(tmp_path):
    tasks = build_repo_taskset(["proc_bundle_loop"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=64,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
        step_penalty=-0.01,
        apply_patch_penalty=-0.02,
        run_tests_penalty=-0.10,
        toolloop_bootstrap_run_tests_penalty=-0.02,
        toolloop_candidate_reward=0.05,
        toolloop_run_tests_penalty=-0.04,
        toolloop_apply_without_candidates_penalty=-0.06,
        toolloop_repeat_apply_penalty=-0.06,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_toolloop", seed=0)

    # Tool-loop tasks start without candidates; patch actions should be unattractive vs RUN_TESTS.
    env.reset(scenario_id=0)
    _, r_apply, _, _ = env.step(1)  # APPLY_PATCH_0

    env.reset(scenario_id=0)
    _, r_test, done, info = env.step(3)  # RUN_TESTS

    assert done is False
    assert info["last_test_passed"] is False
    assert env.current_task is not None
    assert env.current_task.patches
    assert r_test > r_apply

    # After applying a patch once, re-applying the same patch slot should be penalized.
    env.step(1)
    _, r_apply_repeat, _, _ = env.step(1)
    assert r_apply_repeat < -0.03
