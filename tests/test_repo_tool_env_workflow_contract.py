from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def _make_env(tmp_path):
    tasks = build_repo_taskset(["proc_bundle_loop"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=48,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
        step_penalty=-0.01,
        toolloop_bootstrap_run_tests_penalty=-0.02,
        toolloop_run_tests_penalty=-0.04,
        toolloop_idle_with_candidates_penalty=-0.02,
        toolloop_redundant_run_tests_penalty=-0.03,
    )
    return RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_toolloop_contract", seed=0)


def test_redundant_run_tests_penalty_and_event(tmp_path):
    env = _make_env(tmp_path)
    env.reset(scenario_id=0)

    _obs, _reward_first, _done_first, info_first = env.step(3)  # RUN_TESTS bootstrap
    assert "events" in info_first and isinstance(info_first["events"], dict)

    _obs, reward_second, _done_second, info_second = env.step(3)  # RUN_TESTS without workspace changes
    events = info_second.get("events", {})
    assert float(events.get("redundant_run_tests", 0.0)) == 1.0

    # Redundant run-tests should carry an explicit shaping penalty.
    assert reward_second <= -0.08
