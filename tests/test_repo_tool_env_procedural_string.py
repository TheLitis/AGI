from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_procedural_string_task_is_solvable(tmp_path):
    tasks = build_repo_taskset(["proc_string"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=80,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
        procedural_candidates=8,
        procedural_test_cases=6,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_proc_string", seed=0)

    obs = env.reset(scenario_id=0)
    assert env.current_task is not None
    assert env.current_task.initial_files, "procedural task should materialize files"
    assert env.current_task.patches, "procedural task should provide candidate patches"
    assert obs["patch"].min() >= 0
    assert obs["patch"].max() < env.n_cell_types

    # Initial run should fail (bugged implementation).
    _, _, done0, info0 = env.step(3)  # RUN_TESTS
    assert done0 is False
    assert info0["last_test_passed"] is False

    # Try all candidate pairs via CYCLE_PATCHES until a fix is found.
    max_cycles = 10
    for _ in range(max_cycles):
        for apply_action in (1, 2):
            env.step(apply_action)
            _, _, done, info = env.step(3)  # RUN_TESTS
            if info.get("last_test_passed") is True:
                assert done is True
                assert info["tests_total"] >= 1
                assert info["tests_passed"] == info["tests_total"]
                return
            env.step(4)  # REVERT repo state, keep candidate navigation
        env.step(5)  # CYCLE_PATCHES

    raise AssertionError("procedural string task did not become solvable within the candidate search budget")

