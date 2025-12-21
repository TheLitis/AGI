from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_can_fix_task_via_patch_and_pytest(tmp_path):
    tasks = build_repo_taskset(["calc_add"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=12,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_test", seed=0)

    obs0 = env.reset(scenario_id=0)
    assert env.workdir is not None
    assert obs0["patch"].min() >= 0
    assert obs0["patch"].max() < env.n_cell_types

    # CYCLE_PATCHES should not reset the internal episode step counter, and should rotate patch candidates.
    pair0 = list(env.action_patch_indices)
    obs_cycle1, _, done_cycle1, info_cycle1 = env.step(5)
    assert done_cycle1 is False
    assert info_cycle1["steps_taken"] == 1
    assert obs_cycle1["patch"].min() >= 0
    assert obs_cycle1["patch"].max() < env.n_cell_types
    assert list(env.action_patch_indices) != pair0

    obs_cycle2, _, done_cycle2, info_cycle2 = env.step(5)
    assert done_cycle2 is False
    assert info_cycle2["steps_taken"] == 2
    assert obs_cycle2["patch"].min() >= 0
    assert obs_cycle2["patch"].max() < env.n_cell_types
    assert list(env.action_patch_indices) == pair0

    # First run: tests should fail on the initial buggy repo.
    obs_fail, _, done0, info0 = env.step(3)  # RUN_TESTS
    assert done0 is False
    assert info0["tests_total"] >= 1
    assert info0["last_test_passed"] is False
    assert obs_fail["patch"].min() >= 0
    assert obs_fail["patch"].max() < env.n_cell_types

    # Apply the correct patch and rerun: should pass and finish the episode.
    assert env.current_task is not None
    fix_patch_idx = None
    for idx, patch in enumerate(env.current_task.patches):
        if patch.name == "candidate_plus":
            fix_patch_idx = idx
            break
    assert fix_patch_idx is not None

    # Action 1 applies action_patch_indices[0], action 2 applies action_patch_indices[1].
    apply_action = None
    if env.action_patch_indices[0] == fix_patch_idx:
        apply_action = 1
    elif env.action_patch_indices[1] == fix_patch_idx:
        apply_action = 2
    assert apply_action in (1, 2)
    env.step(apply_action)
    obs_pass, reward1, done1, info1 = env.step(3)  # RUN_TESTS
    assert done1 is True
    assert info1["last_test_passed"] is True
    assert info1["tests_total"] >= 1
    assert info1["tests_passed"] == info1["tests_total"]
    assert reward1 > 0.0
    assert obs_pass["patch"].min() >= 0
    assert obs_pass["patch"].max() < env.n_cell_types
    assert env.workdir is None


def test_repo_tool_env_sets_death_flag_on_timeout_failure(tmp_path):
    tasks = build_repo_taskset(["calc_add"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=1,
        timeout_sec=10.0,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_timeout", seed=0)

    env.reset(scenario_id=0)
    _, _, done, info = env.step(3)  # RUN_TESTS (will fail) -> timeout by max_steps
    assert done is True
    assert info["last_test_passed"] is False
    assert info["death_flag"] == 1.0
    assert info["alive"] == 0.0
    assert env.workdir is None
