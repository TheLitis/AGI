from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_noop_cycles_inspection_view(tmp_path):
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
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_view", seed=0)

    obs0 = env.reset(scenario_id=0)
    assert getattr(env, "view_mode", 0) == 0
    assert obs0["patch"].min() >= 0
    assert obs0["patch"].max() < env.n_cell_types

    # Generate pytest output so view=2 has something to tokenize.
    env.step(3)  # RUN_TESTS

    obs1, _, _, _ = env.step(0)  # NO_OP -> view 1 (file list)
    assert getattr(env, "view_mode", 0) == 1
    assert obs1["patch"].min() >= 0
    assert obs1["patch"].max() < env.n_cell_types

    obs2, _, _, _ = env.step(0)  # NO_OP -> view 2 (pytest output)
    assert getattr(env, "view_mode", 0) == 2
    assert obs2["patch"].min() >= 0
    assert obs2["patch"].max() < env.n_cell_types

    # With pytest output present, view 2 should carry at least one non-zero hashed token.
    assert int(obs2["patch"][3].max()) > 0 or int(obs2["patch"][4].max()) > 0

    obs3, _, _, _ = env.step(0)  # NO_OP -> view 3 (focus snippet)
    assert getattr(env, "view_mode", 0) == 3
    assert obs3["patch"].min() >= 0
    assert obs3["patch"].max() < env.n_cell_types
    assert int(obs3["patch"][3].max()) > 0 or int(obs3["patch"][4].max()) > 0
