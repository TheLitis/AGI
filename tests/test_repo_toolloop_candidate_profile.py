import experiment


def test_build_repo_env_pool_applies_toolloop_candidate_profile():
    pool = experiment._build_repo_env_pool(
        seed=0,
        schedule_mode="iid",
        scenario_names=["train:proc_mixed_loop", "test:proc_mixed_loop"],
        repo_toolloop_max_candidates=2,
        repo_toolloop_prefer_solution_first_pair=True,
    )
    assert len(pool.envs) >= 2
    for env in pool.envs:
        cfg = getattr(env, "config", None)
        assert cfg is not None
        assert int(getattr(cfg, "toolloop_max_candidates", -1)) == 2
        assert bool(getattr(cfg, "toolloop_prefer_solution_first_pair", False)) is True
