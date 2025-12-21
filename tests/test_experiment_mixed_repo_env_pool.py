from experiment import _build_mixed_env_pool


def test_build_mixed_env_pool_can_include_repo_envs():
    pool = _build_mixed_env_pool(
        seed=123,
        schedule_mode="iid",
        episodes_per_phase=5,
        repo_scenarios=["calc_add"],
    )
    assert getattr(pool, "task_metadata_repo", None) is not None

    repo_indices = [i for i, e in enumerate(pool.envs) if "repo" in str(getattr(e, "env_family", "")).lower()]
    assert repo_indices, "Expected repo envs to be present in the mixed EnvPool"

    pool.active_env_ids = repo_indices
    obs = pool.reset(split="train")
    assert "repo" in str(obs.get("env_family", "")).lower()

