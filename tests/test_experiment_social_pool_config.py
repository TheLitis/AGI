from experiment import _build_social_env_pool


def test_social_pool_uses_tuned_compete_probability():
    pool = _build_social_env_pool(seed=0, schedule_mode="iid")
    probs = [float(getattr(env.config, "compete_probability", -1.0)) for env in pool.envs]
    assert probs
    assert all(abs(p - 0.25) < 1e-9 for p in probs)
