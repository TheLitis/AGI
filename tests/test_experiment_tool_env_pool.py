from experiment import _build_tool_env_pool


def test_build_tool_env_pool_smoke():
    pool = _build_tool_env_pool(seed=123, schedule_mode="iid")
    assert pool.train_env_ids == [0]
    assert pool.test_env_ids == [1]

    obs = pool.reset(split="train")
    assert obs.get("env_family") == "tools"

