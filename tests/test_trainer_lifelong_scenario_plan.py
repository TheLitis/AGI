from trainer import RegimeConfig, Trainer


def test_build_episode_scenario_plan_stratified_respects_weighted_counts():
    trainer = Trainer.__new__(Trainer)
    regime = RegimeConfig(name="r", scenario_weights={"a": 3.0, "b": 1.0})
    name_to_id = {"a": 10, "b": 20}

    plan = trainer._build_episode_scenario_plan(
        regime=regime,
        name_to_id=name_to_id,
        n_episodes=12,
        stratified=True,
    )

    assert len(plan) == 12
    assert set(plan) == {10, 20}
    assert plan.count(10) == 9
    assert plan.count(20) == 3


def test_build_episode_scenario_plan_returns_none_when_mapping_missing():
    trainer = Trainer.__new__(Trainer)
    regime = RegimeConfig(name="r", scenario_weights={"missing": 1.0})

    plan = trainer._build_episode_scenario_plan(
        regime=regime,
        name_to_id={},
        n_episodes=5,
        stratified=True,
    )

    assert plan == [None, None, None, None, None]
