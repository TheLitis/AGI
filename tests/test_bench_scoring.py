import bench


def test_bounded_return_score_monotonic():
    low = bench._bounded_return_score(-5.0, center=0.0, scale=1.0)
    mid = bench._bounded_return_score(0.0, center=0.0, scale=1.0)
    high = bench._bounded_return_score(5.0, center=0.0, scale=1.0)
    assert low is not None and mid is not None and high is not None
    assert 0.0 <= low < mid < high <= 1.0


def test_language_social_score_use_ood_transfer_floor():
    assert bench._language_score(0.8, 0.6) == 0.6
    assert bench._social_score(0.9, 0.4) == 0.4
    assert bench._language_score(0.7, None) == 0.7
    assert bench._social_score(0.5, None) == 0.5


def test_lifelong_score_prefers_low_forgetting_and_positive_transfer():
    good = bench._lifelong_score(forgetting_gap=0.1, forward_transfer=3.0)
    bad = bench._lifelong_score(forgetting_gap=10.0, forward_transfer=-3.0)
    assert good is not None and bad is not None
    assert 0.0 <= bad < good <= 1.0


def test_lifelong_forward_transfer_uses_weighted_r2_r3():
    v = bench._lifelong_forward_transfer_from_eval(
        {
            "lifelong_adaptation_R2_delta": 1.0,
            "lifelong_adaptation_R3_delta": 0.0,
        }
    )
    assert v == 0.6


def test_lifelong_forward_transfer_falls_back_to_single_delta():
    assert bench._lifelong_forward_transfer_from_eval({"lifelong_adaptation_R2_delta": 0.25}) == 0.25
    assert bench._lifelong_forward_transfer_from_eval({"lifelong_adaptation_R3_delta": -0.5}) == -0.5


def test_long_horizon_score_prefers_better_profiles():
    good = bench._long_horizon_score(
        mean_return=2.0,
        horizon_utilization=0.85,
        planner_gain=1.0,
        timeout_rate=0.05,
    )
    bad = bench._long_horizon_score(
        mean_return=-2.0,
        horizon_utilization=0.30,
        planner_gain=-1.0,
        timeout_rate=0.60,
    )
    assert good is not None and bad is not None
    assert 0.0 <= bad < good <= 1.0


def test_safety_score_prefers_high_compliance_and_low_catastrophic():
    good = bench._safety_score(
        planner_ok=True,
        constraint_compliance=0.95,
        catastrophic_fail_rate=0.02,
    )
    bad = bench._safety_score(
        planner_ok=False,
        constraint_compliance=0.40,
        catastrophic_fail_rate=0.60,
    )
    assert good is not None and bad is not None
    assert 0.0 <= bad < good <= 1.0


def test_safety_metrics_from_eval_falls_back_to_death_rate():
    compliance, catastrophic = bench._safety_metrics_from_eval({"death_rate": 0.25})
    assert compliance == 0.75
    assert catastrophic == 0.25


def test_suite_specs_enable_language_social_lifelong_long_horizon():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    for name in ("language", "social", "lifelong", "long_horizon"):
        suite = specs[name]
        assert suite.implemented is True
        assert len(suite.cases) >= 1


def test_suite_specs_enable_tools_open_repo_case():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    suite = specs["tools_open"]
    assert suite.implemented is True
    assert len(suite.cases) == 1
    assert suite.cases[0].env_type == "repo"
    assert suite.cases[0].repo_scenarios == ["train:proc_mixed_open", "test:proc_mixed_open"]


def test_suite_specs_enable_long_horizon_and_safety_gridworld_cases():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    long_suite = specs["long_horizon"]
    safety_suite = specs["safety"]
    assert long_suite.cases[0].env_type == "gridworld"
    assert int(long_suite.cases[0].max_steps_env or 0) >= 100
    assert int(long_suite.cases[0].max_energy_env or 0) >= int(long_suite.cases[0].max_steps_env or 0)
    assert safety_suite.cases[0].env_type == "gridworld"
    assert int(safety_suite.cases[0].max_steps_env or 0) >= 100
    assert int(safety_suite.cases[0].max_energy_env or 0) >= int(safety_suite.cases[0].max_steps_env or 0)


def test_language_rates_prefer_explicit_success_metrics():
    eval_metrics = {
        "instruction_success_rate": 0.8,
        "instruction_test_success_rate": 0.6,
        "mean_return": -10.0,
        "test_mean_return": -10.0,
    }
    p, ood = bench._language_rates_from_eval(eval_metrics)
    assert p == 0.8
    assert ood == 0.6


def test_social_rates_prefer_explicit_success_metrics():
    eval_metrics = {
        "social_success_rate": 0.9,
        "social_test_success_rate": 0.7,
        "mean_return": -10.0,
        "test_mean_return": -10.0,
    }
    s, t = bench._social_rates_from_eval(eval_metrics)
    assert s == 0.9
    assert t == 0.7


def test_tools_metrics_template_exposes_repo_bc_runtime_config():
    tpl = bench._metric_template("tools")
    assert "repo_online_bc_coef" in tpl
    assert "repo_bc_pretrain_episodes" in tpl
