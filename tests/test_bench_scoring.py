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


def test_lifelong_ci_samples_use_lifelong_score_not_raw_delta():
    run_records = [
        {
            "status": "ok",
            "result": {
                "stage_metrics": {
                    "lifelong_eval": {
                        "lifelong_forgetting_R1_gap": -0.5,
                        "lifelong_adaptation_R2_delta": 1.0,
                        "lifelong_adaptation_R3_delta": 0.0,
                    }
                }
            },
            "eval": {},
        }
    ]
    vals = bench._suite_ci_sample_values("lifelong", run_records)
    expected = bench._lifelong_score(forgetting_gap=-0.5, forward_transfer=0.6)
    assert expected is not None
    assert len(vals) == 1
    assert vals[0] == expected


def test_lifelong_diag_ci_samples_use_lifelong_score_not_raw_delta():
    run_records = [
        {
            "status": "ok",
            "result": {
                "stage_metrics": {
                    "lifelong_eval": {
                        "lifelong_forgetting_R1_gap": -0.5,
                        "lifelong_adaptation_R2_delta": 1.0,
                        "lifelong_adaptation_R3_delta": 0.0,
                    }
                }
            },
            "eval": {},
        }
    ]
    vals = bench._suite_ci_sample_values("lifelong_diag", run_records)
    expected = bench._lifelong_score(forgetting_gap=-0.5, forward_transfer=0.6)
    assert expected is not None
    assert len(vals) == 1
    assert vals[0] == expected


def test_core_ci_samples_use_normalized_core_score():
    run_records = [
        {
            "status": "ok",
            "result": {},
            "eval": {
                "mean_return": 12.0,
                "test_mean_return": 9.0,
            },
        },
        {
            "status": "ok",
            "result": {},
            "eval": {
                "mean_return": 3.0,
                "test_mean_return": 1.0,
            },
        },
    ]
    vals = bench._suite_ci_sample_values("core", run_records)
    expected0 = bench._core_score(12.0, 9.0)
    expected1 = bench._core_score(3.0, 1.0)
    assert expected0 is not None and expected1 is not None
    assert vals == [expected0, expected1]
    assert all(0.0 <= float(v) <= 1.0 for v in vals)


def test_long_horizon_score_prefers_better_profiles():
    good = bench._long_horizon_score(
        mean_return=2.0,
        horizon_utilization=0.85,
        goal_completion_rate=0.75,
        mean_steps_to_goal=25.0,
        horizon_steps=120,
        planner_gain=1.0,
        timeout_rate=0.05,
    )
    bad = bench._long_horizon_score(
        mean_return=-2.0,
        horizon_utilization=0.30,
        goal_completion_rate=0.10,
        mean_steps_to_goal=110.0,
        horizon_steps=120,
        planner_gain=-1.0,
        timeout_rate=0.60,
    )
    assert good is not None and bad is not None
    assert 0.0 <= bad < good <= 1.0


def test_long_horizon_success_metrics_use_explicit_episode_fields():
    eval_metrics = {
        "episode_success_rate": 0.7,
        "mean_steps_to_success": 18.0,
        "reason_counts": {"goal_reached": 1, "max_steps": 9},
    }
    success, mean_steps = bench._long_horizon_success_metrics_from_eval(eval_metrics)
    assert success == 0.7
    assert mean_steps == 18.0


def test_long_horizon_success_metrics_fall_back_to_reason_counts():
    eval_metrics = {
        "reason_counts": {
            "goal_reached": 3,
            "max_steps": 1,
        },
        "mean_length": 22.0,
    }
    success, mean_steps = bench._long_horizon_success_metrics_from_eval(eval_metrics)
    assert success == 0.75
    assert mean_steps == 22.0


def test_long_horizon_success_metrics_does_not_force_zero_without_success_labels():
    eval_metrics = {
        "reason_counts": {
            "done": 8,
        },
        "episode_success_coverage": 0.0,
        "mean_length": 40.0,
    }
    success, mean_steps = bench._long_horizon_success_metrics_from_eval(eval_metrics)
    assert success is None
    assert mean_steps is None


def test_long_horizon_success_metrics_keeps_survival_success_without_step_penalty():
    eval_metrics = {
        "episode_success_rate": 0.75,
        "episode_success_coverage": 1.0,
        "mean_length": 85.0,
        "reason_counts": {
            "eval_max_steps_cap": 6,
            "terminated_danger": 2,
        },
    }
    success, mean_steps = bench._long_horizon_success_metrics_from_eval(eval_metrics)
    assert success == 0.75
    assert mean_steps is None


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


def test_safety_metrics_from_eval_requires_contract_fields():
    compliance, catastrophic = bench._safety_metrics_from_eval({"death_rate": 0.25})
    assert compliance is None
    assert catastrophic is None


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


def test_suite_specs_include_planning_diag_cross_env_cases():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    suite = specs["planning_diag"]
    assert suite.implemented is True
    assert any(c.env_type == "gridworld" for c in suite.cases)
    assert any(c.env_type == "minigrid" for c in suite.cases)


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
    assert any(c.env_type == "minigrid" for c in long_suite.cases)
    assert safety_suite.cases[0].env_type == "gridworld"
    assert int(safety_suite.cases[0].max_steps_env or 0) >= 100
    assert int(safety_suite.cases[0].max_energy_env or 0) >= int(safety_suite.cases[0].max_steps_env or 0)


def test_suite_specs_include_lifelong_diag_cross_env_cases():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    suite = specs["lifelong_diag"]
    assert suite.implemented is True
    assert any(c.env_type == "gridworld" for c in suite.cases)
    assert any(c.env_type == "minigrid" for c in suite.cases)


def test_suite_specs_include_lifelong_cross_env_cases():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    suite = specs["lifelong"]
    assert suite.implemented is True
    assert any(c.env_type == "gridworld" for c in suite.cases)
    assert any(c.env_type == "minigrid" for c in suite.cases)


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


def test_long_horizon_metrics_template_exposes_planner_reality_fields():
    tpl = bench._metric_template("long_horizon")
    assert "success_rate" in tpl
    assert "efficiency_score" in tpl
    assert "planner_reality_steps" in tpl
    assert "planner_score_nstep_corr" in tpl
    assert "planner_top1_match_rate" in tpl
    assert "planner_regret_proxy_nstep" in tpl


def test_planning_diag_metrics_template_exposes_success_efficiency_and_reality_fields():
    tpl = bench._metric_template("planning_diag")
    assert "success_rate" in tpl
    assert "efficiency_score" in tpl
    assert "planner_reality_steps" in tpl
    assert "planner_score_nstep_corr" in tpl
    assert "planner_regret_proxy_nstep" in tpl


def test_planner_reality_metrics_from_eval_sanitizes_values():
    metrics = bench._planner_reality_metrics_from_eval(
        {
            "planner_reality_steps": 12,
            "planner_score_nstep_corr": 0.7,
            "policy_score_nstep_corr": -0.2,
            "planner_score_corr_advantage": 0.9,
            "planner_top1_match_rate": 1.2,
            "policy_top1_match_rate": -0.5,
            "planner_top1_advantage_nstep": 0.4,
            "planner_regret_proxy_nstep": 0.3,
        }
    )
    assert metrics["planner_reality_steps"] == 12.0
    assert metrics["planner_score_nstep_corr"] == 0.7
    assert metrics["policy_score_nstep_corr"] == -0.2
    assert metrics["planner_score_corr_advantage"] == 0.9
    assert metrics["planner_top1_match_rate"] == 1.0
    assert metrics["policy_top1_match_rate"] == 0.0
    assert metrics["planner_top1_advantage_nstep"] == 0.4
    assert metrics["planner_regret_proxy_nstep"] == 0.3
