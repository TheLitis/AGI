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


def test_suite_specs_enable_language_social_lifelong():
    specs = bench._build_suite_specs(
        minigrid_override=None,
        computer_override=None,
        repo_override=None,
        ood=False,
    )
    for name in ("language", "social", "lifelong"):
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
