import bench


def test_extract_repo_metrics_reads_unmasked_recovery():
    eval_metrics = {
        "repo_pass_rate": 0.4,
        "unmasked": {
            "repo_pass_rate": 0.6,
            "repo_steps_to_pass": [9, 11],
            "repo_recovery_rate": 0.5,
        },
    }
    pass_masked, pass_unmasked, steps_unmasked, recovery_unmasked = bench._extract_repo_metrics(eval_metrics)
    assert pass_masked == 0.4
    assert pass_unmasked == 0.6
    assert steps_unmasked == [9, 11]
    assert recovery_unmasked == 0.5


def test_tools_score_penalizes_low_recovery_rate():
    score_high_recovery = bench._tools_score(
        pass_unmasked=0.8,
        invalid_action_rate=0.0,
        mean_steps_unmasked=10.0,
        recovery_rate=1.0,
    )
    score_low_recovery = bench._tools_score(
        pass_unmasked=0.8,
        invalid_action_rate=0.0,
        mean_steps_unmasked=10.0,
        recovery_rate=0.5,
    )
    assert score_high_recovery is not None
    assert score_low_recovery is not None
    assert score_low_recovery < score_high_recovery


def test_tools_metric_templates_include_recovery_rate():
    assert "recovery_rate" in bench._metric_template("tools")
    assert "recovery_rate" in bench._metric_template("tools_open")


def test_extract_eval_metrics_tools_prefers_best_checkpoint():
    run_result = {
        "stage_metrics": {
            "eval_after_stage2": {
                "repo_pass_rate": 1.0,
                "unmasked": {"repo_pass_rate": 0.70, "repo_steps_to_pass": [8, 10]},
            },
            "eval_after_stage4_no_self": {
                "repo_pass_rate": 1.0,
                "unmasked": {"repo_pass_rate": 0.25, "repo_steps_to_pass": [2]},
            },
            "eval_after_stage4_self": {
                "repo_pass_rate": 1.0,
                "unmasked": {"repo_pass_rate": 0.10, "repo_steps_to_pass": [2]},
            },
        }
    }
    selected = bench._extract_eval_metrics(run_result, suite_name="tools")
    assert isinstance(selected, dict)
    pass_masked, pass_unmasked, _steps, _recovery = bench._extract_repo_metrics(selected)
    assert pass_masked == 1.0
    assert pass_unmasked == 0.70


def test_extract_eval_metrics_non_tools_keeps_stage4_selection():
    run_result = {
        "stage_metrics": {
            "eval_after_stage2": {"mean_return": 9.0},
            "eval_after_stage4_no_self": {"mean_return": 1.0},
            "eval_after_stage4_self": {"mean_return": 2.0},
        }
    }
    selected = bench._extract_eval_metrics(run_result, suite_name="core")
    assert isinstance(selected, dict)
    # Non-tools suites should still choose between stage4 self/no_self only.
    assert float(selected.get("mean_return")) == 2.0


def test_extract_eval_metrics_safety_prefers_best_safety_checkpoint():
    run_result = {
        "stage_metrics": {
            "eval_after_stage2": {
                "constraint_compliance": 0.70,
                "catastrophic_fail_rate": 0.0,
            },
            "eval_after_stage3_no_self": {
                "constraint_compliance": 0.82,
                "catastrophic_fail_rate": 0.0,
            },
            "eval_after_stage4_no_self": {
                "constraint_compliance": 0.90,
                "catastrophic_fail_rate": 0.35,
            },
            "eval_after_stage4_self": {
                "constraint_compliance": 0.80,
                "catastrophic_fail_rate": 0.30,
            },
        }
    }
    selected = bench._extract_eval_metrics(run_result, suite_name="safety")
    assert isinstance(selected, dict)
    # stage3_no_self has the best (compliance - catastrophic) quality.
    assert float(selected.get("constraint_compliance")) == 0.82
    assert float(selected.get("catastrophic_fail_rate")) == 0.0


def test_extract_eval_metrics_long_horizon_prefers_best_checkpoint():
    run_result = {
        "stage_metrics": {
            "eval_after_stage2": {
                "mean_return": 9.0,
                "mean_length": 110.0,
                "max_steps": 120,
                "timeout_rate": 0.0,
            },
            "eval_after_stage4_no_self": {
                "mean_return": 1.0,
                "mean_length": 20.0,
                "max_steps": 120,
                "timeout_rate": 0.8,
            },
            "eval_after_stage4_self": {
                "mean_return": 0.5,
                "mean_length": 15.0,
                "max_steps": 120,
                "timeout_rate": 0.9,
            },
        }
    }
    selected = bench._extract_eval_metrics(run_result, suite_name="long_horizon")
    assert isinstance(selected, dict)
    assert float(selected.get("mean_return")) == 9.0
    assert float(selected.get("mean_length")) == 110.0


def test_extract_eval_metrics_lifelong_prefers_best_checkpoint():
    run_result = {
        "stage_metrics": {
            "eval_after_stage2": {"mean_return": 4.0},
            "eval_after_stage4_no_self": {"mean_return": 1.0},
            "eval_after_stage4_self": {"mean_return": 2.0},
        }
    }
    selected = bench._extract_eval_metrics(run_result, suite_name="lifelong")
    assert isinstance(selected, dict)
    assert float(selected.get("mean_return")) == 4.0
