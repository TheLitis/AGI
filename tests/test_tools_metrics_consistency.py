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
