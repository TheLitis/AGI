import bench


def test_refresh_overall_partial_report_keeps_gate1_na():
    report = {
        "suites": [
            {"name": "tools", "status": "ok", "score": 0.8, "metrics": {"pass_rate_unmasked": 0.8}},
        ]
    }
    bench._refresh_overall(report)
    gates = report["overall"]["gates"]
    assert gates["gate0"] == "pass"
    assert gates["gate1"] == "na"
    assert gates["gate2"] == "na"
    assert gates["gate3"] == "na"
    assert gates["gate4"] == "na"


def test_refresh_overall_full_report_gate1_pass_gate2_fail():
    report = {
        "suites": [
            {"name": "core", "status": "ok", "score": 0.80, "metrics": {"mean_return": 10.0, "test_mean_return": 8.0}},
            {"name": "tools", "status": "ok", "score": 0.75, "metrics": {"pass_rate_unmasked": 0.75, "mean_steps_to_pass_unmasked": 11.0}},
            {"name": "language", "status": "ok", "score": 0.60, "metrics": {"pass_rate": 0.60, "causal_drop": 0.10}},
            {"name": "social", "status": "ok", "score": 0.65, "metrics": {"success_rate": 0.65, "transfer_rate": 0.60}},
            {"name": "lifelong", "status": "ok", "score": 0.55, "metrics": {"forgetting_gap": -1.5, "forward_transfer": 0.1}},
            {"name": "safety", "status": "ok", "score": None, "metrics": {"safety_planner_ok": True}},
        ]
    }
    bench._refresh_overall(report)
    gates = report["overall"]["gates"]
    assert gates["gate0"] == "pass"
    assert gates["gate1"] == "pass"
    assert gates["gate2"] == "fail"
    assert gates["gate3"] == "fail"
    assert gates["gate4"] == "fail"


def test_refresh_overall_full_report_gate0_fail_when_suite_not_ok():
    report = {
        "suites": [
            {"name": "core", "status": "ok", "score": 0.8, "metrics": {}},
            {"name": "tools", "status": "timeout", "score": None, "metrics": {}},
            {"name": "language", "status": "ok", "score": 0.6, "metrics": {}},
            {"name": "social", "status": "ok", "score": 0.6, "metrics": {}},
            {"name": "lifelong", "status": "ok", "score": 0.6, "metrics": {}},
            {"name": "safety", "status": "ok", "score": None, "metrics": {}},
        ]
    }
    bench._refresh_overall(report)
    gates = report["overall"]["gates"]
    assert gates["gate0"] == "fail"
    assert gates["gate1"] == "fail"
    assert gates["gate2"] == "fail"
    assert gates["gate3"] == "fail"
    assert gates["gate4"] == "fail"


def test_refresh_overall_gate3_gate4_pass_when_thresholds_met():
    report = {
        "meta": {"seed_list": [0, 1, 2, 3, 4]},
        "suites": [
            {
                "name": "core",
                "status": "ok",
                "score": 0.93,
                "ci": {"half_width": 0.50},
                "metrics": {"mean_return": 10.0, "test_mean_return": 8.0},
            },
            {
                "name": "tools",
                "status": "ok",
                "score": 0.90,
                "ci": {"half_width": 0.02},
                "metrics": {"pass_rate_unmasked": 0.90, "mean_steps_to_pass_unmasked": 8.0},
            },
            {
                "name": "language",
                "status": "ok",
                "score": 0.80,
                "ci": {"half_width": 0.02},
                "metrics": {"pass_rate": 0.80, "causal_drop": 0.05},
            },
            {
                "name": "social",
                "status": "ok",
                "score": 0.78,
                "ci": {"half_width": 0.02},
                "metrics": {"success_rate": 0.80, "transfer_rate": 0.75},
            },
            {
                "name": "lifelong",
                "status": "ok",
                "score": 0.80,
                "ci": {"half_width": 0.10},
                "metrics": {"forgetting_gap": -0.5, "forward_transfer": 0.8},
            },
            {"name": "safety", "status": "ok", "score": None, "metrics": {"safety_planner_ok": True}},
        ],
    }
    bench._refresh_overall(report)
    gates = report["overall"]["gates"]
    assert gates["gate0"] == "pass"
    assert gates["gate1"] == "pass"
    assert gates["gate2"] == "pass"
    assert gates["gate3"] == "pass"
    assert gates["gate4"] == "pass"
    caps = report["overall"]["capabilities"]
    assert caps["generalization_score"] is not None
    assert caps["sample_efficiency_score"] is not None
    assert caps["robustness_score"] is not None
    assert caps["tool_workflow_score"] is not None
    assert report["overall"]["confidence"] is not None
