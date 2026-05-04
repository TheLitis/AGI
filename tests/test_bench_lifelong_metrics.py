import pytest

import bench


def test_lifelong_forgetting_gap_prefers_worst_gap_when_present():
    payload = {
        "lifelong_forgetting_R1_gap": 0.2,
        "lifelong_forgetting_worst_gap": -1.5,
    }

    assert bench._lifelong_forgetting_gap_from_eval(payload) == pytest.approx(-1.5)


def test_lifelong_forgetting_gap_falls_back_to_legacy_r1():
    assert bench._lifelong_forgetting_gap_from_eval({"lifelong_forgetting_R1_gap": -0.25}) == pytest.approx(-0.25)


def test_lifelong_score_penalizes_worst_gap_in_suite_scoring():
    report = {
        "meta": {"seed_list": [0, 1, 2, 3, 4]},
        "suites": [
            {
                "name": "long_horizon",
                "status": "ok",
                "score": 0.80,
                "ci": {"half_width": 0.02},
                "metrics": {"horizon_utilization": 0.85, "timeout_rate": 0.05},
            },
            {
                "name": "core",
                "status": "ok",
                "score": 0.93,
                "ci": {"half_width": 0.02},
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
                "ci": {"half_width": 0.02},
                "metrics": {
                    "forgetting_gap": 0.1,
                    "forgetting_worst_gap": -5.0,
                    "forward_transfer": 0.8,
                },
            },
            {
                "name": "safety",
                "status": "ok",
                "score": None,
                "metrics": {
                    "safety_planner_ok": True,
                    "constraint_compliance": 0.95,
                    "catastrophic_fail_rate": 0.01,
                },
            },
        ],
    }

    bench._refresh_overall(report)

    assert report["overall"]["gates"]["gate2"] == "fail"
