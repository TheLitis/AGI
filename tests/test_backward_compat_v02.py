from __future__ import annotations

from validate_bench_report import DEFAULT_REQUIRED_SUITES, validate_report


def _suite(name: str, metrics: dict | None = None) -> dict:
    return {
        "name": name,
        "status": "ok",
        "score": 0.5,
        "ci": {"n": 1, "mean": 0.5, "std": 0.0, "se": 0.0, "half_width": 0.0, "lower": 0.5, "upper": 0.5},
        "metrics": metrics or {},
        "per_env": [],
        "notes": [],
    }


def test_schema_v02_backward_compatible_with_extra_safety_ood_suite():
    suites = [_suite(name) for name in DEFAULT_REQUIRED_SUITES]
    # Keep required safety metrics for validator.
    for s in suites:
        if s["name"] == "safety":
            s["metrics"] = {"constraint_compliance": 0.9, "catastrophic_fail_rate": 0.05}
    suites.append(
        _suite(
            "safety_ood",
            metrics={"constraint_compliance": 0.85, "catastrophic_fail_rate": 0.10},
        )
    )

    report = {
        "schema_version": "0.2",
        "meta": {
            "timestamp": 1,
            "git_commit": "abc",
            "seed_list": [0],
            "suite": "agi_v1",
            "ood": False,
            "quick": True,
            "config": {},
            "run_manifest": {
                "config_hash": "x",
                "seed_list": [0],
                "seed_count": 1,
                "git_commit": "abc",
                "suite": "agi_v1",
                "environment": {},
            },
        },
        "overall": {
            "agi_score": 0.5,
            "notes": [],
            "gates": {"gate0": "pass", "gate1": "fail", "gate2": "fail", "gate3": "fail", "gate4": "fail"},
            "capabilities": {
                "generalization_score": 0.5,
                "sample_efficiency_score": 0.5,
                "robustness_score": 0.5,
                "tool_workflow_score": 0.5,
            },
            "confidence": 0.5,
        },
        "suites": suites,
    }

    errors = validate_report(
        report,
        require_schema="0.2",
        required_suites=list(DEFAULT_REQUIRED_SUITES),
        expected_gates={},
    )
    assert errors == []

