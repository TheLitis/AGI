import json
from pathlib import Path

from scripts import check_gate_regression


def _write_report(path: Path, *, safety: dict, safety_ood: dict, tools: dict) -> Path:
    payload = {
        "schema_version": "0.2",
        "suites": [
            {"name": "safety", "metrics": safety},
            {"name": "safety_ood", "metrics": safety_ood},
            {"name": "tools", "metrics": tools},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_check_gate_regression_passes_when_candidate_stays_within_limits(monkeypatch, tmp_path):
    baseline = _write_report(
        tmp_path / "baseline.json",
        safety={"constraint_compliance": 0.90, "catastrophic_fail_rate": 0.01},
        safety_ood={"constraint_compliance": 0.92, "catastrophic_fail_rate": 0.02},
        tools={
            "pass_rate_unmasked": 0.95,
            "recovery_rate": 0.90,
            "invalid_action_rate": 0.03,
            "mean_steps_to_pass_unmasked": 4.0,
        },
    )
    candidate = _write_report(
        tmp_path / "candidate.json",
        safety={"constraint_compliance": 0.88, "catastrophic_fail_rate": 0.03},
        safety_ood={"constraint_compliance": 0.89, "catastrophic_fail_rate": 0.03},
        tools={
            "pass_rate_unmasked": 0.92,
            "recovery_rate": 0.88,
            "invalid_action_rate": 0.05,
            "mean_steps_to_pass_unmasked": 5.5,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["check_gate_regression.py", "--baseline", str(baseline), "--candidate", str(candidate)],
    )
    assert check_gate_regression.main() == 0


def test_check_gate_regression_blocks_when_tools_and_safety_regress_too_far(monkeypatch, tmp_path):
    baseline = _write_report(
        tmp_path / "baseline.json",
        safety={"constraint_compliance": 0.90, "catastrophic_fail_rate": 0.01},
        safety_ood={"constraint_compliance": 0.92, "catastrophic_fail_rate": 0.02},
        tools={
            "pass_rate_unmasked": 0.95,
            "recovery_rate": 0.90,
            "invalid_action_rate": 0.03,
            "mean_steps_to_pass_unmasked": 4.0,
        },
    )
    candidate = _write_report(
        tmp_path / "candidate.json",
        safety={"constraint_compliance": 0.80, "catastrophic_fail_rate": 0.09},
        safety_ood={"constraint_compliance": 0.82, "catastrophic_fail_rate": 0.10},
        tools={
            "pass_rate_unmasked": 0.80,
            "recovery_rate": 0.78,
            "invalid_action_rate": 0.12,
            "mean_steps_to_pass_unmasked": 7.5,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["check_gate_regression.py", "--baseline", str(baseline), "--candidate", str(candidate)],
    )
    assert check_gate_regression.main() == 1
