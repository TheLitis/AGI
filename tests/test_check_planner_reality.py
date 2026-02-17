import json
from pathlib import Path

from scripts import check_planner_reality


def _write_report(path: Path, *, suite_name: str, metrics: dict) -> Path:
    payload = {
        "schema_version": "0.2",
        "suites": [
            {
                "name": suite_name,
                "metrics": metrics,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_check_planner_reality_passes_when_thresholds_met(monkeypatch, tmp_path):
    report = _write_report(
        tmp_path / "report.json",
        suite_name="planning_diag",
        metrics={
            "planner_reality_steps": 300,
            "planner_score_nstep_corr": 0.20,
            "policy_score_nstep_corr": 0.05,
            "planner_score_corr_advantage": 0.15,
            "planner_top1_advantage_nstep": 0.08,
            "planner_regret_proxy_nstep": 0.08,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["check_planner_reality.py", "--report", str(report), "--min-steps", "200"],
    )
    assert check_planner_reality.main() == 0


def test_check_planner_reality_blocks_when_missing_signal(monkeypatch, tmp_path):
    report = _write_report(
        tmp_path / "report.json",
        suite_name="long_horizon",
        metrics={
            "planner_reality_steps": 50,
            "planner_score_nstep_corr": 0.01,
            "policy_score_nstep_corr": 0.02,
            "planner_top1_advantage_nstep": -0.01,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["check_planner_reality.py", "--report", str(report), "--min-steps", "200"],
    )
    assert check_planner_reality.main() == 1
