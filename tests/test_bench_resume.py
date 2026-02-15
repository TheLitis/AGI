import json
import sys
from pathlib import Path

import bench


def test_run_suite_replaces_existing_suite_entry(tmp_path):
    suite = bench.SuiteSpec(name="safety", cases=[], implemented=True)
    report = {
        "schema_version": "0.2",
        "meta": {"config": {}},
        "overall": {},
        "suites": [
            {
                "name": "safety",
                "status": "error",
                "score": None,
                "ci": None,
                "metrics": {},
                "per_env": [],
                "notes": ["old_entry"],
            }
        ],
    }
    report_path = Path(tmp_path) / "report.json"

    result = bench._run_suite(
        suite,
        seeds=[0],
        variants=["full"],
        mode="stage4",
        quick=True,
        quick_stub=False,
        log_dir=str(tmp_path / "logs"),
        use_skills=False,
        skill_mode="handcrafted",
        n_latent_skills=0,
        masked_only=False,
        unmasked_only=False,
        eval_max_steps=64,
        force_cpu=True,
        auto_force_cpu_repo=True,
        report=report,
        report_path=report_path,
    )

    assert result["status"] == "ok"
    same_name = [x for x in report["suites"] if x.get("name") == "safety"]
    assert len(same_name) == 1
    assert same_name[0]["status"] == "ok"


def test_main_resume_skips_completed_suite(tmp_path, monkeypatch):
    report_path = Path(tmp_path) / "resume_report.json"
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs.get("env_type"))
        return {
            "stage_metrics": {
                "eval_after_stage4_self": {
                    "mean_return": 0.5,
                    "constraint_compliance": 0.9,
                    "catastrophic_fail_rate": 0.1,
                }
            },
            "config": {},
        }

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--suite",
            "agi_v1",
            "--quick",
            "--suites",
            "safety",
            "--report",
            str(report_path),
        ],
    )
    assert bench.main() == 0

    first = json.loads(report_path.read_text(encoding="utf-8"))
    first_safety = [x for x in first.get("suites", []) if x.get("name") == "safety"]
    assert len(first_safety) == 1
    assert first_safety[0]["status"] == "ok"
    assert calls == ["gridworld", "gridworld"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--suite",
            "agi_v1",
            "--quick",
            "--suites",
            "safety",
            "--resume",
            "--report",
            str(report_path),
        ],
    )
    assert bench.main() == 0

    second = json.loads(report_path.read_text(encoding="utf-8"))
    second_safety = [x for x in second.get("suites", []) if x.get("name") == "safety"]
    assert len(second_safety) == 1
    assert second_safety[0]["status"] == "ok"
    assert calls == ["gridworld", "gridworld"]
