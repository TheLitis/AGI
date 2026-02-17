import json
import sys
from pathlib import Path

import bench


def test_run_suite_forwards_planner_safety_tuning_kwargs(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return {
            "stage_metrics": {
                "eval_after_stage4_self": {
                    "mean_return": 1.0,
                    "test_mean_return": 1.0,
                }
            },
            "config": {},
        }

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)

    suite = bench.SuiteSpec(
        name="core",
        cases=[bench.BenchCase(name="core_grid", env_type="gridworld")],
        implemented=True,
    )
    report = {"meta": {"config": {}}, "suites": []}
    report_path = Path(tmp_path) / "report.json"

    bench._run_suite(
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
        planner_world_reward_blend=0.30,
        safety_penalty_coef=2.0,
        safety_threshold=0.05,
    )

    assert len(calls) == 1
    call = calls[0]
    assert float(call["planner_world_reward_blend"]) == 0.30
    assert float(call["safety_penalty_coef"]) == 2.0
    assert float(call["safety_threshold"]) == 0.05


def test_main_records_planner_safety_tuning_in_meta_config(monkeypatch, tmp_path):
    report_path = Path(tmp_path) / "bench_report.json"

    def fake_run_experiment(**kwargs):
        return {
            "stage_metrics": {
                "eval_after_stage4_self": {
                    "mean_return": 1.0,
                    "test_mean_return": 1.0,
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
            "core",
            "--quick",
            "--report",
            str(report_path),
            "--planner-world-reward-blend",
            "0.5",
            "--safety-penalty-coef",
            "2.0",
            "--safety-threshold",
            "0.05",
        ],
    )
    assert bench.main() == 0

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cfg = payload.get("meta", {}).get("config", {})
    assert float(cfg["planner_world_reward_blend"]) == 0.5
    assert float(cfg["safety_penalty_coef"]) == 2.0
    assert float(cfg["safety_threshold"]) == 0.05
    assert payload.get("meta", {}).get("run_manifest", {}).get("config_hash")
