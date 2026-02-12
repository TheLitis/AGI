from pathlib import Path

import bench


def test_run_suite_skips_minigrid_when_optional_dep_missing(monkeypatch, tmp_path):
    def fake_run_experiment(**kwargs):
        if kwargs.get("env_type") == "minigrid":
            raise ModuleNotFoundError("No module named 'pygame'")
        return {
            "stage_metrics": {
                "eval_after_stage4_self": {
                    "mean_return": 12.0,
                    "test_mean_return": 8.0,
                }
            },
            "config": {},
        }

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)

    suite = bench.SuiteSpec(
        name="core",
        cases=[
            bench.BenchCase(name="grid", env_type="gridworld"),
            bench.BenchCase(name="mini", env_type="minigrid", minigrid_scenarios=["minigrid-empty"]),
        ],
        implemented=True,
    )
    report = {
        "meta": {"config": {}},
        "suites": [],
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
    per_env = {entry["env"]: entry for entry in result["per_env"]}
    assert per_env["gridworld/basic"]["status"] == "ok"
    assert per_env["minigrid/minigrid-empty"]["status"] == "skipped"
    assert any("skipped_optional_dependency:pygame" in note for note in result["notes"])
