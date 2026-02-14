from pathlib import Path

import bench


def test_run_suite_quick_skips_tools_basic_case(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(str(kwargs.get("env_type")))
        return {
            "stage_metrics": {
                "eval_after_stage4_self": {
                    "repo_pass_rate": 1.0,
                    "repo_steps_to_pass": [5],
                    "unmasked": {
                        "repo_pass_rate": 1.0,
                        "repo_steps_to_pass": [7],
                    },
                }
            },
            "config": {},
        }

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)

    suite = bench.SuiteSpec(
        name="tools",
        cases=[
            bench.BenchCase(name="tools_basic", env_type="tools"),
            bench.BenchCase(name="repo_toolloop", env_type="repo", repo_scenarios=["train:proc_mixed_loop", "test:proc_mixed_loop"]),
        ],
        implemented=True,
    )
    report = {"meta": {"config": {}}, "suites": []}
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

    assert calls == ["repo"]
    per_env = {entry["env"]: entry for entry in result["per_env"]}
    assert per_env["tool_env/basic"]["status"] == "skipped"
    assert per_env["repo_tool_env/train:proc_mixed_loop,test:proc_mixed_loop"]["status"] == "ok"
    assert "quick_skip_tools_basic_case" in result["notes"]
