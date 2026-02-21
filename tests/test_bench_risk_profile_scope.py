from pathlib import Path

import bench


def _fake_ok_result() -> dict:
    return {
        "stage_metrics": {
            "eval_after_stage4_self": {
                "mean_return": 1.0,
                "test_mean_return": 1.0,
            }
        },
        "config": {},
    }


def test_run_suite_auto_scope_forces_non_safety_baseline(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return _fake_ok_result()

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
        risk_profile_scope="auto",
        risk_head_coef=0.35,
        enable_risk_shield=True,
        use_constrained_rl=True,
    )

    assert len(calls) == 1
    call = calls[0]
    assert float(call["risk_head_coef"]) == 0.10
    assert bool(call["enable_risk_shield"]) is False
    assert bool(call["use_constrained_rl"]) is False


def test_run_suite_all_scope_uses_global_risk_config(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return _fake_ok_result()

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
        risk_profile_scope="all",
        risk_head_coef=0.35,
        enable_risk_shield=True,
        use_constrained_rl=True,
    )

    assert len(calls) == 1
    call = calls[0]
    assert float(call["risk_head_coef"]) == 0.35
    assert bool(call["enable_risk_shield"]) is True
    assert bool(call["use_constrained_rl"]) is True
