from __future__ import annotations

from pathlib import Path

import bench


def _fake_result() -> dict:
    return {
        "stage_metrics": {
            "eval_after_stage4_self": {
                "mean_return": 1.0,
                "test_mean_return": 1.0,
                "repo_pass_rate": 1.0,
                "unmasked": {"repo_pass_rate": 1.0, "repo_steps_to_pass": [5]},
            },
            "lifelong_eval": {
                "lifelong_forgetting_R1_gap": 0.0,
                "lifelong_adaptation_R2_delta": 0.6,
                "lifelong_adaptation_R3_delta": 0.4,
            },
        },
        "config": {},
    }


def test_quick_tools_profile_uses_gate2_tuned_budget(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return _fake_result()

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)
    suite = bench.SuiteSpec(
        name="tools",
        cases=[bench.BenchCase(name="repo_toolloop", env_type="repo", repo_scenarios=["train:proc_mixed_loop"])],
        implemented=True,
    )
    report = {"meta": {"config": {}}, "suites": []}

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
        report_path=Path(tmp_path) / "report.json",
    )

    assert len(calls) == 1
    call = calls[0]
    assert int(call["repo_bc_pretrain_episodes"]) == 192
    assert float(call["repo_online_bc_coef"]) == 1.0
    assert float(call["action_mask_dropout_prob"]) == 0.0
    assert int(call["action_mask_dropout_warmup_updates"]) == 8
    assert float(call["planning_coef"]) == 0.0
    assert int(call["stage4_updates"]) == 8


def test_quick_long_horizon_profile_uses_extended_planning_budget(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return _fake_result()

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)
    suite = bench.SuiteSpec(
        name="long_horizon",
        cases=[bench.BenchCase(name="long_grid", env_type="gridworld", max_steps_env=120, max_energy_env=160)],
        implemented=True,
    )
    report = {"meta": {"config": {}}, "suites": []}

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
        report_path=Path(tmp_path) / "report.json",
    )

    assert len(calls) == 1
    call = calls[0]
    assert int(call["n_steps"]) == 384
    assert int(call["planning_horizon"]) == 20
    assert int(call["planner_rollouts"]) == 6
    assert int(call["stage2_updates"]) == 6
    assert int(call["stage4_updates"]) == 10


def test_quick_lifelong_profile_uses_balanced_replay(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return _fake_result()

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)
    suite = bench.SuiteSpec(
        name="lifelong",
        cases=[bench.BenchCase(name="life_grid", env_type="gridworld", max_energy_env=80)],
        implemented=True,
    )
    report = {"meta": {"config": {}}, "suites": []}

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
        report_path=Path(tmp_path) / "report.json",
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["mode"] == "lifelong"
    assert call["eval_policy"] == "greedy"
    assert float(call["replay_frac_current"]) == 0.5
    assert int(call["lifelong_episodes_per_chapter"]) == 36
    assert int(call["stage4_updates"]) == 5


def test_quick_core_profile_uses_stable_budget(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(dict(kwargs))
        return _fake_result()

    monkeypatch.setattr(bench, "run_experiment", fake_run_experiment)
    suite = bench.SuiteSpec(
        name="core",
        cases=[bench.BenchCase(name="core_grid", env_type="gridworld")],
        implemented=True,
    )
    report = {"meta": {"config": {}}, "suites": []}

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
        report_path=Path(tmp_path) / "report.json",
    )

    assert len(calls) == 1
    call = calls[0]
    assert int(call["n_steps"]) == 256
    assert int(call["stage4_updates"]) == 5
