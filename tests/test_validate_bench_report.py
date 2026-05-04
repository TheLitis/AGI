import json
import subprocess
import sys
from pathlib import Path


def _base_report() -> dict:
    return {
        "schema_version": "0.2",
        "meta": {
            "run_manifest": {
                "config_hash": "abc123",
                "seed_list": [0, 1, 2],
                "seed_count": 3,
                "git_commit": "deadbeef",
                "suite": "agi_v1",
                "environment": {"platform": "win32"},
            }
        },
        "overall": {
            "gates": {"gate0": "pass", "gate1": "pass", "gate2": "pass", "gate3": "fail", "gate4": "fail"},
            "capabilities": {
                "generalization_score": 0.8,
                "sample_efficiency_score": 0.8,
                "robustness_score": 0.8,
                "tool_workflow_score": 0.8,
            },
            "confidence": 0.7,
        },
        "suites": [
            {"name": "long_horizon", "ci": None, "metrics": {}, "per_env": []},
            {"name": "core", "ci": None, "metrics": {}, "per_env": []},
            {"name": "tools", "ci": None, "metrics": {}, "per_env": []},
            {"name": "language", "ci": None, "metrics": {}, "per_env": []},
            {"name": "social", "ci": None, "metrics": {}, "per_env": []},
            {"name": "lifelong", "ci": None, "metrics": {}, "per_env": []},
            {
                "name": "safety",
                "ci": None,
                "metrics": {
                    "constraint_compliance": 0.9,
                    "catastrophic_fail_rate": 0.01,
                },
                "per_env": [],
            },
        ],
    }


def _strict_report() -> dict:
    report = _base_report()
    report["meta"]["seed_list"] = [0, 1, 2, 3, 4]
    report["meta"]["run_manifest"]["seed_list"] = [0, 1, 2, 3, 4]
    report["meta"]["run_manifest"]["seed_count"] = 5
    report["overall"]["gates"] = {
        "gate0": "pass",
        "gate1": "pass",
        "gate2": "pass",
        "gate3": "fail",
        "gate4": "fail",
    }
    report["suites"] = [
        {
            "name": "long_horizon",
            "status": "ok",
            "score": 0.70,
            "ci": None,
            "metrics": {"horizon_utilization": 0.80},
            "per_env": [],
        },
        {
            "name": "core",
            "status": "ok",
            "score": 0.95,
            "ci": None,
            "metrics": {"mean_return": 10.0, "test_mean_return": 9.0},
            "per_env": [],
        },
        {
            "name": "tools",
            "status": "ok",
            "score": 0.90,
            "ci": None,
            "metrics": {
                "pass_rate_unmasked": 0.90,
                "mean_steps_to_pass_unmasked": 8.0,
                "invalid_action_rate": 0.0,
                "recovery_rate": 1.0,
            },
            "per_env": [],
        },
        {
            "name": "language",
            "status": "ok",
            "score": 0.75,
            "ci": None,
            "metrics": {"pass_rate": 0.75, "causal_drop": 0.05},
            "per_env": [],
        },
        {
            "name": "social",
            "status": "ok",
            "score": 0.80,
            "ci": None,
            "metrics": {"success_rate": 0.80, "transfer_rate": 0.75},
            "per_env": [],
        },
        {
            "name": "lifelong",
            "status": "ok",
            "score": 0.60,
            "ci": None,
            "metrics": {"forgetting_gap": 0.0, "forward_transfer": 0.8, "env_family_coverage": 1.0},
            "per_env": [],
        },
        {
            "name": "safety",
            "status": "ok",
            "score": 0.95,
            "ci": None,
            "metrics": {"constraint_compliance": 0.90, "catastrophic_fail_rate": 0.01},
            "per_env": [],
        },
        {
            "name": "safety_ood",
            "status": "ok",
            "score": 0.95,
            "ci": None,
            "metrics": {"constraint_compliance": 0.90, "catastrophic_fail_rate": 0.01},
            "per_env": [],
        },
    ]
    return report


def test_validate_bench_report_passes(tmp_path):
    report_path = tmp_path / "bench_ok.json"
    report_path.write_text(json.dumps(_base_report()), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "validate_bench_report.py",
        "--report",
        str(report_path),
        "--expect-gate",
        "gate2=pass",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK]" in result.stdout


def test_validate_bench_report_fails_on_missing_gate(tmp_path):
    bad = _base_report()
    bad["overall"]["gates"].pop("gate4")
    report_path = tmp_path / "bench_bad.json"
    report_path.write_text(json.dumps(bad), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "validate_bench_report.py", "--report", str(report_path)]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing key 'gate4'" in result.stdout


def test_validate_bench_report_fails_when_safety_metric_missing(tmp_path):
    bad = _base_report()
    safety = next(s for s in bad["suites"] if s.get("name") == "safety")
    safety["metrics"].pop("catastrophic_fail_rate")
    report_path = tmp_path / "bench_bad_safety.json"
    report_path.write_text(json.dumps(bad), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "validate_bench_report.py", "--report", str(report_path)]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "safety suite metrics missing key 'catastrophic_fail_rate'" in result.stdout


def test_strict_acceptance_requires_safety_ood(tmp_path):
    report = _strict_report()
    report["suites"] = [s for s in report["suites"] if s.get("name") != "safety_ood"]
    report_path = tmp_path / "bench_missing_safety_ood.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "validate_bench_report.py", "--report", str(report_path), "--strict-acceptance"]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing required suites: safety_ood" in result.stdout


def test_strict_acceptance_recomputes_gate_values(tmp_path):
    report = _strict_report()
    tools = next(s for s in report["suites"] if s.get("name") == "tools")
    tools["metrics"]["pass_rate_unmasked"] = 0.0
    report_path = tmp_path / "bench_gate_lie.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "validate_bench_report.py", "--report", str(report_path), "--strict-acceptance"]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "overall.gates.gate1 disagrees with recomputed value" in result.stdout
    assert "overall.gates.gate2 disagrees with recomputed value" in result.stdout


def test_strict_acceptance_rejects_out_of_range_rates(tmp_path):
    report = _strict_report()
    language = next(s for s in report["suites"] if s.get("name") == "language")
    language["metrics"]["pass_rate"] = 1.2
    report_path = tmp_path / "bench_bad_rate.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "validate_bench_report.py", "--report", str(report_path), "--strict-acceptance"]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "language suite metric 'pass_rate' must be in [0,1]" in result.stdout
