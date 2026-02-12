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
            {"name": "core", "ci": None, "metrics": {}, "per_env": []},
            {"name": "tools", "ci": None, "metrics": {}, "per_env": []},
            {"name": "language", "ci": None, "metrics": {}, "per_env": []},
            {"name": "social", "ci": None, "metrics": {}, "per_env": []},
            {"name": "lifelong", "ci": None, "metrics": {}, "per_env": []},
            {"name": "safety", "ci": None, "metrics": {}, "per_env": []},
        ],
    }


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
