import json
import subprocess
import sys
from pathlib import Path

import bench


def test_agi_bench_quick_report(tmp_path):
    report_path = tmp_path / "bench.json"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "bench.py",
        "--suite",
        "quick",
        "--seeds",
        "0",
        "--report",
        str(report_path),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "0.2"
    for key in ("meta", "overall", "suites"):
        assert key in data

    run_manifest = data["meta"].get("run_manifest")
    assert isinstance(run_manifest, dict)
    for key in ("config_hash", "seed_list", "seed_count", "git_commit", "suite", "environment"):
        assert key in run_manifest

    gates = data["overall"].get("gates", {})
    for key in ("gate0", "gate1", "gate2", "gate3", "gate4"):
        assert key in gates
    assert "capabilities" in data["overall"]
    assert "confidence" in data["overall"]

    suites = data["suites"]
    assert isinstance(suites, list) and suites

    for suite in suites:
        assert "metrics" in suite
        assert "ci" in suite
        name = suite.get("name")
        expected = bench.SUITE_METRICS_KEYS.get(name, [])
        for metric_key in expected:
            assert metric_key in suite["metrics"]
        assert "per_env" in suite
