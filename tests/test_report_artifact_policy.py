from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_bench_milestone_report_path_and_manifest_policy():
    repo_root = Path(__file__).resolve().parents[1]
    milestone_id = "pytest_artifact_policy"
    report_path = repo_root / "reports" / "milestones" / f"{milestone_id}_quick.json"
    if report_path.exists():
        report_path.unlink()

    cmd = [
        sys.executable,
        "bench.py",
        "--suite",
        "quick",
        "--suites",
        "safety",
        "--seeds",
        "0",
        "--milestone-id",
        milestone_id,
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert report_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data.get("meta", {}).get("artifact_policy") == "milestone"
    run_manifest = data.get("meta", {}).get("run_manifest", {})
    assert isinstance(run_manifest, dict)
    assert run_manifest.get("artifact_policy") == "milestone"

