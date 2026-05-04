import json
import subprocess
import sys
from pathlib import Path


def test_visualize_bench_dashboard_smoke(tmp_path):
    report = {
        "schema_version": "0.2",
        "overall": {
            "agi_score": 0.5,
            "confidence": 0.7,
            "gates": {"gate0": "pass", "gate1": "pass", "gate2": "fail", "gate3": "na", "gate4": "na"},
            "capabilities": {
                "generalization_score": 0.6,
                "sample_efficiency_score": 0.5,
                "robustness_score": 0.4,
                "tool_workflow_score": 0.3,
            },
        },
        "suites": [
            {
                "name": "lifelong",
                "status": "running",
                "score": None,
                "metrics": {"forward_transfer": None},
                "per_env": [],
                "run_cache": [
                    {
                        "seed": 0,
                        "status": "ok",
                        "case": {"name": "lifelong_gridworld"},
                        "result": {
                            "stage_metrics": {
                                "lifelong_eval": {
                                    "lifelong_per_chapter": [
                                        {
                                            "regime": "R1",
                                            "mean_return": 12.0,
                                            "mean_damage": 1.0,
                                            "trait_change_norm": 0.2,
                                            "scenario_counts": {"balanced": 3},
                                            "planner_debug": {"planner_override_rate": 0.4},
                                        }
                                    ]
                                }
                            }
                        },
                    }
                ],
            }
        ],
    }
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "dashboard.html"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/visualize_bench_dashboard.py",
            "--report",
            str(report_path),
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    html = out_path.read_text(encoding="utf-8")
    assert "AGI Bench Dashboard" in html
    assert "lifelong" in html
    assert "Lifelong Chapter Board" in html
    assert "planner_override_rate" not in html
    assert "R1" in html
