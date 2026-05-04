import json
import subprocess
import sys
from pathlib import Path


def test_render_trajectory_replay_creates_html(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    trajectory = tmp_path / "trace.jsonl"
    out = tmp_path / "replay.html"
    payload = {
        "schema_version": "trajectory.v0.1",
        "run_id": "test-run",
        "stage": "eval_no_self",
        "episode_index": 0,
        "env_name": "gridworld",
        "scenario_name": "default",
        "total_return": 0.5,
        "length": 1,
        "final_info": {"success": True},
        "steps": [
            {
                "step": 0,
                "obs_patch": [[0, 1], [2, 3]],
                "action": 1,
                "reward": 0.5,
                "done": False,
                "next_obs_patch": [[1, 1], [2, 0]],
                "info": {"got_food": 1, "success": True},
            }
        ],
    }
    trajectory.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "render_trajectory_replay.py"),
            "--trajectory",
            str(trajectory),
            "--out",
            str(out),
        ],
        check=True,
        cwd=repo,
    )

    html = out.read_text(encoding="utf-8")
    assert "Trajectory Replay" in html
    assert "test-run" in html
    assert "got_food" in html
