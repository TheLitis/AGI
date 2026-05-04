import json
from types import SimpleNamespace

import numpy as np

from trainer import Trainer


def test_trajectory_export_writes_jsonl(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.trajectory_export_dir = tmp_path
    trainer.trajectory_max_episodes = 1
    trainer.trajectory_exported_episodes = 0
    trainer.trajectory_export_error_count = 0
    trainer.logger = SimpleNamespace(run_id="trace/run:1")

    trainer._export_trajectory_episode(
        stage="eval_no_self",
        episode_index=0,
        use_self=False,
        planning_coef=0.0,
        env_name="gridworld",
        scenario_name="default",
        env_id=0,
        scenario_id=0,
        total_return=1.25,
        length=1,
        steps=[
            {
                "step": 0,
                "obs_patch": [[0, 1], [2, 3]],
                "energy": 10.0,
                "action": 2,
                "reward": 1.25,
                "done": True,
                "next_obs_patch": [[1, 1], [2, 0]],
                "next_energy": 9.0,
                "info": {"success": True, "events": {"got_food": 1.0}},
            }
        ],
        final_info={"success": True, "constraint_violation": False, "events": {"got_food": 1.0}},
    )

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["schema_version"] == "trajectory.v0.1"
    assert payload["total_return"] == 1.25
    assert payload["steps"][0]["action"] == 2
    assert trainer.trajectory_exported_episodes == 1
    assert trainer.trajectory_export_error_count == 0


def test_compact_patch_for_trace_keeps_small_grids():
    patch = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert Trainer._compact_patch_for_trace(patch) == [[1, 2], [3, 4]]


def test_compact_patch_for_trace_summarizes_large_arrays():
    patch = np.zeros((32, 32), dtype=np.float32)
    assert Trainer._compact_patch_for_trace(patch) == {"shape": [32, 32], "dtype": "float32"}
