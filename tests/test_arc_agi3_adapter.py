from types import SimpleNamespace

import numpy as np
import pytest

from arc_agi3_adapter import (
    ArcAgi3EpisodeResult,
    ArcAgi3StepTrace,
    arc_frame_to_obspacket,
    available_action_ids,
    choose_action_id,
    write_arc_agi3_report,
)


def _fake_frame(**overrides):
    data = {
        "game_id": "zz00",
        "state": SimpleNamespace(name="NOT_FINISHED"),
        "levels_completed": 1,
        "win_levels": 3,
        "available_actions": [1, 3, 2, 2],
        "frame": [
            np.array([[0, 1], [2, 3]], dtype=np.int32),
            np.array([[4, 5], [6, 7]], dtype=np.int32),
        ],
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_arc_frame_to_obspacket_preserves_layers_and_action_mask():
    packet = arc_frame_to_obspacket(_fake_frame(), episode_id="ep", step_id=4)

    assert packet.episode_id == "ep"
    assert packet.step_id == 4
    assert packet.env_family == "arc_agi3"
    assert packet.tokens.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert packet.token_types.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert packet.action_mask is not None
    assert packet.action_mask.tolist() == [False, True, True, True]
    assert packet.events["levels_completed"] == 1.0
    assert packet.text["game_id"] == "zz00"


def test_available_action_ids_supports_env_action_space_shape():
    env = SimpleNamespace(action_space=[SimpleNamespace(value=3), SimpleNamespace(value=1)])

    assert available_action_ids(env) == [1, 3]


def test_choose_action_id_uses_legal_non_reset_actions():
    frame = _fake_frame(available_actions=[0, 2, 5])

    assert choose_action_id(frame, policy="first") == 2
    assert choose_action_id(frame, policy="last") == 5


def test_write_arc_agi3_report(tmp_path):
    result = ArcAgi3EpisodeResult(
        schema_version="arc_agi3_run.v0.1",
        game_id="zz00",
        seed=0,
        mode="offline",
        policy="first",
        max_steps=10,
        steps=1,
        final_state="WIN",
        levels_completed=1,
        win_levels=1,
        score=1.0,
        scorecard_id="card",
        started_at=1.0,
        ended_at=2.0,
        traces=[
            ArcAgi3StepTrace(
                step=0,
                action_id=1,
                action_name="ACTION1",
                state="WIN",
                levels_completed=1,
                win_levels=1,
                available_actions=[1],
                token_count=4,
            )
        ],
    )
    out = tmp_path / "arc.json"

    payload = write_arc_agi3_report([result], out)

    assert out.exists()
    assert payload["summary"]["success_rate"] == pytest.approx(1.0)
    assert payload["episodes"][0]["traces"][0]["action_name"] == "ACTION1"
