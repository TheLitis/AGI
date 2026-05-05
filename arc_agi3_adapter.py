"""ARC-AGI-3 integration helpers.

This module is intentionally optional: importing it does not require the
official ARC-AGI SDK to be installed. Runtime entry points raise a clear error
when the dependency is missing, while conversion utilities remain testable with
plain Python frame-like objects.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from interface_adapters import ObsPacket


try:  # pragma: no cover - availability is environment-dependent.
    import arc_agi  # type: ignore
    from arc_agi import OperationMode  # type: ignore
    from arcengine import GameAction, GameState  # type: ignore

    ARC_AGI3_AVAILABLE = True
except Exception:  # pragma: no cover - exercised by optional-dep tests indirectly.
    arc_agi = None  # type: ignore
    OperationMode = None  # type: ignore
    GameAction = None  # type: ignore
    GameState = None  # type: ignore
    ARC_AGI3_AVAILABLE = False


@dataclass
class ArcAgi3StepTrace:
    step: int
    action_id: int
    action_name: str
    state: str
    levels_completed: int
    win_levels: int
    available_actions: List[int]
    token_count: int


@dataclass
class ArcAgi3EpisodeResult:
    schema_version: str
    game_id: str
    seed: int
    mode: str
    policy: str
    max_steps: int
    steps: int
    final_state: str
    levels_completed: int
    win_levels: int
    score: Optional[float]
    scorecard_id: Optional[str]
    started_at: float
    ended_at: float
    traces: List[ArcAgi3StepTrace]
    error: Optional[str] = None


def _require_arc_agi3() -> None:
    if not ARC_AGI3_AVAILABLE:
        raise RuntimeError("ARC-AGI-3 SDK is not installed. Install with: python -m pip install arc-agi")


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value)


def _frame_layers(frame_data: Any) -> List[np.ndarray]:
    layers = getattr(frame_data, "frame", None)
    if layers is None:
        layers = []
    out: List[np.ndarray] = []
    for layer in layers:
        try:
            arr = np.asarray(layer, dtype=np.int32)
        except Exception:
            continue
        if arr.size == 0:
            continue
        out.append(arr)
    return out


def available_action_ids(frame_or_env: Any) -> List[int]:
    """Extract legal ARC action ids from a frame or environment wrapper."""
    raw = getattr(frame_or_env, "available_actions", None)
    if raw is None:
        raw = getattr(frame_or_env, "action_space", None)
    if raw is None:
        raw = []
    ids: List[int] = []
    for action in raw:
        value = getattr(action, "value", action)
        try:
            ids.append(int(value))
        except Exception:
            continue
    return sorted(set(ids))


def arc_frame_to_obspacket(
    frame_data: Any,
    *,
    episode_id: str = "",
    step_id: int = 0,
    split: str = "eval",
) -> ObsPacket:
    """Convert ARC-AGI-3 frame data into the project's universal ObsPacket."""
    layers = _frame_layers(frame_data)
    token_chunks: List[np.ndarray] = []
    type_chunks: List[np.ndarray] = []
    for layer_idx, layer in enumerate(layers):
        flat = layer.reshape(-1).astype(np.int32, copy=False)
        token_chunks.append(flat)
        type_chunks.append(np.full(flat.shape, int(layer_idx), dtype=np.int8))
    if token_chunks:
        tokens = np.concatenate(token_chunks).astype(np.int32, copy=False)
        token_types = np.concatenate(type_chunks).astype(np.int8, copy=False)
    else:
        tokens = np.zeros((0,), dtype=np.int32)
        token_types = np.zeros((0,), dtype=np.int8)
    token_mask = np.ones(tokens.shape, dtype=np.bool_)

    action_ids = available_action_ids(frame_data)
    action_mask: Optional[np.ndarray]
    if action_ids:
        size = max(action_ids) + 1
        action_mask = np.zeros((size,), dtype=np.bool_)
        for action_id in action_ids:
            if action_id >= 0:
                action_mask[action_id] = True
    else:
        action_mask = None

    levels_completed = int(getattr(frame_data, "levels_completed", 0) or 0)
    win_levels = int(getattr(frame_data, "win_levels", 0) or 0)
    state = _enum_name(getattr(frame_data, "state", "UNKNOWN"))
    dense = np.array([float(levels_completed), float(win_levels), float(len(layers))], dtype=np.float32)
    events = {
        "levels_completed": float(levels_completed),
        "win_levels": float(win_levels),
        "is_win": 1.0 if state == "WIN" else 0.0,
        "is_game_over": 1.0 if state == "GAME_OVER" else 0.0,
    }
    text = {
        "game_id": str(getattr(frame_data, "game_id", "") or ""),
        "state": state,
    }
    return ObsPacket(
        episode_id=str(episode_id),
        step_id=int(step_id),
        env_id=0,
        env_family="arc_agi3",
        scenario_id=0,
        split=str(split),
        tokens=tokens,
        token_types=token_types,
        token_mask=token_mask,
        dense=dense,
        action_mask=action_mask,
        events=events,
        text=text,
    )


def action_id_to_game_action(action_id: int) -> Any:
    """Map an integer id to the SDK GameAction enum."""
    _require_arc_agi3()
    return GameAction.from_id(int(action_id))  # type: ignore[union-attr]


def choose_action_id(
    frame_data: Any,
    *,
    policy: str = "random",
    rng: Optional[random.Random] = None,
) -> int:
    """Minimal legal-action policy used for smoke and baseline runs."""
    ids = [x for x in available_action_ids(frame_data) if int(x) != 0]
    if not ids:
        ids = available_action_ids(frame_data) or [0]
    policy_norm = str(policy or "random").strip().lower()
    if policy_norm == "first":
        return int(ids[0])
    if policy_norm == "last":
        return int(ids[-1])
    rnd = rng or random.Random()
    return int(rnd.choice(ids))


def _scorecard_score(scorecard: Any) -> Optional[float]:
    if scorecard is None:
        return None
    for key in ("score", "score_normalized", "game_score"):
        value = getattr(scorecard, key, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    if isinstance(scorecard, Mapping):
        value = scorecard.get("score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def make_arcade(
    *,
    mode: str = "normal",
    environments_dir: str = "environment_files",
    recordings_dir: str = "reports/arc_agi3/recordings",
) -> Any:
    """Create the official Arcade client with a string operation mode."""
    _require_arc_agi3()
    mode_norm = str(mode or "normal").strip().lower()
    op_mode = OperationMode(mode_norm)  # type: ignore[operator]
    return arc_agi.Arcade(  # type: ignore[union-attr]
        operation_mode=op_mode,
        environments_dir=str(environments_dir),
        recordings_dir=str(recordings_dir),
    )


def list_arc_agi3_games(*, mode: str = "normal", environments_dir: str = "environment_files") -> List[str]:
    arcade = make_arcade(mode=mode, environments_dir=environments_dir)
    games: List[str] = []
    for env_info in arcade.get_environments():
        game_id = getattr(env_info, "game_id", None)
        if game_id:
            games.append(str(game_id))
    return sorted(set(games))


def run_arc_agi3_episode(
    *,
    game_id: str,
    seed: int = 0,
    max_steps: int = 200,
    mode: str = "normal",
    policy: str = "random",
    render_mode: Optional[str] = None,
    save_recording: bool = True,
    environments_dir: str = "environment_files",
    recordings_dir: str = "reports/arc_agi3/recordings",
    make_retries: int = 2,
) -> ArcAgi3EpisodeResult:
    """Run a bounded ARC-AGI-3 episode and return a JSON-serializable result."""
    started = time.time()
    traces: List[ArcAgi3StepTrace] = []
    rng = random.Random(int(seed))
    final_frame: Any = None
    score: Optional[float] = None
    scorecard_id: Optional[str] = None
    error: Optional[str] = None

    try:
        arcade = make_arcade(mode=mode, environments_dir=environments_dir, recordings_dir=recordings_dir)
        scorecard_id = arcade.open_scorecard(tags=["agi-project", "arc-agi-3-smoke"])
        env = None
        last_make_error: Optional[str] = None
        for attempt in range(max(1, int(make_retries))):
            try:
                env = arcade.make(
                    str(game_id),
                    seed=int(seed),
                    scorecard_id=scorecard_id,
                    save_recording=bool(save_recording),
                    include_frame_data=True,
                    render_mode=render_mode,
                )
                if env is not None:
                    break
                last_make_error = "arcade.make returned None"
            except Exception as make_exc:
                last_make_error = str(make_exc)
            if attempt + 1 < max(1, int(make_retries)):
                time.sleep(1.0 + float(attempt))
        if env is None:
            raise RuntimeError(f"ARC-AGI-3 game not available: {game_id}; {last_make_error or 'unknown error'}")
        final_frame = getattr(env, "observation_space", None)
        if final_frame is None:
            final_frame = env.reset()

        for step in range(int(max_steps)):
            packet = arc_frame_to_obspacket(final_frame, episode_id=str(scorecard_id or ""), step_id=step)
            action_id = choose_action_id(final_frame, policy=policy, rng=rng)
            action = action_id_to_game_action(action_id)
            final_frame = env.step(action, reasoning={"policy": policy, "token_count": int(packet.tokens.size)})
            if final_frame is None:
                break
            traces.append(
                ArcAgi3StepTrace(
                    step=int(step),
                    action_id=int(action_id),
                    action_name=str(getattr(action, "name", action_id)),
                    state=_enum_name(getattr(final_frame, "state", "UNKNOWN")),
                    levels_completed=int(getattr(final_frame, "levels_completed", 0) or 0),
                    win_levels=int(getattr(final_frame, "win_levels", 0) or 0),
                    available_actions=available_action_ids(final_frame),
                    token_count=int(packet.tokens.size),
                )
            )
            state = _enum_name(getattr(final_frame, "state", ""))
            if state in {"WIN", "GAME_OVER"}:
                break
        scorecard = arcade.close_scorecard(scorecard_id)
        score = _scorecard_score(scorecard)
    except Exception as exc:
        error = str(exc)

    ended = time.time()
    final_state = _enum_name(getattr(final_frame, "state", "UNKNOWN"))
    return ArcAgi3EpisodeResult(
        schema_version="arc_agi3_run.v0.1",
        game_id=str(game_id),
        seed=int(seed),
        mode=str(mode),
        policy=str(policy),
        max_steps=int(max_steps),
        steps=int(len(traces)),
        final_state=final_state,
        levels_completed=int(getattr(final_frame, "levels_completed", 0) or 0),
        win_levels=int(getattr(final_frame, "win_levels", 0) or 0),
        score=score,
        scorecard_id=scorecard_id,
        started_at=float(started),
        ended_at=float(ended),
        traces=traces,
        error=error,
    )


def write_arc_agi3_report(results: Sequence[ArcAgi3EpisodeResult], output: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "arc_agi3_report.v0.1",
        "created_at": time.time(),
        "episodes": [
            {
                **asdict(result),
                "traces": [asdict(trace) for trace in result.traces],
            }
            for result in results
        ],
    }
    scores = [result.score for result in results if isinstance(result.score, (int, float))]
    wins = [result for result in results if result.final_state == "WIN"]
    payload["summary"] = {
        "n": int(len(results)),
        "success_rate": float(len(wins) / len(results)) if results else None,
        "mean_score": float(np.mean(scores)) if scores else None,
        "errors": int(sum(1 for result in results if result.error)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return payload
