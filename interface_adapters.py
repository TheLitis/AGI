"""
Lightweight v2 adapter layer.

This module introduces additive adapters without breaking existing int-action
and dict-observation paths used by the current pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass
class ObsPacket:
    episode_id: str
    step_id: int
    env_id: int
    env_family: str
    scenario_id: int
    split: str
    tokens: np.ndarray
    token_types: np.ndarray
    token_mask: np.ndarray
    dense: Optional[np.ndarray]
    action_mask: Optional[np.ndarray]
    events: Dict[str, float]
    text: Dict[str, str]


@dataclass
class ToolCallEnvelope:
    kind: str = "primitive"
    primitive_id: Optional[int] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    skill_id: Optional[int] = None
    skill_args: Optional[Dict[str, Any]] = None


def obs_to_packet(
    obs: Mapping[str, Any],
    *,
    episode_id: str = "",
    step_id: int = 0,
    split: str = "train",
    action_mask: Optional[np.ndarray] = None,
    events: Optional[Mapping[str, float]] = None,
) -> ObsPacket:
    patch_raw = obs.get("patch")
    if isinstance(patch_raw, np.ndarray):
        patch = patch_raw.astype(np.int32, copy=False)
        tokens = patch.reshape(-1)
    else:
        tokens = np.zeros((0,), dtype=np.int32)
    token_types = np.zeros(tokens.shape, dtype=np.int8)
    token_mask = np.ones(tokens.shape, dtype=np.bool_)
    energy = obs.get("energy")
    dense: Optional[np.ndarray]
    if isinstance(energy, (int, float)):
        dense = np.array([float(energy)], dtype=np.float32)
    else:
        dense = None
    text: Dict[str, str] = {}
    for key in ("env_name", "scenario_name", "instruction", "task_desc", "description"):
        val = obs.get(key)
        if val is not None:
            text[key] = str(val)
    packet_events: Dict[str, float] = {}
    if isinstance(events, Mapping):
        for k, v in events.items():
            if isinstance(v, (int, float, bool)):
                packet_events[str(k)] = float(v)
    return ObsPacket(
        episode_id=str(episode_id),
        step_id=int(step_id),
        env_id=int(obs.get("env_id", 0) or 0),
        env_family=str(obs.get("env_family", "") or ""),
        scenario_id=int(obs.get("scenario_id", 0) or 0),
        split=str(split),
        tokens=tokens.astype(np.int32, copy=False),
        token_types=token_types,
        token_mask=token_mask,
        dense=dense,
        action_mask=np.asarray(action_mask, dtype=np.bool_) if action_mask is not None else None,
        events=packet_events,
        text=text,
    )


def packet_to_obs(packet: ObsPacket, *, patch_shape: tuple[int, int] = (5, 5)) -> Dict[str, Any]:
    need = int(max(1, patch_shape[0]) * max(1, patch_shape[1]))
    tokens = np.asarray(packet.tokens, dtype=np.int32).reshape(-1)
    if tokens.size < need:
        padded = np.zeros((need,), dtype=np.int32)
        if tokens.size > 0:
            padded[: tokens.size] = tokens
        tokens = padded
    patch = tokens[:need].reshape(patch_shape).astype(np.int64, copy=False)
    energy = 0.0
    if packet.dense is not None and len(packet.dense) > 0:
        energy = float(packet.dense[0])
    obs: Dict[str, Any] = {
        "patch": patch,
        "energy": float(energy),
        "scenario_id": int(packet.scenario_id),
        "env_id": int(packet.env_id),
        "env_family": str(packet.env_family),
    }
    for key, value in packet.text.items():
        obs[str(key)] = str(value)
    return obs


def int_action_to_tool_call(action: int) -> ToolCallEnvelope:
    return ToolCallEnvelope(kind="primitive", primitive_id=int(action))


def tool_call_to_int_action(
    action: Any,
    *,
    tool_name_to_action: Optional[Mapping[str, int]] = None,
    default_action: int = 0,
) -> int:
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, ToolCallEnvelope):
        if action.kind == "primitive" and action.primitive_id is not None:
            return int(action.primitive_id)
        if action.kind == "tool_call" and action.tool_name and tool_name_to_action:
            return int(tool_name_to_action.get(str(action.tool_name), int(default_action)))
        return int(default_action)
    if isinstance(action, Mapping):
        kind = str(action.get("kind", "") or "").strip().lower()
        if kind == "primitive" and action.get("primitive_id") is not None:
            try:
                return int(action.get("primitive_id"))
            except Exception:
                return int(default_action)
        if kind == "tool_call":
            name = str(action.get("tool_name", "") or "").strip()
            if tool_name_to_action and name:
                return int(tool_name_to_action.get(name, int(default_action)))
            return int(default_action)
    return int(default_action)

