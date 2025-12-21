"""
Checkpoint utilities for saving/loading agent (and optional optimizer) state.

The project runs end-to-end experiments, but without persistence it's hard to:
  - reuse a trained agent for evaluation,
  - resume long runs,
  - compare capability regressions over time.

This module provides a minimal, forward-compatible checkpoint format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


CHECKPOINT_VERSION = 1

PathLike = Union[str, Path]


def _agent_device(agent: torch.nn.Module) -> torch.device:
    try:
        return next(agent.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def save_checkpoint(
    path: PathLike,
    agent: torch.nn.Module,
    *,
    trainer: Optional[Any] = None,
    save_optim: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "meta": {"version": CHECKPOINT_VERSION, "kind": "proto_creature"},
        "agent_state_dict": agent.state_dict(),
    }

    if save_optim:
        for key in ("optim_world", "optim_policy", "optim_self"):
            opt = getattr(agent, key, None)
            if opt is not None and hasattr(opt, "state_dict"):
                payload[f"{key}_state_dict"] = opt.state_dict()
        if trainer is not None:
            opt = getattr(trainer, "lifelong_optimizer", None)
            if opt is not None and hasattr(opt, "state_dict"):
                payload["lifelong_optimizer_state_dict"] = opt.state_dict()

    if trainer is not None:
        payload["trainer_state"] = {
            "meta_conflict_ma": float(getattr(trainer, "meta_conflict_ma", 0.0) or 0.0),
            "meta_uncertainty_ma": float(getattr(trainer, "meta_uncertainty_ma", 0.0) or 0.0),
        }

    if extra:
        payload["extra"] = dict(extra)

    torch.save(payload, out_path)
    return out_path


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def load_checkpoint(
    path: PathLike,
    agent: torch.nn.Module,
    *,
    trainer: Optional[Any] = None,
    load_optim: bool = True,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    device = _agent_device(agent)
    map_loc = map_location if map_location is not None else device
    payload = torch.load(in_path, map_location=map_loc)

    agent_state = payload.get("agent_state_dict", None)
    if agent_state is None:
        raise ValueError(f"Invalid checkpoint (missing agent_state_dict): {in_path}")
    agent.load_state_dict(agent_state, strict=bool(strict))

    if load_optim:
        for key in ("optim_world", "optim_policy", "optim_self"):
            opt_state = payload.get(f"{key}_state_dict")
            opt = getattr(agent, key, None)
            if opt_state is not None and opt is not None:
                opt.load_state_dict(opt_state)
                _move_optimizer_state_to_device(opt, device)
        opt_state = payload.get("lifelong_optimizer_state_dict")
        if opt_state is not None and trainer is not None:
            opt = getattr(trainer, "lifelong_optimizer", None)
            if opt is not None:
                opt.load_state_dict(opt_state)
                _move_optimizer_state_to_device(opt, device)

    return {
        "meta": payload.get("meta", {}),
        "trainer_state": payload.get("trainer_state", {}),
        "extra": payload.get("extra", {}),
        "path": str(in_path),
    }

