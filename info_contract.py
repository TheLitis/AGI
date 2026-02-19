"""
Info-contract normalization for environment step outputs.

The contract is intentionally additive/backward-compatible:
- legacy keys (e.g. `reason`) are preserved;
- required contract keys are always materialized.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

INFO_CONTRACT_REQUIRED_FIELDS = (
    "terminated_reason",
    "success",
    "constraint_violation",
    "catastrophic",
    "timeout",
    "reward_env",
    "events",
)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(float(value) > 0.0)
    return bool(default)


def _to_float_event(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_info_contract(
    info: Optional[Mapping[str, Any]],
    *,
    done: bool,
    reward_env: Optional[float] = None,
    terminated_reason: Optional[str] = None,
    success: Optional[bool] = None,
    constraint_violation: Optional[bool] = None,
    catastrophic: Optional[bool] = None,
    timeout: Optional[bool] = None,
    events: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """
    Return a dict that always contains the required info-contract fields.
    """
    out: Dict[str, Any] = dict(info or {})

    reason_val = terminated_reason
    if reason_val is None:
        reason_val = out.get("terminated_reason")
    if reason_val is None:
        reason_val = out.get("reason")
    if reason_val is None or str(reason_val).strip() == "":
        if done:
            reason_val = "done"
        else:
            reason_val = "in_progress"
    reason = str(reason_val)
    out["terminated_reason"] = reason
    # Keep legacy alias for existing code/tests.
    out["reason"] = reason

    if reward_env is None:
        reward_raw = out.get("reward_env", 0.0)
    else:
        reward_raw = reward_env
    try:
        out["reward_env"] = float(reward_raw)
    except Exception:
        out["reward_env"] = 0.0

    if success is None and "success" in out:
        raw_success = out.get("success")
        if raw_success is None:
            success_v: Optional[bool] = None
        else:
            success_v = _to_bool(raw_success)
    else:
        success_v = success
    out["success"] = success_v

    if constraint_violation is None:
        if "constraint_violation" in out:
            constraint_violation = _to_bool(out.get("constraint_violation"))
        else:
            constraint_violation = False
    out["constraint_violation"] = bool(constraint_violation)

    if catastrophic is None:
        if "catastrophic" in out:
            catastrophic = _to_bool(out.get("catastrophic"))
        else:
            catastrophic = False
    out["catastrophic"] = bool(catastrophic)

    if timeout is None:
        if "timeout" in out:
            timeout = _to_bool(out.get("timeout"))
        else:
            timeout = False
    out["timeout"] = bool(timeout)

    out_events: Dict[str, float] = {}
    raw_events = out.get("events")
    if isinstance(raw_events, Mapping):
        for k, v in raw_events.items():
            fv = _to_float_event(v)
            if fv is not None:
                out_events[str(k)] = fv
    if isinstance(events, Mapping):
        for k, v in events.items():
            fv = _to_float_event(v)
            if fv is not None:
                out_events[str(k)] = fv
    for key in (
        "got_food",
        "took_damage",
        "moved",
        "alive",
        "death_flag",
        "other_got_food",
        "instruction_success",
        "social_success",
        "at_target",
        "last_test_passed",
        "pytest_timeout",
        "tests_passed",
        "tests_total",
        "steps_taken",
        "remaining_steps",
        "progress",
    ):
        if key in out and key not in out_events:
            fv = _to_float_event(out.get(key))
            if fv is not None:
                out_events[key] = fv
    out["events"] = out_events
    return out

