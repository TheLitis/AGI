from __future__ import annotations

import bench
from trainer import Trainer


def test_bench_safety_metrics_use_contract_fields_only():
    compliance, catastrophic = bench._safety_metrics_from_eval(
        {
            "constraint_compliance": 0.9,
            "catastrophic_fail_rate": 0.1,
            "death_rate": 1.0,  # should be ignored when contract metrics exist
        }
    )
    assert compliance == 0.9
    assert catastrophic == 0.1


def test_bench_safety_metrics_do_not_fallback_to_death_rate():
    compliance, catastrophic = bench._safety_metrics_from_eval({"death_rate": 0.2})
    assert compliance is None
    assert catastrophic is None


def test_trainer_extract_contract_flags_prefers_contract_booleans():
    flags = Trainer._extract_contract_flags(
        {
            "terminated_reason": "max_steps",
            "constraint_violation": True,
            "catastrophic": False,
            "timeout": True,
            "reward_env": 0.0,
            "events": {},
        },
        done=True,
    )
    assert flags["constraint_violation"] is True
    assert flags["catastrophic"] is False
    assert flags["timeout"] is True
    assert flags["terminated_reason"] == "max_steps"

