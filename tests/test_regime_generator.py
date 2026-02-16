import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regime_generator import RegimeGenerator, RegimeGeneratorConfig
from trainer import RegimeConfig


def _stats(
    avg_return: float,
    forgetting_gap: float,
    uncertainty: float,
    retain_delta: float = None,
    episodes_seen: int = 10,
):
    if retain_delta is None:
        retain_delta = -float(forgetting_gap)
    return {
        "regime_name": "dummy",
        "episodes_seen": episodes_seen,
        "avg_return": avg_return,
        "avg_survival": 0.5,
        "avg_food": 0.3,
        "avg_damage": 0.1,
        "retain_delta": retain_delta,
        "forgetting_gap": forgetting_gap,
        "uncertainty": uncertainty,
    }


def test_prioritizes_forgetting_and_penalizes_easy_regimes():
    cfg = RegimeGeneratorConfig()
    gen = RegimeGenerator(cfg, all_env_descriptors=[{"id": 0}, {"id": 1}])
    current = {
        "easy": RegimeConfig("easy", {"balanced": 1.0}),
        "forget": RegimeConfig("forget", {"balanced": 1.0}),
    }
    stats = {
        "easy": _stats(avg_return=0.95, forgetting_gap=0.0, uncertainty=0.1),
        "forget": _stats(avg_return=0.55, forgetting_gap=0.25, uncertainty=0.2),
    }
    proposals = gen.propose_regimes(current, stats)
    priorities = {p.name: p.priority for p in proposals if p.name in current}
    assert priorities["forget"] > priorities["easy"]


def test_respects_max_active_regimes():
    cfg = RegimeGeneratorConfig()
    cfg.max_active_regimes = 2
    gen = RegimeGenerator(cfg, all_env_descriptors=[{"id": 0}, {"id": 1}])
    current = {
        "r1": RegimeConfig("r1", {"balanced": 1.0}),
        "r2": RegimeConfig("r2", {"balanced": 1.0}),
    }
    proposals = gen.propose_regimes(current, {})
    new_names = [p.name for p in proposals if p.name not in current]
    assert len(new_names) == 0


def test_proposes_new_regime_when_capacity_allows():
    cfg = RegimeGeneratorConfig()
    cfg.max_active_regimes = 3
    gen = RegimeGenerator(cfg, all_env_descriptors=[{"id": 0}, {"id": 1}])
    current = {
        "r1": RegimeConfig("r1", {"balanced": 1.0}),
    }
    proposals = gen.propose_regimes(current, {})
    assert any(p.name.startswith("regime_new") for p in proposals)


def test_retain_delta_negative_is_treated_as_forgetting():
    cfg = RegimeGeneratorConfig()
    gen = RegimeGenerator(cfg, all_env_descriptors=[{"id": 0}, {"id": 1}])
    current = {
        "stable": RegimeConfig("stable", {"balanced": 1.0}),
        "forgetting": RegimeConfig("forgetting", {"balanced": 1.0}),
    }
    stats = {
        "stable": _stats(avg_return=0.7, forgetting_gap=0.0, retain_delta=0.0, uncertainty=0.1),
        # Legacy forgetting_gap is intentionally non-positive here to validate
        # that retain_delta (primary convention) drives prioritization.
        "forgetting": _stats(avg_return=0.7, forgetting_gap=-0.2, retain_delta=-0.3, uncertainty=0.1),
    }
    proposals = gen.propose_regimes(current, stats)
    priorities = {p.name: p.priority for p in proposals if p.name in current}
    assert priorities["forgetting"] > priorities["stable"]
