from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
import random
import math

if TYPE_CHECKING:  # pragma: no cover
    from trainer import RegimeConfig, RewardProfile, RegimeStats


@dataclass
class RegimeProposal:
    name: str
    env_descriptors: List[Any]
    reward_profile: Optional["RewardProfile"]
    priority: float


class RegimeGeneratorConfig:
    # hyperparameters for auto-curriculum
    target_avg_return_low: float = 0.2
    target_avg_return_high: float = 0.8
    forgetting_weight: float = 1.0
    uncertainty_weight: float = 1.0
    novelty_weight: float = 0.5
    max_active_regimes: int = 5
    easy_gap_tolerance: float = 0.05
    safety_priority_penalty: float = 1.0
    danger_priority_penalty: float = 1.0
    safety_utility_threshold: float = 0.0
    max_dangerous_regime_fraction: float = 0.4
    alpha_forgetting: float = 0.02
    alpha_easy: float = 0.01
    target_avg_return_high: float = 0.8
    target_return_for_curriculum: float = 0.7


class RegimeGenerator:
    def __init__(self, config: RegimeGeneratorConfig, all_env_descriptors: List[Any]):
        self.config = config
        self.all_env_descriptors = list(all_env_descriptors)
        self._used_descriptor_ids: Set[int] = set()
        self.probabilities: Dict[str, float] = {}

    @staticmethod
    def _forgetting_loss(stats: "RegimeStats") -> float:
        """
        Canonical forgetting magnitude:
        - retain_delta (current - baseline), negative means forgetting.
        - legacy fallback forgetting_gap (baseline - current), positive means forgetting.
        """
        if not isinstance(stats, dict):
            return 0.0
        retain_delta = stats.get("retain_delta")
        if isinstance(retain_delta, (int, float)):
            try:
                if math.isfinite(float(retain_delta)):
                    return float(max(0.0, -float(retain_delta)))
            except Exception:
                pass
        forgetting_gap = stats.get("forgetting_gap")
        if isinstance(forgetting_gap, (int, float)):
            try:
                if math.isfinite(float(forgetting_gap)):
                    return float(max(0.0, float(forgetting_gap)))
            except Exception:
                pass
        return 0.0

    def _score_existing(self, stats: "RegimeStats") -> float:
        c = self.config
        priority = 0.0
        forgetting_loss = self._forgetting_loss(stats)
        if stats["avg_return"] > c.target_avg_return_high and forgetting_loss < c.easy_gap_tolerance:
            priority -= 1.0  # too easy, de-prioritize
        if stats["avg_return"] < c.target_avg_return_low:
            priority -= 0.5
        priority += c.forgetting_weight * forgetting_loss
        priority += c.uncertainty_weight * max(stats["uncertainty"], 0.0)
        if c.target_avg_return_low <= stats["avg_return"] <= c.target_avg_return_high and forgetting_loss > 0:
            priority += 0.5  # golden middle with forgetting -> boost
        return priority

    def _propose_new(self, existing: Dict[str, "RegimeConfig"], regime_stats: Dict[str, "RegimeStats"]) -> List[RegimeProposal]:
        proposals: List[RegimeProposal] = []
        if len(existing) >= getattr(self.config, "max_active_regimes", len(existing)):
            return proposals

        # pick env descriptors that have not been used yet; fall back to random
        available_indices = [i for i in range(len(self.all_env_descriptors)) if i not in self._used_descriptor_ids]
        if not available_indices:
            available_indices = list(range(len(self.all_env_descriptors)))
        if not available_indices:
            return proposals

        # novelty guided by highest uncertainty regime if present
        target_idx = available_indices[0]
        if regime_stats:
            max_unc_reg = max(regime_stats.values(), key=lambda s: s.get("uncertainty", 0.0))
            target_idx = available_indices[max(0, min(len(available_indices) - 1, int(max_unc_reg["uncertainty"] * len(available_indices))))]
        else:
            target_idx = random.choice(available_indices)

        desc = self.all_env_descriptors[target_idx]
        self._used_descriptor_ids.add(target_idx)
        name = f"regime_new_{len(self._used_descriptor_ids)}"
        proposals.append(
            RegimeProposal(
                name=name,
                env_descriptors=[desc],
                reward_profile=None,
                priority=self.config.novelty_weight,
            )
        )
        return proposals

    def propose_regimes(
        self,
        current_regimes: Dict[str, "RegimeConfig"],
        regime_stats: Dict[str, "RegimeStats"],
    ) -> List[RegimeProposal]:
        """
        Based on current regimes and their statistics proposes:
        - which regimes to boost or down-weight,
        - which new regimes to add.
        """
        proposals: List[RegimeProposal] = []
        danger_threshold = getattr(self.config, "safety_utility_threshold", 0.0)
        dangerous_fraction = 0.0
        if regime_stats:
            dangerous_count = sum(1 for s in regime_stats.values() if s.get("danger_score", 0.0) > danger_threshold)
            dangerous_fraction = dangerous_count / max(1, len(regime_stats))
        for name, cfg in current_regimes.items():
            stats = regime_stats.get(name)
            priority = 0.0
            weight = self.probabilities.get(name, 1.0)
            if stats is not None:
                base_priority = self._score_existing(stats)
                danger_score = float(stats.get("danger_score", 0.0))
                safety_utility = float(stats.get("avg_safety_utility", 0.0))
                safety_deficit = max(0.0, getattr(self.config, "safety_utility_threshold", 0.0) - safety_utility)
                priority = (
                    base_priority
                    - getattr(self.config, "safety_priority_penalty", 1.0) * safety_deficit
                    - getattr(self.config, "danger_priority_penalty", 1.0) * danger_score
                )
                if danger_score > danger_threshold and dangerous_fraction > getattr(
                    self.config, "max_dangerous_regime_fraction", 1.0
                ):
                    priority = min(priority, base_priority, 0.0)
                gap_k = float(self._forgetting_loss(stats))
                avg_ret = float(stats.get("avg_return", 0.0))
                target_ret = getattr(self.config, "target_return_for_curriculum", 0.7)
                alpha_f = getattr(self.config, "alpha_forgetting", 0.02)
                alpha_easy = getattr(self.config, "alpha_easy", 0.01)
                weight = weight * math.exp(alpha_f * max(0.0, gap_k) - alpha_easy * max(0.0, avg_ret - target_ret))
            self.probabilities[name] = weight
            if weight > 0:
                priority += math.log(weight)
            proposals.append(
                RegimeProposal(
                    name=name,
                    env_descriptors=self.all_env_descriptors,
                    reward_profile=cfg.reward_profile,
                    priority=priority,
                )
            )

        proposals.extend(self._propose_new(current_regimes, regime_stats))
        return proposals
