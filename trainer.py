"""
Trainer and helpers for the 4-stage proto-creature pipeline:
1) Random experience + world model pretrain
2) A2C without self-model
3) Self-model offline training + self-reflection on traits
4) A2C with self-model + planner

Also includes lifecycle evaluation phases A/B/C (train/test env splits).
Includes small stability fixes and optional instruction-conditioned perception.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, TypedDict
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
import random
import statistics
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env import GridWorldEnv
from agent import ProtoCreatureAgent
from memory import ReplayBuffer, Transition
from models import traits_to_preference_weights
from text_utils import hash_text_to_ids
from skills import (
    Skill,
    LatentSkill,
    SkillContext,
    SkillDemoTransition,
    SkillDemoBatch,
    SkillLibrary,
    flatten_skill_context,
    get_default_skills,
)
from regime_generator import RegimeGenerator, RegimeGeneratorConfig, RegimeProposal

logger = logging.getLogger(__name__)

# =========================
#  Trainer
# =========================


class ExperimentLogger:
    """Minimal JSONL logger for scalar metrics per stage."""
    def __init__(self, run_id: str, logdir: str = "logs"):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.logdir / f"{self.run_id}.jsonl"
        self._fh = open(self.filepath, "a", encoding="utf-8")

    def log_scalar(self, stage: str, metric: str, value: float, **extra):
        if self._fh is None:
            return
        record = {
            "timestamp": time.time(),
            "stage": stage,
            "metric": metric,
            "value": float(value),
        }
        if extra:
            record.update(extra)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log(self, record: Dict[str, Any]):
        """
        Append an arbitrary JSON-serializable record (auto-adds timestamp).
        """
        if self._fh is None:
            return
        payload = {"timestamp": time.time()}
        payload.update(record)
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class RegimeConfig:
    """Lightweight description of a regime for lifelong evaluation."""
    name: str
    scenario_weights: Dict[str, float]
    reward_profile: Optional[Dict[str, float]] = None
    description: str = ""


# Reward profile alias for clarity/typing.
RewardProfile = Dict[str, float]


@dataclass
class SelfModelProbeStats:
    corr_return: float
    corr_return_defined: bool
    corr_survival: float
    corr_survival_defined: bool
    mean_true_return: float
    mean_pred_return: float
    mean_true_survival: float
    mean_pred_survival: float
    num_samples: int

    def good_enough_for_reflection(self, cfg: "SelfReflectionSafetyConfig") -> bool:
        """
        Decide whether probe quality is sufficient to run trait reflection.
        """
        if self.num_samples < cfg.min_probe_episodes:
            return False
        if abs(self.corr_return) < cfg.min_return_corr:
            return False
        if self.corr_survival_defined and abs(self.corr_survival) < cfg.min_survival_corr:
            return False
        return True


@dataclass
class SelfReflectionSafetyConfig:
    min_corr_return_for_reflection: float = 0.20
    min_corr_survival_for_reflection: float = 0.10
    reflection_lr_scale_when_low_quality: float = 0.1
    skip_reflection_if_very_low_quality: bool = True
    very_low_corr_threshold: float = 0.0
    max_trait_delta_norm_stage3b: float = 0.3
    max_trait_delta_norm_phaseC: float = 0.5
    forbid_survival_weight_decrease: bool = True
    forbid_danger_aversion_decrease: bool = True
    survival_epsilon_down: float = 0.0
    danger_aversion_epsilon_down: float = 0.0
    min_probe_episodes: int = 2
    min_return_corr: float = 0.20
    min_survival_corr: float = 0.10
    std_eps: float = 1e-6


@dataclass
class LatentSkillTrainingConfig:
    demos_per_skill: int = 2000
    max_steps_per_episode: int = 64
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    shuffle_buffer_size: int = 10000


class RegimeStats(TypedDict):
    regime_name: str
    episodes_seen: int
    avg_return: float
    avg_survival: float
    avg_food: float
    avg_damage: float
    forgetting_gap: float  # baseline_return - current_return
    uncertainty: float     # mean SelfModel uncertainty for this regime
    avg_safety_utility: float  # utility of the safety faction
    danger_score: float        # aggregate measure of regime danger


def get_faction_preference_weights(agent) -> torch.Tensor:
    """
    Возвращает preference weights per faction для текущих traits:
    shape: [num_factions, num_components]
    """
    traits = agent.traits  # [F, D]
    weights = traits_to_preference_weights(traits)  # [F, C]
    if weights.dim() == 1:
        weights = weights.view(1, -1)
    return weights


class Trainer:
    """Coordinator for staged training, planning, and lifecycle evaluation."""
    def __init__(
        self,
        env: GridWorldEnv,
        agent: ProtoCreatureAgent,
        buffer_capacity: int = 20000,
        device: Optional[torch.device] = None,
        planning_horizon: int = 12,
        planner_gamma: float = 0.99,
        planner_mode: str = "rollout",  # "repeat" (v1) или "rollout" (multi-rollout)
        planner_rollouts: int = 4,
        safety_threshold: float = 0.0,
        safety_penalty_coef: float = 1.0,
        action_mask_internalization_coef: float = 0.10,
        action_mask_dropout_prob: float = 0.0,
        action_mask_prediction_coef: float = 0.10,
        repo_online_bc_coef: float = 0.10,
        train_env_ids: Optional[list] = None,
        test_env_ids: Optional[list] = None,
        env_descriptors: Optional[torch.Tensor] = None,
        logger: Optional[ExperimentLogger] = None,
        use_skills: bool = False,
        skills: Optional[List[Skill]] = None,
        regime_aware_replay: bool = False,
        replay_frac_current: float = 0.5,
        skill_mode: str = "handcrafted",
        n_latent_skills: int = 0,
    ):
        self.env = env
        self.env_family: str = "gridworld"
        envs_list = getattr(env, "envs", None)
        if envs_list is not None:
            self.has_computer_env = any(
                "computer" in str(getattr(e, "env_family", "")).lower() for e in envs_list
            )
            for e in envs_list:
                name = getattr(e, "env_name", "")
                module_name = getattr(e.__class__, "__module__", "")
                env_family_attr = getattr(e, "env_family", "")
                if isinstance(env_family_attr, str) and env_family_attr.lower() == "tools":
                    self.env_family = "tools"
                    break
                if (
                    (isinstance(name, str) and name.startswith("MiniGrid-"))
                    or (isinstance(env_family_attr, str) and env_family_attr.lower() == "minigrid")
                    or ("minigrid_env" in str(module_name))
                ):
                    self.env_family = "minigrid"
                    break
        else:
            self.has_computer_env = "computer" in str(getattr(env, "env_family", "")).lower()
            name = getattr(env, "env_name", "")
            module_name = getattr(env.__class__, "__module__", "")
            env_family_attr = getattr(env, "env_family", "")
            if isinstance(env_family_attr, str) and env_family_attr.lower() == "tools":
                self.env_family = "tools"
            if (
                (isinstance(name, str) and name.startswith("MiniGrid-"))
                or (isinstance(env_family_attr, str) and env_family_attr.lower() == "minigrid")
                or ("minigrid_env" in str(module_name))
            ):
                self.env_family = "minigrid"
            if isinstance(env_family_attr, str) and "computer" in env_family_attr.lower():
                self.env_family = "computer"
        self.is_minigrid = self.env_family == "minigrid"
        self.is_computer = self.env_family == "computer"
        self.agent = agent
        self.use_skills = bool(use_skills)
        self.skills: List[Skill] = list(skills) if skills is not None else (get_default_skills() if self.use_skills else [])
        self.n_skills = len(self.skills)
        self.skill_mode = (skill_mode or "handcrafted").lower()
        self.n_latent_skills = int(n_latent_skills if self.skill_mode in {"latent", "mixed"} else 0)
        self.skill_library: Optional[SkillLibrary] = getattr(self.agent, "skill_library", None)
        if self.skill_library is not None:
            self.n_latent_skills = len(self.skill_library)
        self.total_skills = self.n_skills + (self.n_latent_skills if self.use_skills else 0)
        self.fast_params_initial = [p.detach().clone() for p in agent.get_fast_params()]
        self.device = device or agent.device
        self.env_max_steps_by_id = self._compute_env_max_steps_by_id()
        self.text_token_table = self._compute_text_token_table()
        self.buffer = ReplayBuffer(buffer_capacity)
        self.lifelong_buffer = ReplayBuffer(buffer_capacity)
        self.lifelong_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.agent.policy.parameters())
                    + list(self.agent.value_model.parameters())
                    + list(self.agent.workspace.parameters()),
                    "lr": 1e-4,
                },
                {"params": list(self.agent.self_model.parameters()), "lr": 1e-4},
            ]
        )
        self.logger = logger
        self.regime_aware_replay = bool(regime_aware_replay)
        try:
            self.replay_frac_current = max(0.0, min(1.0, float(replay_frac_current)))
        except Exception:
            self.replay_frac_current = 0.5
        self.current_regime_name: str = ""
        self.regime_stats: Dict[str, RegimeStats] = {}
        self.regimes: Dict[str, RegimeConfig] = {}
        self.regime_priorities: Dict[str, float] = {}
        self.regime_env_descriptors: Dict[str, Any] = {}
        self.lifelong_trait_memory: Dict[str, torch.Tensor] = {}
        self.lifelong_trait_memory_score: Dict[str, float] = {}
        self.lifelong_reflect_early_step_boost = 2
        self.lifelong_reflect_early_step_size_scale = 1.2
        self.lifelong_reflect_early_lambda_prior_scale = 0.5
        self.lifelong_reflect_late_step_delta = -1
        self.lifelong_reflect_late_step_size_scale = 0.8
        self.lifelong_reflect_late_lambda_prior_scale = 1.5
        self.lifelong_reflect_safety_step_size_scale = 0.8
        self.lifelong_reflect_safety_lambda_prior_scale = 1.5
        self.safety_threshold = float(safety_threshold)
        self.safety_penalty_coef = float(safety_penalty_coef)
        self.safety = SelfReflectionSafetyConfig()
        self.latent_skill_training = LatentSkillTrainingConfig()
        self.last_self_probe: Optional[SelfModelProbeStats] = None
        # When environments expose an action-mask (e.g. tool UIs), we can still train the
        # policy to "internalize" invalid actions by penalizing probability mass outside the mask.
        self.action_mask_internalization_coef = float(action_mask_internalization_coef)
        try:
            self.action_mask_dropout_prob = max(0.0, min(1.0, float(action_mask_dropout_prob)))
        except Exception:
            self.action_mask_dropout_prob = 0.0
        self.action_mask_prediction_coef = max(0.0, float(action_mask_prediction_coef or 0.0))
        # Conservative unmasked bias from predicted action-mask logits:
        # only high-confidence predictions are used, with limited strength.
        # We adapt both threshold and mix from mask-predictor quality to avoid
        # over-constraining unmasked evaluation when the predictor is weak.
        # Keep unmasked transfer biased enough toward learned validity priors so
        # repo/tool loops stay stable across seeds while still requiring confidence.
        self.unmasked_mask_bias_mix = 0.75
        self.unmasked_mask_confidence_threshold = 0.85
        self.unmasked_mask_confidence_threshold_high = 0.90
        self.unmasked_mask_auc_quality_threshold = 0.82
        self.mask_pred_auc_ema = float("nan")
        self.mask_pred_auc_ema_decay = 0.95
        self.repo_online_bc_coef = max(0.0, float(repo_online_bc_coef or 0.0))
        self._trait_safety_ctx: Dict[str, Dict[str, Any]] = {}

        descriptors = None
        if hasattr(env, "get_all_descriptors"):
            try:
                descriptors = env.get_all_descriptors()
            except Exception:
                descriptors = None
        if descriptors is not None:
            try:
                self.regime_generator: Optional[RegimeGenerator] = RegimeGenerator(
                    RegimeGeneratorConfig(),
                    all_env_descriptors=list(descriptors),
                )
            except Exception:
                self.regime_generator = None
        else:
            self.regime_generator = None

        # train/test env splits for reporting (default: all envs -> train)
        if train_env_ids is None:
            if hasattr(env, "n_envs"):
                train_env_ids = list(range(getattr(env, "n_envs", 0)))
            else:
                train_env_ids = [0]
        self.train_env_ids = train_env_ids
        self.test_env_ids = test_env_ids or []

        self.env_descriptors = None
        self.env_desc_dim = None
        if env_descriptors is not None:
            self.env_descriptors = env_descriptors.to(self.device)
            self.env_desc_dim = self.env_descriptors.shape[-1]

        # Structured, in-memory log of trait reflection events (shared with the agent for external access).
        self.trait_reflection_log: List[Dict[str, Any]] = []
        # Trait reflection hyperparameters (shared across offline/online/lifelong).
        self.trait_reflection_mode = "gradient"  # {"gradient", "axis_search"}; gradient is default
        self.trait_reflection_lr = 0.02
        self.trait_reflection_steps_per_batch = 4  # inner steps per reflection batch
        self.trait_reflection_steps = self.trait_reflection_steps_per_batch  # legacy alias
        self.trait_reflection_max_l2_from_anchor = 1.0
        self.trait_reflection_max_global_l2_from_init = 3.0
        self.trait_reflection_min_improvement = 0.0
        self.trait_reflection_reward_normalization = True
        self.trait_reflection_lambda_l2 = 0.05
        self.trait_reflection_grad_clip = 1.0
        self.trait_reflection_use_axis_search = False  # legacy override to force axis-aligned search
        self.trait_reflection_debug: Dict[str, Any] = {}
        # share log with agent for external access
        if hasattr(self.agent, "trait_reflection_log"):
            self.agent.trait_reflection_log = self.trait_reflection_log

        if self.is_minigrid:
            self.trait_reflection_lr = 0.05
            self.trait_reflection_steps_per_batch = max(self.trait_reflection_steps_per_batch, 6)
            self.trait_reflection_steps = self.trait_reflection_steps_per_batch
            self.trait_reflection_max_l2_from_anchor = min(self.trait_reflection_max_l2_from_anchor, 0.25)
            self.trait_reflection_max_global_l2_from_init = max(self.trait_reflection_max_global_l2_from_init, 3.0)
            if self.trait_reflection_min_improvement > -0.002:
                self.trait_reflection_min_improvement = -0.002

        # meta-controller для адаптивного исследования / любопытства
        self.meta_conflict_ma = 0.0
        self.meta_uncertainty_ma = 0.0
        self.meta_beta = 0.02  # чуть более быстрая адаптация
        # Adaptive GAE(lambda) for long-horizon credit assignment.
        self.gae_lambda_base = 0.95
        self.gae_lambda_min = 0.90
        self.gae_lambda_max = 0.99
        self.gae_lambda_adapt_strength = 0.04
        self.gae_lambda_horizon_scale = 24.0
        # optional skill usage stats for analysis/logging
        self.skill_usage_counts: Dict[int, int] = {}

        # параметры многшагового планировщика
        self.planning_horizon = planning_horizon
        self.planner_gamma = planner_gamma
        self.planner_mode = planner_mode
        self.planner_rollouts = planner_rollouts

        # Pipeline overview:
        #   Stage 1: collect_random_experience + train_world_model
        #   Stage 2: train_policy (use_self=False, planner off)
        #   Stage 3: train_self_model_offline (+ optional self_reflect_on_traits)
        #   Stage 3c (optional): refresh SelfModel after reflection (same routine)
        #   Stage 4: train_policy (use_self=True, planner on)
        # Lifecycle eval phases:
        #   A = train envs only, B = train+test, C = test envs only.

    # ----- meta-controller (adaptive exploration) -----

    def _update_meta_stats(self, conflicts: torch.Tensor, uncertainties: torch.Tensor):
        """
        Обновляем скользящие средние по конфликту и неопределённости self-модели.
        Это даёт медленный "mood" агента, который используется для адаптации
        энтропии политики и силы любопытства.
        """
        with torch.no_grad():
            c = float(conflicts.mean().item())
            u = float(uncertainties.mean().item())
            beta = self.meta_beta
            self.meta_conflict_ma = (1.0 - beta) * self.meta_conflict_ma + beta * c
            self.meta_uncertainty_ma = (1.0 - beta) * self.meta_uncertainty_ma + beta * u

    def get_adaptive_entropy_coef(self, base_entropy: float) -> float:
        """
        Чем больше долгосрочный конфликт/неопределённость,
        тем меньше шум в политике (агент сильнее опирается на уже выученную стратегию).
        """
        c = self.meta_conflict_ma
        u = self.meta_uncertainty_ma
        # простое ограниченное инверсное масштабирование
        factor = 1.0 / (1.0 + 0.5 * (c + u))
        factor = max(0.5, min(1.5, factor))
        return base_entropy * factor

    def get_adaptive_curiosity_beta(self, base_beta: float) -> float:
        """
        При высокой долгосрочной неопределённости уменьшаем бонус за "сырое" предсказательное
        любопытство, чтобы агент не переобучался на шум.
        """
        u = self.meta_uncertainty_ma
        factor = 1.0 / (1.0 + 0.5 * u)
        factor = max(0.3, min(1.0, factor))
        return base_beta * factor

    def get_adaptive_planning_coef(self, base_planning_coef: float) -> float:
        """
        Чем более согласованы SelfModel и ValueModel и чем ниже неопределённость,
        тем больше смысла вкладываться в планирование (оно надёжнее).
        """
        c = self.meta_conflict_ma
        u = self.meta_uncertainty_ma

        # конфликт/неопределённость большие → фактор маленький
        x = 1.0 / (1.0 + c + u)  # ∈ (0,1]
        factor = 0.5 + 0.5 * x   # ∈ [0.5,1.0]

        return base_planning_coef * factor

    def get_adaptive_gae_lambda(self, base_gae_lambda: float) -> float:
        """
        Adaptive GAE(lambda) for long-horizon credit assignment.
        - Stable regime (low conflict/uncertainty) + longer horizon -> larger lambda.
        - Unstable regime -> smaller lambda to reduce variance.
        """
        c = max(0.0, float(self.meta_conflict_ma))
        u = max(0.0, float(self.meta_uncertainty_ma))
        horizon = max(1.0, float(getattr(self, "planning_horizon", 1)))
        horizon_scale = max(1.0, float(getattr(self, "gae_lambda_horizon_scale", 24.0) or 24.0))
        strength = max(0.0, float(getattr(self, "gae_lambda_adapt_strength", 0.04) or 0.04))
        lam_min = max(0.0, min(1.0, float(getattr(self, "gae_lambda_min", 0.90) or 0.90)))
        lam_max = max(lam_min, min(1.0, float(getattr(self, "gae_lambda_max", 0.99) or 0.99)))
        base = max(lam_min, min(lam_max, float(base_gae_lambda)))

        quality = 1.0 / (1.0 + c + u)  # in (0, 1]
        quality_centered = 2.0 * quality - 1.0  # in (-1, 1]
        horizon_factor = min(1.0, horizon / horizon_scale)
        lam = base + strength * horizon_factor * quality_centered
        return max(lam_min, min(lam_max, lam))

    def _env_desc_from_ids(self, env_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.env_descriptors is None:
            return None
        # Explicitly validate indices on CPU to avoid CUDA device-side asserts.
        max_id = int(self.env_descriptors.shape[0] - 1)
        env_ids_cpu = env_ids.detach().cpu()
        min_id = int(env_ids_cpu.min().item())
        max_seen = int(env_ids_cpu.max().item())
        if min_id < 0 or max_seen > max_id:
            raise RuntimeError(
                f"env_id out of range in _env_desc_from_ids: "
                f"min={min_id}, max={max_seen}, n_envs={self.env_descriptors.shape[0]}"
            )
        return self.env_descriptors[env_ids]

    def _get_action_mask_tensor(self) -> Optional[torch.Tensor]:
        env = self.env
        if not hasattr(env, "get_action_mask"):
            return None
        try:
            mask = env.get_action_mask()
        except Exception:
            return None
        if mask is None:
            return None
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.ndim == 1:
            mask_arr = mask_arr.reshape(1, -1)
        return torch.from_numpy(mask_arr).to(self.device)

    def _get_action_mask_for_logits(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        mask = self._get_action_mask_tensor()
        if mask is None:
            return None
        if mask.shape[-1] != logits.shape[-1]:
            return None
        if mask.shape[0] != logits.shape[0]:
            if mask.shape[0] == 1:
                mask = mask.expand(logits.shape[0], -1)
            else:
                return None
        if not torch.any(mask):
            return None
        return mask

    def _apply_action_mask(self, logits: torch.Tensor) -> torch.Tensor:
        mask = self._get_action_mask_for_logits(logits)
        if mask is None:
            return logits
        return logits.masked_fill(~mask, -1.0e9)

    def _policy_forward_with_mask(self, G_t: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward policy and (optionally) auxiliary mask head.
        """
        policy = self.agent.policy
        if hasattr(policy, "forward_with_mask"):
            logits, mask_logits = policy.forward_with_mask(G_t)
            return logits, mask_logits
        return policy(G_t), None

    def _apply_predicted_mask_bias(
        self,
        logits: torch.Tensor,
        mask_logits: Optional[torch.Tensor],
        *,
        min_prob: float = 1.0e-3,
        mix: float = 1.0,
        confidence_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Softly bias logits by predicted validity probabilities.
        `mix` controls strength in [0, 1]; 1.0 is full multiplicative prior.
        When `confidence_threshold` is set, only confident predictions are used.
        """
        if mask_logits is None:
            return logits
        mix = max(0.0, min(1.0, float(mix)))
        if mix <= 0.0:
            return logits
        probs = torch.sigmoid(mask_logits).clamp(min=min_prob, max=1.0)
        if confidence_threshold is not None:
            thr = max(0.5, min(1.0, float(confidence_threshold)))
            conf = torch.maximum(probs, 1.0 - probs)
            confident = conf >= thr
            if not bool(torch.any(confident).item()):
                return logits
            probs = torch.where(confident, probs, torch.ones_like(probs))
        blended_probs = ((1.0 - mix) + mix * probs).clamp(min=min_prob, max=1.0)
        return logits + torch.log(blended_probs)

    def _effective_unmasked_mask_confidence_threshold(self) -> float:
        low = float(getattr(self, "unmasked_mask_confidence_threshold", 0.90) or 0.90)
        high = float(getattr(self, "unmasked_mask_confidence_threshold_high", low) or low)
        quality_thr = float(getattr(self, "unmasked_mask_auc_quality_threshold", 0.84) or 0.84)
        try:
            auc = float(getattr(self, "mask_pred_auc_ema", float("nan")))
        except Exception:
            auc = float("nan")
        if math.isfinite(auc) and auc < quality_thr:
            return max(0.5, min(1.0, low))
        return max(0.5, min(1.0, high))

    def _effective_unmasked_mask_bias_mix(self) -> float:
        base = float(getattr(self, "unmasked_mask_bias_mix", 0.0) or 0.0)
        base = max(0.0, min(1.0, base))
        if base <= 0.0:
            return 0.0
        quality_thr = float(getattr(self, "unmasked_mask_auc_quality_threshold", 0.84) or 0.84)
        quality_thr = max(1.0e-6, quality_thr)
        try:
            auc = float(getattr(self, "mask_pred_auc_ema", float("nan")))
        except Exception:
            auc = float("nan")
        if not math.isfinite(auc):
            return base
        if auc >= quality_thr:
            return base
        ratio = max(0.0, min(1.0, auc / quality_thr))
        # Keep a small floor so confident predictions can still help when quality is moderate.
        ratio = max(0.25, ratio)
        return base * ratio

    def _compose_policy_logits_with_masks(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_logits_pred: Optional[torch.Tensor],
        *,
        apply_hard_mask: bool = True,
    ) -> torch.Tensor:
        """
        Compose final policy logits using optional hard action mask and/or learned
        mask predictor.
        """
        has_invalid = bool(mask is not None and bool(torch.any(~mask).item()))
        if apply_hard_mask and has_invalid:
            return logits.masked_fill(~mask, -1.0e9)
        if (
            mask_logits_pred is not None
            and float(getattr(self, "action_mask_prediction_coef", 0.0) or 0.0) > 0.0
        ):
            if has_invalid:
                # Oracle mask has invalid actions: use full prior from predicted mask logits.
                return self._apply_predicted_mask_bias(logits, mask_logits_pred, mix=1.0)
            # Unmasked mode: apply a conservative, confidence-gated prior only.
            unmasked_mix = self._effective_unmasked_mask_bias_mix()
            conf_thr = self._effective_unmasked_mask_confidence_threshold()
            return self._apply_predicted_mask_bias(
                logits,
                mask_logits_pred,
                mix=unmasked_mix,
                confidence_threshold=conf_thr,
            )
        return logits

    def _get_active_env_instance(self) -> Any:
        """
        Return currently active concrete env (unwrap EnvPool when possible).
        """
        env_obj = self.env
        envs_list = getattr(env_obj, "envs", None)
        active_idx = getattr(env_obj, "active_env_idx", None)
        if envs_list is not None and active_idx is not None:
            try:
                idx = int(active_idx)
                if 0 <= idx < len(envs_list):
                    return envs_list[idx]
            except Exception:
                pass
        return env_obj

    def _get_repo_expert_action(self) -> Optional[int]:
        env_inst = self._get_active_env_instance()
        expert_fn = getattr(env_inst, "get_expert_action", None)
        if not callable(expert_fn):
            return None
        try:
            return int(expert_fn())
        except Exception:
            return None

    def _compute_env_max_steps_by_id(self) -> torch.Tensor:
        """
        Build a stable per-env-id max_steps table (float32 tensor on self.device).

        This is used to normalize survival targets in SelfModel training. We avoid
        using `EnvPool.max_steps` directly because it may represent the *minimum*
        across envs, which can be smaller than sampled sequence lengths and lead
        to survival targets > 1.0 (crashes BCE on CUDA).
        """
        envs_list = getattr(self.env, "envs", None)
        if not envs_list:
            ms = int(getattr(self.env, "max_steps", 1) or 1)
            ms = max(1, ms)
            return torch.tensor([float(ms)], device=self.device, dtype=torch.float32)

        max_steps: List[int] = []
        for env in envs_list:
            ms: Optional[int] = None
            task_set = getattr(env, "task_set", None)
            tasks = getattr(task_set, "tasks", None) if task_set is not None else None
            if tasks:
                try:
                    ms = max(int(getattr(t, "max_steps", 0) or 0) for t in tasks)
                except Exception:
                    ms = None
            if ms is None:
                try:
                    ms = int(getattr(env, "max_steps", 0) or 0)
                except Exception:
                    ms = 0
            if ms <= 0:
                ms = int(getattr(self.env, "max_steps", 1) or 1)
            max_steps.append(max(1, int(ms)))

        return torch.tensor([float(x) for x in max_steps], device=self.device, dtype=torch.float32)

    def _env_max_steps_from_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Map env_id tensor -> max_steps (float) with CPU-side bounds checking.
        """
        table = getattr(self, "env_max_steps_by_id", None)
        if table is None:
            ms = float(max(1, int(getattr(self.env, "max_steps", 1) or 1)))
            return torch.full_like(env_ids, ms, dtype=torch.float32, device=self.device)

        max_id = int(table.shape[0] - 1)
        env_ids_cpu = env_ids.detach().cpu()
        min_id = int(env_ids_cpu.min().item())
        max_seen = int(env_ids_cpu.max().item())
        if min_id < 0 or max_seen > max_id:
            raise RuntimeError(
                f"env_id out of range in _env_max_steps_from_ids: "
                f"min={min_id}, max={max_seen}, n_envs={table.shape[0]}"
            )
        return table[env_ids].to(self.device)

    def _scenario_text_for(self, env_obj: Any, scenario_id: int) -> str:
        """
        Best-effort scenario text used for hashed instruction tokens.

        The goal is not perfect NLP, but to provide a stable string that can
        condition the policy/world model on "what task is this?".
        """
        env_family = str(getattr(env_obj, "env_family", "") or "")
        env_name = str(getattr(env_obj, "env_name", "") or "")
        sid = int(scenario_id)

        scenario_name = None
        scenario_desc = None

        configs = getattr(env_obj, "scenario_configs", None)
        if isinstance(configs, list) and 0 <= sid < len(configs):
            conf = configs[sid]
            if isinstance(conf, dict):
                scenario_name = conf.get("name")
                scenario_desc = conf.get("description")

        if scenario_name is None:
            task_set = getattr(env_obj, "task_set", None)
            tasks = getattr(task_set, "tasks", None) if task_set is not None else None
            if tasks and 0 <= sid < len(tasks):
                task = tasks[sid]
                scenario_name = getattr(task, "name", None)
                scenario_desc = getattr(task, "description", scenario_desc)

        if scenario_name is None:
            scenario_names = getattr(env_obj, "scenario_names", None)
            if isinstance(scenario_names, list) and 0 <= sid < len(scenario_names):
                scenario_name = scenario_names[sid]

        if scenario_name is None:
            scenario_name = str(getattr(env_obj, "current_scenario_name", "") or f"scenario_{sid}")

        if scenario_desc is None:
            scenario_desc = str(getattr(env_obj, "description", "") or "")

        parts = [env_family, env_name, str(scenario_name), str(scenario_desc)]
        text = " ".join(p for p in parts if p).strip()
        return text or str(scenario_name)

    def _compute_text_token_table(self) -> Optional[torch.Tensor]:
        """
        Precompute hashed text tokens for (env_id, scenario_id) pairs.

        Output shape: [n_envs, n_scenarios, L], dtype long on self.device.
        """
        max_len = int(getattr(self.agent.perception, "text_max_len", 0) or 0)
        vocab_size = int(getattr(self.agent.perception, "text_vocab_size", 0) or 0)
        if max_len <= 0 or vocab_size <= 0:
            return None

        envs_list = getattr(self.env, "envs", None)
        if envs_list:
            n_envs = int(len(envs_list))
        else:
            n_envs = 1
            envs_list = [self.env]

        n_scenarios = int(getattr(self.env, "n_scenarios", 1) or 1)
        n_scenarios = max(1, n_scenarios)

        table = torch.zeros(
            n_envs,
            n_scenarios,
            max_len,
            dtype=torch.long,
            device=self.device,
        )
        for env_id in range(n_envs):
            env_obj = envs_list[env_id]
            for sid in range(n_scenarios):
                text = self._scenario_text_for(env_obj, sid)
                ids = hash_text_to_ids(text, max_len=max_len, vocab_size=vocab_size)
                table[env_id, sid] = torch.from_numpy(ids).to(self.device, dtype=torch.long)
        return table

    def _text_tokens_from_ids(self, env_ids: torch.Tensor, scenario_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Gather [*, L] text tokens aligned with env_ids/scenario_ids.
        Returns None when text conditioning is disabled.
        """
        table = getattr(self, "text_token_table", None)
        if table is None:
            return None

        # Validate indices on CPU to avoid CUDA device-side asserts.
        env_max = int(table.shape[0] - 1)
        sc_max = int(table.shape[1] - 1)
        env_cpu = env_ids.detach().cpu()
        sc_cpu = scenario_ids.detach().cpu()
        env_min_seen = int(env_cpu.min().item())
        env_max_seen = int(env_cpu.max().item())
        sc_min_seen = int(sc_cpu.min().item())
        sc_max_seen = int(sc_cpu.max().item())
        if env_min_seen < 0 or env_max_seen > env_max:
            raise RuntimeError(
                f"env_id out of range in _text_tokens_from_ids: "
                f"min={env_min_seen}, max={env_max_seen}, n_envs={table.shape[0]}"
            )
        if sc_min_seen < 0 or sc_max_seen > sc_max:
            raise RuntimeError(
                f"scenario_id out of range in _text_tokens_from_ids: "
                f"min={sc_min_seen}, max={sc_max_seen}, n_scenarios={table.shape[1]}"
            )

        return table[env_ids, scenario_ids]

    def _sample_sequences_from_buffer(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        seq_len: int,
        with_events: bool = False,
        current_regime: Optional[str] = None,
    ):
        """
        Wrapper that enables regime-aware replay when configured.
        """
        if self.regime_aware_replay:
            mix_cfg = {
                "current_regime": current_regime or self.current_regime_name or "",
                "frac_current": self.replay_frac_current,
            }
            return buffer.sample_mixed(
                batch_size=batch_size,
                seq_len=seq_len,
                mix_config=mix_cfg,
                with_events=with_events,
            )
        if with_events:
            return buffer.sample_sequences_with_events(batch_size, seq_len)
        return buffer.sample_sequences(batch_size, seq_len)

    def _get_faction_mix(self, num_factions: Optional[int] = None) -> torch.Tensor:
        """
        Softmax-normalized mixture weights over factions (default: uniform if absent).
        """
        n_f = int(num_factions if num_factions is not None else getattr(self.agent.traits, "shape", [1])[0])
        if hasattr(self.agent, "faction_weights"):
            w = self.agent.faction_weights
            if w.numel() < n_f:
                pad = torch.zeros(n_f - w.numel(), device=self.device, dtype=w.dtype)
                w = torch.cat([w, pad], dim=0)
            w = w[:n_f]
            return F.softmax(w, dim=0)
        return torch.ones(n_f, device=self.device) / float(max(1, n_f))

    def _get_mixed_traits(self) -> torch.Tensor:
        """
        Collapse multi-faction traits into a single vector using faction mix weights.
        """
        traits = self.agent.traits
        if traits.dim() == 1:
            return traits.view(1, -1)
        if traits.size(0) == 1:
            return traits
        mix = self._get_faction_mix(traits.size(0)).view(-1, 1)
        mixed = torch.sum(mix * traits, dim=0, keepdim=True)
        return mixed

    def _mixed_traits(self) -> torch.Tensor:
        """
        Backward-compatible alias for mixed traits accessor.
        """
        return self._get_mixed_traits()

    def _main_traits(self) -> torch.Tensor:
        traits = self.agent.traits
        if traits.dim() == 1 or traits.size(0) == 1:
            return traits.view(1, -1)
        return traits[:1]

    def _safety_traits(self) -> torch.Tensor:
        traits = self.agent.traits
        if traits.dim() > 1 and traits.size(0) > 1:
            return traits[1:2]
        return self._main_traits()

    def _faction_preference_weights(self) -> torch.Tensor:
        w = traits_to_preference_weights(self.agent.traits)
        if w.dim() == 1:
            return w.view(1, -1)
        return w

    def _combined_preference_weights(self) -> torch.Tensor:
        weights = self._faction_preference_weights()
        mix = self._get_faction_mix(weights.size(0)).view(-1, 1)
        combined = (mix * weights).sum(dim=0, keepdim=True)
        return combined

    def _start_trait_safety_phase(
        self,
        phase: str,
        base_traits_main: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        ctx = self._trait_safety_ctx.get(phase)
        if ctx is not None and base_traits_main is None:
            return ctx
        base_main = base_traits_main if base_traits_main is not None else self._main_traits().detach().clone()
        if base_main.dim() == 1:
            base_main = base_main.view(1, -1)
        traits_clone = self.agent.traits.detach().clone()
        if traits_clone.dim() == 1:
            traits_clone = traits_clone.view(1, -1)
        if traits_clone.size(0) == 0:
            traits_clone = base_main.clone()
        else:
            traits_clone = traits_clone.clone()
            traits_clone[0] = base_main[0]
        weights_base = traits_to_preference_weights(traits_clone)
        if weights_base.dim() == 1:
            weights_base = weights_base.view(1, -1)
        ctx = {
            "base_traits": base_main.detach().clone(),
            "base_weights": weights_base.detach().clone(),
            "w_survive_base": float(weights_base[0, 0].item()),
            "w_danger_base": float(weights_base[0, 2].item()),
        }
        self._trait_safety_ctx[phase] = ctx
        return ctx

    def apply_safe_trait_update(
        self,
        agent: ProtoCreatureAgent,
        traits_main_new: torch.Tensor,
        phase: str,
        finalize: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply a trait update with survival/danger safeguards and optional trust-region clipping.
        """
        cfg = self.safety
        ctx = self._trait_safety_ctx.get(phase)
        if ctx is None:
            ctx = self._start_trait_safety_phase(phase)
        base_traits = ctx["base_traits"]
        w_survive_base = ctx["w_survive_base"]
        w_danger_base = ctx["w_danger_base"]

        if finalize:
            limit = cfg.max_trait_delta_norm_stage3b if phase == "stage3b" else cfg.max_trait_delta_norm_phaseC
            current_main = self._main_traits().detach()
            total_delta = float((current_main - base_traits).norm().item())
            clipped = False
            if limit is not None and total_delta > limit:
                scale = limit / (total_delta + 1e-8)
                new_main = base_traits + (current_main - base_traits) * scale
                agent.traits.data[0] = new_main[0]
                clipped = True
                if self.logger is not None:
                    self.logger.log(
                        {
                            "event": "self_reflection_delta_clipped",
                            "phase": phase,
                            "delta_before": total_delta,
                            "delta_after": limit,
                        }
                    )
            return {"delta_before": total_delta, "clipped": clipped}

        cand = traits_main_new.detach()
        if cand.dim() == 1:
            cand = cand.view(1, -1)
        traits_clone = agent.traits.detach().clone()
        if traits_clone.dim() == 1:
            traits_clone = traits_clone.view(1, -1)
        if traits_clone.size(0) == 0:
            traits_clone = cand.clone()
        else:
            traits_clone = traits_clone.clone()
            traits_clone[0] = cand[0]
        w_new_all = traits_to_preference_weights(traits_clone)
        if w_new_all.dim() == 1:
            w_new_all = w_new_all.view(1, -1)
        w_survive_new = float(w_new_all[0, 0].item())
        w_danger_new = float(w_new_all[0, 2].item())
        violate_survival = cfg.forbid_survival_weight_decrease and (
            w_survive_new + cfg.survival_epsilon_down < w_survive_base
        )
        violate_danger = cfg.forbid_danger_aversion_decrease and (
            abs(w_danger_new) + cfg.danger_aversion_epsilon_down < abs(w_danger_base)
        )

        if violate_survival or violate_danger:
            if self.logger is not None:
                self.logger.log(
                    {
                        "event": "self_reflection_step_rejected",
                        "reason": "survival_or_danger_constraint",
                        "phase": phase,
                        "w_survive_base": w_survive_base,
                        "w_survive_new": w_survive_new,
                        "w_danger_base": w_danger_base,
                        "w_danger_new": w_danger_new,
                    }
                )
            return {
                "accepted": False,
                "w_survive_new": w_survive_new,
                "w_danger_new": w_danger_new,
            }

        new_main = torch.clamp(cand[0].to(agent.traits.device), -2.0, 2.0)
        agent.traits.data[0] = new_main
        return {
            "accepted": True,
            "w_survive_new": w_survive_new,
            "w_danger_new": w_danger_new,
            "delta_from_base": float((new_main - base_traits[0].to(new_main.device)).norm().item()),
        }

    def _compute_faction_utilities(self, components: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        components: (..., 4) ordered as (survival, food, damage, move)
        Returns (combined_utility, per_faction_utilities) where combined collapses factions
        using the current faction mix.
        """
        if components.size(-1) != 4:
            raise ValueError(f"Expected components last dim=4, got {components.shape}")
        weights = self._faction_preference_weights()  # (F,4)
        if components.dim() == 1:
            comp = components.view(1, -1)
        else:
            comp = components
        w_shape = [weights.size(0)] + [1] * (comp.dim() - 1) + [4]
        c_shape = [1] * 1 + list(comp.shape)
        w_exp = weights.view(*w_shape)
        comp_exp = comp.view(*c_shape)
        utilities = (w_exp * comp_exp).sum(dim=-1)
        mix = self._get_faction_mix(weights.size(0))
        mix_shape = [mix.shape[0]] + [1] * (utilities.dim() - 1)
        combined = (mix.view(*mix_shape) * utilities).sum(dim=0)
        return combined, utilities

    def _apply_safety_penalty(self, score_main: torch.Tensor, score_safety: torch.Tensor) -> torch.Tensor:
        """
        Apply a soft safety penalty when safety-utility falls below threshold.
        """
        threshold = torch.as_tensor(self.safety_threshold, device=score_main.device, dtype=score_main.dtype)
        coef = float(self.safety_penalty_coef)
        gap = torch.clamp(threshold - score_safety, min=0.0)
        return score_main - coef * gap

    @staticmethod
    def _traits_to_dict(traits: torch.Tensor) -> Dict[str, float]:
        arr = traits.detach().cpu().numpy().flatten()
        return {
            "survival": float(arr[0]) if len(arr) > 0 else 0.0,
            "food": float(arr[1]) if len(arr) > 1 else 0.0,
            "damage": float(arr[2]) if len(arr) > 2 else 0.0,
            "move": float(arr[3]) if len(arr) > 3 else 0.0,
        }

    def _log_trait_reflection_event(
        self,
        traits_before: torch.Tensor,
        traits_after: torch.Tensor,
        regime_name: str,
        scenario_id: Optional[int] = None,
        env_id: Optional[int] = None,
        comment: str = "",
        reason: Optional[Dict[str, Any]] = None,
        observed_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Append a structured trait reflection event to the log and optionally emit via logger.
        """
        before = self._traits_to_dict(traits_before)
        after = self._traits_to_dict(traits_after)
        delta = {k: after[k] - before.get(k, 0.0) for k in after.keys()}
        event = {
            "timestamp": float(time.time()),
            "regime": regime_name,
            "scenario_id": scenario_id,
            "env_id": env_id,
            "traits_before": before,
            "traits_after": after,
            "delta_traits": delta,
            "reason": reason or {},
            "observed": observed_stats or {},
            "comment": comment,
        }
        self.trait_reflection_log.append(event)
        if self.logger is not None:
            self.logger.log_scalar(
                stage="trait_reflection",
                metric="update",
                value=1.0,
                regime=regime_name,
                scenario_id=scenario_id,
                env_id=env_id,
                delta_traits=delta,
                comment=comment,
            )

    # ----- printing traits -----

    def print_traits(self):
        with torch.no_grad():
            traits_all = self.agent.traits.detach().cpu().numpy()
            traits_mixed = self._mixed_traits().detach().cpu().numpy()
            w_all = traits_to_preference_weights(self.agent.traits).detach().cpu().numpy()
            w_combined = self._combined_preference_weights().detach().cpu().numpy()
        print("  traits (factions x dim):", traits_all)
        print("  mixed traits (weighted):", traits_mixed)
        print("  preference weights per faction [survive, food, danger, move]:", w_all)
        print("  combined preference weights:", w_combined)

    # ----- trait reflection helpers -----

    @staticmethod
    def describe_trait_delta(trait_before: np.ndarray, trait_after: np.ndarray) -> str:
        """
        Produce a short human-readable description of the dominant trait change.
        """
        delta = trait_after - trait_before
        if delta.ndim > 1:
            delta = delta.reshape(-1)
        idx = int(np.argmax(np.abs(delta)))
        dv = float(delta[idx])
        if abs(dv) < 1e-6:
            return "Reflection: no meaningful trait change."
        if idx == 0:
            return "Reflection: increased survival focus." if dv > 0 else "Reflection: became more willing to risk survival for other goals."
        if idx == 1:
            return "Reflection: increased food-seeking / extrinsic reward focus." if dv > 0 else "Reflection: reduced emphasis on food rewards."
        if idx == 2:
            return "Reflection: became more curious about danger / damage." if dv > 0 else "Reflection: increased danger avoidance."
        if idx == 3:
            return "Reflection: increased exploration / movement preference." if dv > 0 else "Reflection: increased efficiency / reduced unnecessary movement."
        return "Reflection: updated traits."

    def _propose_trait_candidates(self, base_traits: torch.Tensor, step_size: float = 0.25) -> List[torch.Tensor]:
        """
        Generate axis-aligned candidate trait vectors around the current traits.
        """
        candidates: List[torch.Tensor] = []
        base = base_traits.detach().clone()
        candidates.append(base)
        for dim in range(base.shape[1]):
            for sign in (1.0, -1.0):
                cand = base.clone()
                cand[0, dim] += float(sign) * step_size
                cand.clamp_(-2.0, 2.0)
                candidates.append(cand)
        return candidates

    def _score_candidate(
        self,
        w_eff: torch.Tensor,
        surv_pred: torch.Tensor,
        food_pred: torch.Tensor,
        dmg_pred: torch.Tensor,
        move_pred: torch.Tensor,
    ) -> float:
        """
        Compute a scalar score (mean calibrated return) for a candidate weight vector.
        """
        w_survive, w_food, w_danger, w_move = w_eff[0]
        R_unscaled = (
            w_survive * surv_pred
            + w_food * food_pred
            + w_danger * dmg_pred
            + w_move * move_pred
        )
        R_calib = self.agent.self_model.head_return_calib(R_unscaled)
        return float(R_calib.mean().item())

    def _prepare_reflection_features(
        self,
        batch: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> Dict[str, torch.Tensor]:
        """
        Shared pre-computation for trait reflection: run perception/world-model/self-model
        to get predicted (survival, food, damage, move) sequences and cached rewards.
        """
        (
            obs_seq,
            H_seq,
            a_seq,
            r_seq,
            d_seq,
            death_seq,
            food_seq,
            dmg_seq,
            move_seq,
            alive_seq,
            scenario_seq,
            env_seq,
        ) = batch

        with torch.no_grad():
            B, T, p, _ = obs_seq.shape
            patch = torch.from_numpy(obs_seq).long().to(self.device)
            H = torch.from_numpy(H_seq).float().to(self.device)
            a = torch.from_numpy(a_seq).long().to(self.device)
            r = torch.from_numpy(r_seq).float().to(self.device)
            scenario = torch.from_numpy(scenario_seq).long().to(self.device)
            env_ids = torch.from_numpy(env_seq).long().to(self.device)
            env_desc_seq = self._env_desc_from_ids(env_ids)

            patch_flat = patch.view(B * T, p, p)
            H_flat = H.view(B * T, 1)
            scenario_flat = scenario.view(B * T)
            env_flat = env_ids.view(B * T)
            env_desc_flat = (
                env_desc_seq.reshape(B * T, -1) if env_desc_seq is not None else None
            )
            text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)

            z_flat = self.agent.perception(
                patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
            )
            z_seq = z_flat.reshape(B, T, -1)

            a_emb = self.agent.world_model.act_emb(a)
            x_w = torch.cat([z_seq, H, a_emb], dim=-1)
            h0 = torch.zeros(
                1,
                B,
                self.agent.world_model.gru.hidden_size,
                device=self.device,
            )
            W_seq, _ = self.agent.world_model.gru(x_w, h0)

            M = self.agent.memory
            (
                S_seq,
                S_last,
                surv_pred,
                food_pred,
                dmg_pred,
                move_pred,
                unc_pred,
                surv_raw_pred,
            ) = self.agent.self_model.forward_seq(
                W_seq,
                H,
                a,
                r,
                M=M,
                env_desc=env_desc_seq,
            )

        feat_seq = torch.stack(
            [
                surv_pred.squeeze(-1),
                food_pred.squeeze(-1),
                dmg_pred.squeeze(-1),
                move_pred.squeeze(-1),
            ],
            dim=-1,
        )  # (B,T,4)

        return {
            "surv_pred": surv_pred.detach(),
            "food_pred": food_pred.detach(),
            "dmg_pred": dmg_pred.detach(),
            "move_pred": move_pred.detach(),
            "feat_seq": feat_seq.detach(),
            "reward_seq": r.detach().unsqueeze(-1),
        }

    def _compute_reflection_objective(
        self,
        trait_vec: torch.Tensor,
        features: Optional[Dict[str, torch.Tensor]] = None,
        batch: Optional[
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ] = None,
        reward_profile: Optional[Dict[str, float]] = None,
        lambda_l2: Optional[float] = None,
        trait_anchor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Reflection objective: maximize calibrated return while staying near an anchor.
        Returns a loss (to minimize) and lightweight debug scalars.
        """
        if features is None:
            if batch is None:
                raise ValueError("Either features or batch must be provided for reflection objective.")
            features = self._prepare_reflection_features(batch)
        feat_seq = features["feat_seq"]  # (B,T,4)
        reward_seq = features.get("reward_seq")

        w_raw = traits_to_preference_weights(trait_vec)  # (1,4) or (F,4)
        if w_raw.dim() == 2 and w_raw.size(0) > 1:
            w_raw = w_raw[:1]
        elif w_raw.dim() == 1:
            w_raw = w_raw.view(1, -1)
        alpha_vec = torch.tensor(
            [
                float((reward_profile or {}).get("survive", 1.0)),
                float((reward_profile or {}).get("food", 1.0)),
                float((reward_profile or {}).get("danger", 1.0)),
                float((reward_profile or {}).get("move", 1.0)),
            ],
            device=trait_vec.device,
            dtype=trait_vec.dtype,
        ).view(1, 1, 4)
        w_eff = w_raw.view(1, 1, 4) * alpha_vec

        utility_seq = (feat_seq * w_eff).sum(dim=-1, keepdim=True)  # (B,T,1)
        ret_pred = self.agent.self_model.head_return_calib(utility_seq)

        target_base = reward_seq if reward_seq is not None else ret_pred.detach()
        target_mean = target_base.mean()
        eps_std = float(getattr(self.safety, "std_eps", 1e-6))
        target_std = target_base.std(unbiased=False)
        if not torch.isfinite(target_std):
            target_std = torch.tensor(eps_std, device=target_base.device, dtype=target_base.dtype)
        target_std = torch.clamp(target_std, min=eps_std)
        if self.trait_reflection_reward_normalization:
            denom = target_std
            target_proc = torch.clamp((target_base - target_mean) / denom, -5.0, 5.0)
            ret_proc = torch.clamp((ret_pred - target_mean) / denom, -5.0, 5.0)
        else:
            target_proc = torch.clamp(target_base, -5.0, 5.0)
            ret_proc = torch.clamp(ret_pred, -5.0, 5.0)

        anchor = trait_anchor if trait_anchor is not None else trait_vec.detach()
        lambda_term = lambda_l2 if lambda_l2 is not None else self.trait_reflection_lambda_l2
        anchor_l2 = torch.norm(trait_vec - anchor) ** 2
        loss = -ret_proc.mean() + lambda_term * anchor_l2

        debug_info = {
            "pred_mean": float(ret_pred.mean().detach().item()),
            "target_mean": float(target_proc.mean().detach().item()),
            "anchor_l2": float(anchor_l2.detach().item()),
        }
        return loss, debug_info

    def _predict_reflection_utility(
        self,
        trait_vec: torch.Tensor,
        features: Dict[str, torch.Tensor],
        reward_profile: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Scalar predicted utility (mean calibrated return) for improvement checks."""
        try:
            feat_seq = features["feat_seq"]
        except Exception:
            return None
        try:
            w_raw = traits_to_preference_weights(trait_vec)
        except Exception:
            return None
        if w_raw.dim() == 2 and w_raw.size(0) > 1:
            w_raw = w_raw[:1]
        elif w_raw.dim() == 1:
            w_raw = w_raw.view(1, -1)
        alpha_vec = torch.tensor(
            [
                float((reward_profile or {}).get("survive", 1.0)),
                float((reward_profile or {}).get("food", 1.0)),
                float((reward_profile or {}).get("danger", 1.0)),
                float((reward_profile or {}).get("move", 1.0)),
            ],
            device=trait_vec.device,
            dtype=trait_vec.dtype,
        ).view(1, 1, 4)
        try:
            w_eff = w_raw.view(1, 1, 4) * alpha_vec
            utility_seq = (feat_seq * w_eff).sum(dim=-1, keepdim=True)
            ret_pred = self.agent.self_model.head_return_calib(utility_seq)
            return float(ret_pred.mean().item())
        except Exception:
            return None

    def _apply_trust_region_trait_update(
        self,
        trait_vec: torch.Tensor,
        grad: torch.Tensor,
        anchor_traits: torch.Tensor,
        init_traits: Optional[torch.Tensor] = None,
        lr_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Trust-region style trait update: clipped local step + projection to a global ball.
        """
        lr = lr_override if lr_override is not None else self.trait_reflection_lr
        grad_norm = float(grad.norm().item())
        if self.trait_reflection_grad_clip is not None and grad_norm > self.trait_reflection_grad_clip:
            grad = grad * (self.trait_reflection_grad_clip / (grad_norm + 1e-8))
            grad_norm = float(grad.norm().item())

        delta = -lr * grad
        step_norm = float(delta.norm().item())
        step_cap = self.trait_reflection_max_l2_from_anchor
        if step_cap is not None and step_norm > step_cap:
            delta = delta * (step_cap / (step_norm + 1e-8))
            step_norm = float(delta.norm().item())

        new_traits = trait_vec + delta
        new_traits = torch.clamp(new_traits, -2.0, 2.0)

        init_ref = init_traits if init_traits is not None else anchor_traits
        dist_from_init = float(torch.norm(new_traits - init_ref).item())
        global_cap = self.trait_reflection_max_global_l2_from_init
        if global_cap is not None and dist_from_init > global_cap:
            direction = new_traits - init_ref
            new_traits = init_ref + direction * (global_cap / (dist_from_init + 1e-8))
            dist_from_init = float(torch.norm(new_traits - init_ref).item())

        step_norm = float(torch.norm(new_traits - trait_vec).item())

        info = {
            "step_norm": step_norm,
            "grad_norm": grad_norm,
            "dist_from_init": dist_from_init,
        }
        return new_traits.detach(), info

    def _reflect_traits_propose_commit(
        self,
        batch: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        reward_profile: Optional[Dict[str, float]] = None,
        step_size: float = 0.25,
        regime_name: str = "reflection",
        trait_anchor: Optional[torch.Tensor] = None,
        init_traits: Optional[torch.Tensor] = None,
        scenario_id: Optional[int] = None,
        env_id: Optional[int] = None,
        safety_phase: Optional[str] = None,
        safety_baseline: Optional[torch.Tensor] = None,
        log_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Propose -> evaluate -> commit trait updates using SelfModel predictions.
        Returns a debug dict for logging.
        """
        features = self._prepare_reflection_features(batch)
        surv_pred = features["surv_pred"]
        food_pred = features["food_pred"]
        dmg_pred = features["dmg_pred"]
        move_pred = features["move_pred"]

        with torch.no_grad():
            base_traits = self._main_traits().detach().clone()
            anchor = trait_anchor if trait_anchor is not None else base_traits
            init_ref = init_traits if init_traits is not None else anchor
            candidates = self._propose_trait_candidates(base_traits, step_size=step_size)
            best_traits = base_traits
            best_score = -1e9
            base_score = None

            alpha_vec = torch.tensor(
                [
                    float((reward_profile or {}).get("survive", 1.0)),
                    float((reward_profile or {}).get("food", 1.0)),
                    float((reward_profile or {}).get("danger", 1.0)),
                    float((reward_profile or {}).get("move", 1.0)),
                ],
                device=self.device,
                dtype=base_traits.dtype,
            ).view(1, 4)

            for cand in candidates:
                w_raw = traits_to_preference_weights(cand)
                w_eff = w_raw * alpha_vec
                score = self._score_candidate(
                    w_eff, surv_pred=surv_pred, food_pred=food_pred, dmg_pred=dmg_pred, move_pred=move_pred
                )
                if base_score is None and torch.allclose(cand, base_traits):
                    base_score = score
                if score > best_score:
                    best_score = score
                    best_traits = cand.clone()

            if base_score is None:
                base_score = best_score

            changed = (best_score > base_score + 1e-6) and (torch.norm(best_traits - base_traits).item() > 1e-6)
            if changed:
                self.agent.traits.data[0:1] = best_traits.clone()
                self.agent.traits.data[0:1].clamp_(-2.0, 2.0)
                step_norm_val = float(torch.norm(self.agent.traits[:1] - base_traits).item())
                cap = self.trait_reflection_max_global_l2_from_init
                if cap is not None:
                    diff = self.agent.traits[:1] - init_ref
                    dist = torch.norm(diff)
                    if dist.item() > cap:
                        self.agent.traits.data[0:1] = init_ref + diff * (cap / (dist + 1e-8))
                        step_norm_val = float(torch.norm(self.agent.traits[:1] - base_traits).item())
                msg = self.describe_trait_delta(
                    base_traits.detach().cpu().numpy(), self.agent.traits[:1].detach().cpu().numpy()
                )
                observed_stats = {
                    "survival_pred_mean": float(surv_pred.mean().item()),
                    "food_pred_mean": float(food_pred.mean().item()),
                    "damage_pred_mean": float(dmg_pred.mean().item()),
                    "move_pred_mean": float(move_pred.mean().item()),
                }
                if "reward_seq" in features:
                    observed_stats["reward_mean"] = float(features["reward_seq"].mean().item())
                reason = {
                    "type": "axis_search",
                    "improvement": float(best_score - (base_score or best_score)),
                    "reward_profile": reward_profile or {},
                }
                self._log_trait_reflection_event(
                    traits_before=base_traits,
                    traits_after=self.agent.traits[:1],
                    regime_name=regime_name,
                    scenario_id=scenario_id,
                    env_id=env_id,
                    comment=msg,
                    reason=reason,
                    observed_stats=observed_stats,
                )
                if log_progress:
                    print(f"[Self-reflection][{regime_name}] {msg}")
                dist_from_anchor = float(torch.norm(self.agent.traits[:1] - anchor).item())
                dist_from_init = float(torch.norm(self.agent.traits[:1] - init_ref).item())
                return {
                    "updated": changed,
                    "steps": 1,
                    "step_norms": [step_norm_val],
                    "dist_from_anchor": dist_from_anchor,
                    "dist_from_init": dist_from_init,
                    "mode": "axis_search",
                }
        return {
            "updated": False,
            "steps": 0,
            "step_norms": [],
            "dist_from_anchor": 0.0,
            "dist_from_init": 0.0,
            "mode": "axis_search",
        }



    def _reflect_traits_gradient(
        self,
        batch: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        reward_profile: Optional[Dict[str, float]] = None,
        step_size: float = 0.05,
        regime_name: str = "reflection",
        lambda_l2: float = 0.05,
        n_steps: Optional[int] = None,
        trait_anchor: Optional[torch.Tensor] = None,
        init_traits: Optional[torch.Tensor] = None,
        scenario_id: Optional[int] = None,
        env_id: Optional[int] = None,
        safety_phase: Optional[str] = None,
        safety_baseline: Optional[torch.Tensor] = None,
        log_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Smooth, gradient-like reflection update on traits (default path).
        """
        features = self._prepare_reflection_features(batch)
        surv_pred = features["surv_pred"]
        food_pred = features["food_pred"]
        dmg_pred = features["dmg_pred"]
        move_pred = features["move_pred"]
        base_traits = self._main_traits().detach().clone()
        anchor = trait_anchor if trait_anchor is not None else base_traits
        init_ref = init_traits if init_traits is not None else anchor
        inner_steps = n_steps if n_steps is not None else self.trait_reflection_steps_per_batch
        inner_steps = max(1, int(inner_steps))
        lr_step = min(step_size, self.trait_reflection_lr)

        utility_before = self._predict_reflection_utility(base_traits, features, reward_profile)
        if utility_before is None:
            return {
                "updated": False,
                "steps": 0,
                "improvement": 0.0,
                "utility_before": None,
                "utility_after": None,
                "step_norms": [],
                "grad_norms": [],
            }
        traits_candidate = base_traits
        step_norms: List[float] = []
        grad_norms: List[float] = []
        steps_taken = 0
        last_debug: Dict[str, Any] = {}
        improvement = 0.0
        if safety_phase is not None:
            baseline_traits = safety_baseline if safety_baseline is not None else base_traits
            self._start_trait_safety_phase(safety_phase, baseline_traits)

        for _ in range(inner_steps):
            self.agent.self_model.zero_grad(set_to_none=True)
            traits_var = traits_candidate.detach().clone().to(self.device)
            traits_var.requires_grad_(True)
            loss, debug_info = self._compute_reflection_objective(
                traits_var,
                features=features,
                reward_profile=reward_profile,
                lambda_l2=lambda_l2,
                trait_anchor=anchor,
            )
            loss.backward()
            grad = traits_var.grad
            if grad is None or grad.norm().item() < 1e-8:
                break

            proposed_traits, step_info = self._apply_trust_region_trait_update(
                trait_vec=traits_var.detach(),
                grad=grad.detach(),
                anchor_traits=anchor,
                init_traits=init_ref,
                lr_override=lr_step,
            )

            utility_after = self._predict_reflection_utility(proposed_traits, features, reward_profile)
            if utility_after is None:
                break
            improvement = utility_after - utility_before
            debug_info.update(
                {
                    "loss": float(loss.detach().item()),
                    "utility_before": utility_before,
                    "utility_after": utility_after,
                    "improvement": improvement,
                    **step_info,
                }
            )
            last_debug = debug_info

            if improvement < self.trait_reflection_min_improvement:
                break

            step_accepted = True
            if safety_phase is not None:
                accept_info = self.apply_safe_trait_update(self.agent, proposed_traits, phase=safety_phase)
                step_accepted = bool(accept_info.get("accepted", False))
                if not step_accepted:
                    continue
                traits_candidate = self._main_traits().detach().clone()
            else:
                traits_candidate = proposed_traits

            utility_before = utility_after
            step_norms.append(step_info["step_norm"])
            grad_norms.append(step_info["grad_norm"])
            steps_taken += 1

        with torch.no_grad():
            final_traits = self._main_traits().detach().clone() if safety_phase is not None else traits_candidate
            dist_from_anchor = float(torch.norm(final_traits - anchor).item())
            dist_from_init = float(torch.norm(final_traits - init_ref).item())
            change_norm = float(torch.norm(final_traits - base_traits).item())
            updated = change_norm > 1e-6
            if updated:
                if safety_phase is None:
                    traits_update = traits_candidate
                    if traits_update.dim() == 2 and traits_update.size(0) > 1:
                        traits_update = traits_update[:1]
                    elif traits_update.dim() == 1:
                        traits_update = traits_update.view(1, -1)
                    self.agent.traits.data[0:1] = traits_update.clone()
                    self.agent.traits.data[0:1].clamp_(-2.0, 2.0)
                    traits_after = self.agent.traits[:1]
                else:
                    traits_after = final_traits
                msg = self.describe_trait_delta(
                    base_traits.detach().cpu().numpy(), traits_after.detach().cpu().numpy()
                )
                extra = (
                    f"steps={steps_taken}, lr={lr_step:.3f}, "
                    f"||delta||={change_norm:.4f}, mean_step_norm={np.mean(step_norms) if step_norms else 0.0:.4f}"
                )
                observed_stats = {
                    "survival_pred_mean": float(surv_pred.mean().item()),
                    "food_pred_mean": float(food_pred.mean().item()),
                    "damage_pred_mean": float(dmg_pred.mean().item()),
                    "move_pred_mean": float(move_pred.mean().item()),
                }
                if "reward_seq" in features:
                    observed_stats["reward_mean"] = float(features["reward_seq"].mean().item())
                reason = {
                    "type": "gradient",
                    "improvement": float(improvement),
                    "reward_profile": reward_profile or {},
                }
                self._log_trait_reflection_event(
                    traits_before=base_traits,
                    traits_after=traits_after,
                    regime_name=regime_name,
                    scenario_id=scenario_id,
                    env_id=env_id,
                    comment=f"{msg} ({extra})",
                    reason=reason,
                    observed_stats=observed_stats,
                )
                if log_progress:
                    print(f"[Self-reflection][{regime_name}] {msg} | {extra}")

        return {
            "updated": bool(updated),
            "steps": steps_taken,
            "step_norms": step_norms,
            "grad_norms": grad_norms,
            "dist_from_anchor": dist_from_anchor,
            "dist_from_init": dist_from_init,
            "change_norm": change_norm,
            "utility_after": utility_before,
            "last_debug": last_debug,
            "mode": "gradient",
        }

    def _run_trait_reflection(
        self,
        batch: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        reward_profile: Optional[Dict[str, float]] = None,
        step_size: float = 0.05,
        regime_name: str = "reflection",
        lambda_l2: float = 0.05,
        n_steps: Optional[int] = None,
        trait_anchor: Optional[torch.Tensor] = None,
        init_traits: Optional[torch.Tensor] = None,
        safety_phase: Optional[str] = None,
        safety_baseline: Optional[torch.Tensor] = None,
        log_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Choose between smooth gradient-based reflection (default) and legacy axis search.
        """
        scenario_id = None
        env_id = None
        try:
            scenario_seq = batch[-2]
            env_seq = batch[-1]
            scenario_id = int(scenario_seq[0, 0]) if scenario_seq.size > 0 else None
            env_id = int(env_seq[0, 0]) if env_seq.size > 0 else None
        except Exception:
            pass

        mode_axis = self.trait_reflection_mode == "axis_search" or self.trait_reflection_use_axis_search
        if mode_axis:
            return self._reflect_traits_propose_commit(
                batch=batch,
                reward_profile=reward_profile,
                step_size=step_size,
                regime_name=regime_name,
                trait_anchor=trait_anchor,
                init_traits=init_traits,
                scenario_id=scenario_id,
                env_id=env_id,
                safety_phase=safety_phase,
                safety_baseline=safety_baseline,
                log_progress=log_progress,
            )
        return self._reflect_traits_gradient(
            batch=batch,
            reward_profile=reward_profile,
            step_size=step_size,
            regime_name=regime_name,
            lambda_l2=lambda_l2,
            n_steps=n_steps,
            trait_anchor=trait_anchor,
            init_traits=init_traits,
            scenario_id=scenario_id,
            env_id=env_id,
            safety_phase=safety_phase,
            safety_baseline=safety_baseline,
            log_progress=log_progress,
        )

    def get_trait_reflection_summary(
        self,
        max_messages_per_regime: int = 3,
    ) -> Dict[str, Any]:
        """
        Build a compact, JSON-serializable summary of trait reflection events.

        Groups entries in self.trait_reflection_log by 'regime', and for each regime
        returns:
          - total number of updates,
          - first message,
          - last message,
          - a short list of recent messages (up to max_messages_per_regime).

        If no reflections happened, returns {}.
        """
        if not self.trait_reflection_log:
            return {}

        summary: Dict[str, Any] = {}
        total_updates = 0
        max_msgs = max(0, int(max_messages_per_regime))
        for entry in self.trait_reflection_log:
            if isinstance(entry, dict):
                regime = str(entry.get("regime", "unknown"))
                msg = str(entry.get("comment", entry.get("message", "")))
            else:
                regime = "unknown"
                msg = str(entry)
            total_updates += 1
            reg_summary = summary.setdefault(
                regime, {"count": 0, "first": "", "last": "", "recent": []}
            )
            reg_summary["count"] = int(reg_summary.get("count", 0)) + 1
            if not reg_summary["first"]:
                reg_summary["first"] = msg
            reg_summary["last"] = msg
            if max_msgs > 0:
                recent = list(reg_summary.get("recent", []))
                recent.append(msg)
                reg_summary["recent"] = recent[-max_msgs:]

        summary["_total_updates"] = total_updates
        return summary

    def get_trait_reflection_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return the structured trait reflection log (optionally truncated).
        """
        if limit is None or limit <= 0:
            return list(self.trait_reflection_log)
        return list(self.trait_reflection_log[-limit:])

    # ----- regime helpers for lifelong eval -----
# ----- regime helpers for lifelong eval -----
    # ----- regime helpers for lifelong eval -----

    def _get_scenario_name_to_id(self) -> Dict[str, int]:
        """Build a lower-case name -> scenario_id map from the active env/pool."""
        mapping: Dict[str, int] = {}
        envs: List[Any] = []
        if hasattr(self.env, "envs"):
            envs = list(getattr(self.env, "envs", []))
        else:
            envs = [self.env]

        for env_obj in envs:
            configs = getattr(env_obj, "scenario_configs", None)
            if not configs:
                continue
            for idx, conf in enumerate(configs):
                name = str(conf.get("name", f"scenario_{idx}")).lower()
                mapping[name] = idx
        if not mapping:
            n = int(getattr(self.env, "n_scenarios", 1) or 1)
            mapping = {f"scenario_{i}": i for i in range(n)}
        return mapping

    def _evaluate_regime_performance(
        self,
        regime: RegimeConfig,
        scenario_map: Dict[str, int],
        episodes: int = 10,
        max_steps: int = 200,
        use_self: bool = True,
        eval_policy: str = "sample",
        stratified_scenarios: bool = False,
    ) -> float:
        """
        Lightweight evaluation for a specific regime (used for forgetting metrics).
        """
        self.current_regime_name = regime.name
        returns: List[float] = []
        reward_profile = regime.reward_profile or {}
        eval_policy_norm = (eval_policy or "sample").lower()
        if eval_policy_norm not in {"sample", "greedy"}:
            eval_policy_norm = "sample"
        scenario_plan = self._build_episode_scenario_plan(
            regime=regime,
            name_to_id=scenario_map,
            n_episodes=max(1, int(episodes)),
            stratified=bool(stratified_scenarios),
        )
        for ep_idx in range(max(1, episodes)):
            scenario_id = scenario_plan[ep_idx] if ep_idx < len(scenario_plan) else None
            if scenario_id is None:
                scenario_id = self._sample_scenario_for_regime(regime, scenario_map)
            obs = self.env.reset(scenario_id=scenario_id)
            patch = obs["patch"]
            energy = obs["energy"]
            scenario_id_ep = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id)))
            env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

            h_w = torch.zeros(1, 1, self.agent.world_model.gru.hidden_size, device=self.device)
            h_s = torch.zeros(1, 1, self.agent.self_model.gru.hidden_size, device=self.device)
            last_action = torch.zeros(1, dtype=torch.long, device=self.device)
            last_reward = 0.0
            traits = self._mixed_traits()
            M = self.agent.memory
            total_r = 0.0
            t = 0
            done = False

            while not done and t < max_steps:
                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id_ep], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)

                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    W_t = h_w.squeeze(0)

                    if use_self:
                        r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                        a_emb = self.agent.self_model.act_emb(last_action)
                        env_emb = self.agent.self_model.env_desc_to_emb(env_desc_t)
                        W_in = W_t.unsqueeze(1)
                        H_in = H_t.unsqueeze(1)
                        r_in = r_t.unsqueeze(1)
                        M_b = M.unsqueeze(1)
                        x_s = torch.cat([W_in, H_in, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)], dim=-1)
                        out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                        S_t = out_s.squeeze(1)
                        surv_t = torch.sigmoid(self.agent.self_model.head_survival(S_t))
                        food_t = self.agent.self_model.head_food(S_t)
                        dmg_t = self.agent.self_model.head_damage(S_t)
                        move_t = self.agent.self_model.head_move(S_t)
                        w = traits_to_preference_weights(traits)
                        w_survive, w_food, w_danger, w_move = w[0]
                        R_unscaled = (w_survive * surv_t + w_food * food_t + w_danger * dmg_t + w_move * move_t).view(
                            1, 1
                        )
                        R_self = self.agent.self_model.head_return_calib(R_unscaled)
                        U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
                    else:
                        R_self = torch.zeros(1, 1, device=self.device)
                        U_t = torch.zeros(1, 1, device=self.device)
                        S_t = torch.zeros(1, self.agent.self_model.gru.hidden_size, device=self.device)

                    V_pi = self.agent.value_model(W_t, H_t, traits, M)
                    delta_self = torch.zeros_like(R_self) if not use_self else torch.tanh(R_self - V_pi)
                    G_t = self.agent.workspace(W_t, S_t, H_t, V_pi, delta_self, U_t, traits, M)
                    logits = self.agent.policy(G_t)
                    logits = self._apply_action_mask(logits)
                    if eval_policy_norm == "greedy":
                        action = torch.argmax(logits, dim=-1)
                    else:
                        dist = Categorical(logits=logits)
                        action = dist.sample()

                next_obs, _, done, info = self.env.step(action.item())
                reward_env = self.compute_preference_reward(info, reward_profile=reward_profile)
                total_r += reward_env
                t += 1

                patch = next_obs["patch"]
                energy = next_obs["energy"]
                scenario_id_ep = int(next_obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id_ep)))
                env_id = int(next_obs.get("env_id", env_id))

                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id_ep], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)
                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs_next = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    _, h_w, _, _ = self.agent.world_model.forward_step(z_obs_next, H_t, action, h_w)

                last_reward = reward_env
                last_action = action
            returns.append(total_r)
        return float(np.mean(returns)) if returns else 0.0

    def _sample_scenario_for_regime(
        self, regime: RegimeConfig, name_to_id: Dict[str, int]
    ) -> Optional[int]:
        """Sample a scenario id according to the regime's scenario weights."""
        if not regime.scenario_weights:
            return None
        entries = []
        weights = []
        for name, w in regime.scenario_weights.items():
            sid = name_to_id.get(name.lower())
            if sid is None:
                continue
            entries.append(int(sid))
            weights.append(float(w))
        if not entries:
            return None
        total = sum(weights)
        if total <= 0:
            return int(entries[0])
        # manual categorical sampling to avoid numpy dependency here
        r = random.random() * total
        acc = 0.0
        for sid, w in zip(entries, weights):
            acc += w
            if r <= acc:
                return int(sid)
        return int(entries[-1])

    def _build_episode_scenario_plan(
        self,
        regime: RegimeConfig,
        name_to_id: Dict[str, int],
        n_episodes: int,
        *,
        stratified: bool,
    ) -> List[Optional[int]]:
        """
        Build per-episode scenario ids for a regime.

        When `stratified=True`, enforce weighted counts per chapter to reduce
        sampling variance while keeping the same expected scenario mix.
        """
        n = max(0, int(n_episodes))
        if n == 0:
            return []
        if not regime.scenario_weights:
            return [None] * n

        entries: List[int] = []
        weights: List[float] = []
        for name, w in regime.scenario_weights.items():
            sid = name_to_id.get(str(name).lower())
            if sid is None:
                continue
            entries.append(int(sid))
            weights.append(max(0.0, float(w)))
        if not entries:
            return [None] * n

        if not stratified:
            return [self._sample_scenario_for_regime(regime, name_to_id) for _ in range(n)]

        total = float(sum(weights))
        if total <= 0.0:
            return [int(entries[0])] * n

        raw = [float(n) * (w / total) for w in weights]
        counts = [int(math.floor(x)) for x in raw]
        remaining = int(n - sum(counts))
        if remaining > 0:
            order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i]), reverse=True)
            for i in order[:remaining]:
                counts[i] += 1
        elif remaining < 0:
            order = sorted(range(len(raw)), key=lambda i: counts[i], reverse=True)
            for i in order[: (-remaining)]:
                counts[i] = max(0, counts[i] - 1)

        plan: List[Optional[int]] = []
        for sid, cnt in zip(entries, counts):
            if cnt > 0:
                plan.extend([int(sid)] * int(cnt))
        while len(plan) < n:
            plan.append(int(entries[-1]))
        if len(plan) > n:
            plan = plan[:n]
        random.shuffle(plan)
        return plan

    def _reward_profile_for_regime(self, regime_name: str) -> Dict[str, float]:
        """
        Map regime names to reward profile modifiers.
        """
        if regime_name.endswith("_return"):
            base = regime_name[: -len("_return")]
            return self._reward_profile_for_regime(base)
        profile = {"survive": 1.0, "food": 1.0, "danger": 1.0, "move": 1.0}
        name = regime_name.lower()

        if "r1" in name:
            if self.is_minigrid:
                profile["survive"] = 1.0
                profile["food"] = 1.3
                profile["danger"] = 0.7
                profile["move"] = 0.4
            else:
                profile["survive"] = 1.0
                profile["food"] = 1.0
                profile["danger"] = 1.0
                profile["move"] = 0.1
        elif "r2" in name:
            if self.is_minigrid:
                profile["survive"] = 0.8
                profile["food"] = 1.4
                profile["danger"] = 1.1
                profile["move"] = 0.8
            else:
                profile["survive"] = 1.2
                profile["food"] = 0.1
                profile["danger"] = 1.5
                profile["move"] = -0.2
        elif "r3" in name:
            if self.is_minigrid:
                profile["survive"] = 0.7
                profile["food"] = 0.3
                profile["danger"] = 2.0
                profile["move"] = 0.6
            else:
                profile["survive"] = 0.6
                profile["food"] = 0.2
                profile["danger"] = -1.2
                profile["move"] = -0.4
        elif "r4" in name or "door" in name:
            if self.is_minigrid:
                profile["survive"] = 0.9
                profile["food"] = 0.8
                profile["danger"] = 1.2
                profile["move"] = 0.6
            else:
                profile["survive"] = 1.0
                profile["food"] = 0.6
                profile["danger"] = 0.8
                profile["move"] = -0.1
        elif "r5" in name or "lava" in name:
            profile["survive"] = 1.2
            profile["food"] = 0.2
            profile["danger"] = -1.4
            profile["move"] = -0.3
        elif "tool" in name or "comp" in name:
            profile = {"survive": 0.0, "food": 0.0, "danger": 0.0, "move": -0.05}
        return profile

    def _default_regimes(self) -> List[RegimeConfig]:
        """
        Extended default regimes over existing scenario types and reward profiles:
          - R1..R5 cover safe/food-rich/danger/door/lava styles,
          - R_tools / computer regimes added when available.
        """
        if self.env_family == "tools":
            return [
                RegimeConfig(
                    name="tools_basic",
                    scenario_weights={"tools_basic": 1.0},
                    reward_profile={"survive": 0.0, "food": 0.0, "danger": 0.0, "move": 0.0},
                    description="ToolEnv: базовая арифметика, безопасный режим.",
                )
            ]
        if self.env_family == "computer":
            base_profile = {"survive": 0.0, "food": 0.0, "danger": 0.0, "move": -0.05}
            return [
                RegimeConfig(
                    name="comp_simple",
                    scenario_weights={"simple_project": 1.0},
                    reward_profile=base_profile,
                    description="Компьютер: простая задача (1-2 теста).",
                ),
                RegimeConfig(
                    name="comp_refactor",
                    scenario_weights={"refactor_project": 1.0},
                    reward_profile=base_profile,
                    description="Компьютер: рефакторинг, 3-5 тестов.",
                ),
                RegimeConfig(
                    name="comp_flaky",
                    scenario_weights={"flaky_tests_project": 1.0},
                    reward_profile=base_profile,
                    description="Компьютер: флейковые тесты, контроль шагов.",
                ),
            ]

        if self.is_minigrid:
            regimes = [
                RegimeConfig(
                    name="R1",
                    scenario_weights={
                        "minigrid-empty": 0.6,
                        "empty_easy": 0.4,
                    },
                    reward_profile=self._reward_profile_for_regime("R1"),
                    description="Базовый безопасный (empty / easy).",
                ),
                RegimeConfig(
                    name="R2",
                    scenario_weights={
                        "minigrid-empty": 0.3,
                        "minigrid-doorkey": 0.7,
                    },
                    reward_profile=self._reward_profile_for_regime("R2"),
                    description="Food-rich + doorkey.",
                ),
                RegimeConfig(
                    name="R3",
                    scenario_weights={
                        "minigrid-lavacrossing": 0.6,
                        "lava_gap": 0.4,
                    },
                    reward_profile=self._reward_profile_for_regime("R3"),
                    description="Danger-heavy лавовые сценарии.",
                ),
                RegimeConfig(
                    name="R4_doorkey",
                    scenario_weights={
                        "minigrid-doorkey": 1.0,
                    },
                    reward_profile=self._reward_profile_for_regime("R4"),
                    description="Doorkey длинный горизонт.",
                ),
                RegimeConfig(
                    name="R5_lavacrossing",
                    scenario_weights={
                        "minigrid-lavacrossing": 0.8,
                        "lava_gap": 0.2,
                    },
                    reward_profile=self._reward_profile_for_regime("R5"),
                    description="LavaCrossing длинный горизонт.",
                ),
            ]
        else:
            regimes = [
                RegimeConfig(
                    name="R1",
                    scenario_weights={
                        "balanced": 0.4,
                        "food_rich": 0.25,
                        "empty": 0.35,
                    },
                    reward_profile=self._reward_profile_for_regime("R1"),
                    description="Базовый безопасный (empty/balanced).",
                ),
                RegimeConfig(
                    name="R2",
                    scenario_weights={
                        "food_rich": 0.6,
                        "balanced": 0.2,
                        "sparse_large": 0.2,
                    },
                    reward_profile=self._reward_profile_for_regime("R2"),
                    description="Food-rich, быстрый сбор ресурсов.",
                ),
                RegimeConfig(
                    name="R3",
                    scenario_weights={
                        "dangerous": 0.5,
                        "lavacrossing": 0.3,
                        "sparse_large": 0.2,
                    },
                    reward_profile=self._reward_profile_for_regime("R3"),
                    description="Опасные зоны, штраф за урон.",
                ),
                RegimeConfig(
                    name="R4_doorkey",
                    scenario_weights={
                        "door_key": 0.7,
                        "balanced": 0.3,
                    },
                    reward_profile=self._reward_profile_for_regime("R4"),
                    description="Door/Key аналог (больше шагов).",
                ),
                RegimeConfig(
                    name="R5_lavacrossing",
                    scenario_weights={
                        "lavacrossing": 0.8,
                        "dangerous": 0.2,
                    },
                    reward_profile=self._reward_profile_for_regime("R5"),
                    description="Lavacrossing/опасность, длинный горизонт.",
                ),
            ]
        if self.has_computer_env:
            regimes.append(
                RegimeConfig(
                    name="R_tools",
                    scenario_weights={
                        "simple_project": 0.5,
                        "refactor_project": 0.5,
                    },
                    reward_profile={"survive": 0.0, "food": 0.0, "danger": 0.0, "move": -0.05},
                    description="Компьютерный режим: проекты с тестами.",
                )
            )
        return regimes

    # ----- shared self-reflection step -----

    def _run_self_reflection_step(
        self,
        batch: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        optim: torch.optim.Optimizer,
        lambda_reg: float,
        lambda_conflict: float,
        reward_profile: Optional[Dict[str, float]] = None,
    ):
        """
        Single self-reflection update on traits (same objective as Stage 3b).
        Updates only self.agent.traits using the provided optimizer.
        """
        (
            obs_seq,
            H_seq,
            a_seq,
            r_seq,
            d_seq,
            death_seq,
            food_seq,
            dmg_seq,
            move_seq,
            alive_seq,
            scenario_seq,
            env_seq,
        ) = batch

        B, T, p, _ = obs_seq.shape

        patch = torch.from_numpy(obs_seq).long().to(self.device)
        H = torch.from_numpy(H_seq).float().to(self.device)
        a = torch.from_numpy(a_seq).long().to(self.device)
        r = torch.from_numpy(r_seq).float().to(self.device)
        scenario = torch.from_numpy(scenario_seq).long().to(self.device)
        env_ids = torch.from_numpy(env_seq).long().to(self.device)
        env_desc_seq = self._env_desc_from_ids(env_ids)

        with torch.no_grad():
            patch_flat = patch.view(B * T, p, p)
            H_flat = H.view(B * T, 1)
            scenario_flat = scenario.view(B * T)
            env_flat = env_ids.view(B * T)
            env_desc_flat = env_desc_seq.reshape(B * T, -1) if env_desc_seq is not None else None

            text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
            z_flat = self.agent.perception(
                patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
            )
            z_seq = z_flat.reshape(B, T, -1)

            a_emb = self.agent.world_model.act_emb(a)
            x_w = torch.cat([z_seq, H, a_emb], dim=-1)
            h0 = torch.zeros(
                1,
                B,
                self.agent.world_model.gru.hidden_size,
                device=self.device,
            )
            out_w, _ = self.agent.world_model.gru(x_w, h0)
            W_seq = out_w  # (B,T,w_dim)

        traits = self._mixed_traits()  # (1,4)
        M = self.agent.memory       # (1,mem_dim)

        (
            S_seq,
            S_last,
            surv_pred,
            food_pred,
            dmg_pred,
            move_pred,
            unc_pred,
            surv_raw_pred,
        ) = self.agent.self_model.forward_seq(
            W_seq,
            H,
            a,
            r,
            M=M,
            env_desc=env_desc_seq,
        )

        base_lrs = [g.get("lr", 0.0) for g in optim.param_groups]
        step_size_eff = base_lrs[0] if base_lrs else 0.0
        if self.is_minigrid:
            step_size_eff = float(self.trait_reflection_lr)
        n_steps_eff = 1 if not self.is_minigrid else max(1, self.trait_reflection_steps_per_batch)
        if optim.param_groups:
            for g in optim.param_groups:
                g["lr"] = step_size_eff

        step_norms: List[float] = []
        traits_prev = self.agent.traits.detach().clone()
        try:
            surv0 = surv_pred[:, 0, 0]   # (B,)
            food0 = food_pred[:, 0, 0]
            dmg0 = dmg_pred[:, 0, 0]
            move0 = move_pred[:, 0, 0]

            feat_seq = torch.stack(
                [
                    surv_pred.squeeze(-1),   # (B,T)
                    food_pred.squeeze(-1),
                    dmg_pred.squeeze(-1),
                    move_pred.squeeze(-1),
                ],
                dim=-1,                    # (B,T,4)
            )

            with torch.no_grad():
                B_, T_, w_dim = W_seq.shape
                W_flat = W_seq.reshape(B_ * T_, w_dim)
                H_flat2 = H.reshape(B_ * T_, -1)
                M_vm = self.agent.memory.detach().expand(B_ * T_, -1)

            for _ in range(n_steps_eff):
                with torch.no_grad():
                    traits_vm = self._main_traits().detach().expand(B_ * T_, -1)
                    V_flat = self.agent.value_model(W_flat, H_flat2, traits_vm, M_vm)
                    V_seq = V_flat.view(B_, T_, 1)

                w_raw = traits_to_preference_weights(self._main_traits())  # (1,4)
                if reward_profile:
                    alpha_vec = torch.tensor(
                        [
                            float(reward_profile.get("survive", 1.0)),
                            float(reward_profile.get("food", 1.0)),
                            float(reward_profile.get("danger", 1.0)),
                            float(reward_profile.get("move", 1.0)),
                        ],
                        device=w_raw.device,
                        dtype=w_raw.dtype,
                    ).view(1, 4)
                    w_eff = w_raw * alpha_vec
                else:
                    w_eff = w_raw
                w_survive, w_food, w_danger, w_move = w_eff[0]
                w_prior_survive, w_prior_food, w_prior_danger, w_prior_move = w_raw[0]

                base_term = (
                    w_survive * surv0
                    + w_food * food0
                    + w_danger * dmg0
                    + w_move * move0
                )  # (B,)

                reg = (self.agent.traits ** 2).mean()

                w_seq = w_eff.view(1, 1, 4)    # (1,1,4)
                U_seq = (feat_seq * w_seq).sum(dim=-1, keepdim=True)  # (B,T,1)

                conflict = torch.abs(U_seq - V_seq)
                loss_conflict = conflict.mean()

                prior_terms = []
                prior_terms.append(F.relu(0.5 - w_prior_survive))
                prior_terms.append(F.relu(w_prior_danger + 0.4))
                prior_terms.append(F.relu(w_prior_move + 0.3))
                prior_terms.append(F.relu(w_prior_move - 0.8))
                prior_loss = torch.stack(prior_terms).mean()

                loss = (
                    -base_term.mean()
                    + lambda_reg * reg
                    + lambda_conflict * loss_conflict
                    + 1.0 * prior_loss
                )

                optim.zero_grad()
                loss.backward()
                optim.step()
                with torch.no_grad():
                    step_norms.append(float(torch.norm(self.agent.traits - traits_prev).item()))
                    traits_prev = self.agent.traits.detach().clone()
        finally:
            if optim.param_groups:
                for g, lr_orig in zip(optim.param_groups, base_lrs):
                    g["lr"] = lr_orig

        # clamp traits post-update
        with torch.no_grad():
            self.agent.traits.clamp_(-2.0, 2.0)

        updated = bool(step_norms and max(step_norms) > 1e-6)
        steps = int(n_steps_eff)
        stats = self.trait_reflection_debug.setdefault(
            "self_reflection",
            {
                "trait_reflection_n_calls": 0,
                "trait_reflection_n_updates": 0,
                "trait_reflection_total_steps": 0,
                "trait_reflection_step_norm_sum": 0.0,
            },
        )
        stats["trait_reflection_n_calls"] += 1
        stats["trait_reflection_total_steps"] += steps
        if updated:
            stats["trait_reflection_n_updates"] += 1
            stats["trait_reflection_step_norm_sum"] += float(sum(step_norms))

        return {"updated": updated, "steps": steps, "step_norms": step_norms}

    # ----- reward from traits + events -----

    def compute_preference_reward(
        self, info: Dict[str, Any], reward_profile: Optional[Dict[str, float]] = None
    ) -> float:
        env_info = info or {}
        if "reward_env" in env_info:
            try:
                return float(env_info["reward_env"])
            except Exception:
                pass
        with torch.no_grad():
            w = self._combined_preference_weights()  # (1,4)
            w_survive, w_food, w_danger, w_move = w.view(-1).tolist()

        profile = reward_profile or {}
        alpha_survive = float(profile.get("survive", 1.0))
        alpha_food = float(profile.get("food", 1.0))
        alpha_danger = float(profile.get("danger", 1.0))
        alpha_move = float(profile.get("move", 1.0))

        def _safe_float(val: Any, default: float) -> float:
            try:
                if val is None:
                    return float(default)
                return float(val)
            except Exception:
                return float(default)

        alive = _safe_float(env_info.get("alive", 1.0), 1.0)
        got_food = _safe_float(env_info.get("got_food", 0.0), 0.0)
        took_damage = _safe_float(env_info.get("took_damage", 0.0), 0.0)
        moved = _safe_float(env_info.get("moved", 0.0), 0.0)

        r = (
            alpha_survive * w_survive * alive
            + alpha_food * w_food * got_food
            + alpha_danger * w_danger * took_damage
            + alpha_move * w_move * moved
        )
        return float(r)

    @staticmethod
    def _infer_episode_success_from_info(ep_info: Dict[str, Any]) -> Optional[bool]:
        """
        Best-effort terminal success inference across repo/instruction/social env families.
        """
        if not isinstance(ep_info, dict):
            return None
        if "instruction_success" in ep_info:
            v = ep_info.get("instruction_success")
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return bool(float(v) >= 0.5)
        if "social_success" in ep_info:
            v = ep_info.get("social_success")
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return bool(float(v) >= 0.5)
        if "last_test_passed" in ep_info:
            return bool(ep_info.get("last_test_passed"))

        reason = str(ep_info.get("reason", "")).strip().lower()
        if reason in {"took_correct_goal", "you_got_food", "food_collected"}:
            return True
        if reason in {"took_wrong_goal", "other_got_food"}:
            return False
        if reason in {"max_steps", "energy_depleted"}:
            return None

        reward_env = ep_info.get("reward_env")
        if isinstance(reward_env, (int, float)) and math.isfinite(float(reward_env)):
            v = float(reward_env)
            if v > 0.0:
                return True
            if v < 0.0:
                return False
        return None

    def _record_regime_stats(
        self,
        regime_name: str,
        returns: List[float],
        survival_flags: List[float],
        food_counts: List[int],
        damage_counts: List[int],
        uncertainty_vals: List[float],
        baseline_return: float,
        move_counts: Optional[List[int]] = None,
    ) -> RegimeStats:
        episodes = len(returns)
        avg_return = float(np.mean(returns)) if returns else 0.0
        avg_survival = float(np.mean(survival_flags)) if survival_flags else 0.0
        avg_food = float(np.mean(food_counts)) if food_counts else 0.0
        avg_damage = float(np.mean(damage_counts)) if damage_counts else 0.0
        avg_uncertainty = float(np.mean(uncertainty_vals)) if uncertainty_vals else 0.0
        forgetting_gap = float(baseline_return - avg_return)
        moves = move_counts if move_counts is not None else [0 for _ in range(episodes)]
        safety_utils: List[float] = []
        try:
            weights_all = get_faction_preference_weights(self.agent)
            if weights_all.dim() == 1 or weights_all.size(0) == 1:
                w_safety = weights_all[0]
            else:
                w_safety = weights_all[1]
            for surv, food, dmg, mv in zip(survival_flags, food_counts, damage_counts, moves):
                comps = torch.tensor([surv, food, dmg, mv], device=self.device, dtype=torch.float32)
                safety_utils.append(float((w_safety.to(comps.device) * comps).sum().item()))
        except Exception:
            safety_utils = []
        avg_safety_utility = float(np.mean(safety_utils)) if safety_utils else 0.0
        danger_score = avg_damage
        prev = self.regime_stats.get(regime_name)
        total_eps = episodes + (prev["episodes_seen"] if prev else 0)
        stats: RegimeStats = {
            "regime_name": regime_name,
            "episodes_seen": int(total_eps),
            "avg_return": avg_return,
            "avg_survival": avg_survival,
            "avg_food": avg_food,
            "avg_damage": avg_damage,
            "forgetting_gap": forgetting_gap,
            "uncertainty": avg_uncertainty,
            "avg_safety_utility": avg_safety_utility,
            "danger_score": danger_score,
        }
        self.regime_stats[regime_name] = stats
        if self.logger is not None:
            payload: Dict[str, Any] = {"event": "regime_stats_update", **stats}
            self.logger.log(payload)
        return stats

    def _apply_regime_proposals(self, proposals: List[RegimeProposal]) -> Tuple[List[str], List[str]]:
        new_regimes: List[str] = []
        changed_regimes: List[str] = []
        if not proposals:
            return new_regimes, changed_regimes

        for prop in proposals:
            stats_for_regime = self.regime_stats.get(prop.name)
            priority_before = self.regime_priorities.get(prop.name)
            if prop.name in self.regimes:
                self.regime_priorities[prop.name] = prop.priority
                self.regime_env_descriptors[prop.name] = prop.env_descriptors
                changed_regimes.append(prop.name)
            else:
                max_regs = (
                    self.regime_generator.config.max_active_regimes
                    if self.regime_generator is not None
                    else len(self.regimes) + 1
                )
                if len(self.regimes) >= max_regs:
                    continue
                scenario_weights = {}
                if self.regimes:
                    scenario_weights = dict(next(iter(self.regimes.values())).scenario_weights)
                else:
                    scenario_weights = {"balanced": 1.0}
                self.regimes[prop.name] = RegimeConfig(
                    name=prop.name,
                    scenario_weights=scenario_weights,
                    reward_profile=prop.reward_profile,
                )
                self.regime_priorities[prop.name] = prop.priority
                self.regime_env_descriptors[prop.name] = prop.env_descriptors
                new_regimes.append(prop.name)
            if self.logger is not None:
                self.logger.log(
                    {
                        "event": "regime_update",
                        "regime_name": prop.name,
                        "priority_before": priority_before,
                        "priority_after": prop.priority,
                        "avg_safety_utility": stats_for_regime.get("avg_safety_utility") if stats_for_regime else None,
                        "danger_score": stats_for_regime.get("danger_score") if stats_for_regime else None,
                    }
                )

        if self.logger is not None and (new_regimes or changed_regimes):
            self.logger.log(
                {
                    "event": "regime_update",
                    "new_regimes": list(new_regimes),
                    "changed_regimes": list(changed_regimes),
                }
            )
        return new_regimes, changed_regimes

    def _regime_base_key(self, regime_name: str) -> str:
        name = str(regime_name or "").strip()
        if name.endswith("_return"):
            return name[: -len("_return")]
        return name

    def _set_main_traits(self, main_traits: torch.Tensor) -> None:
        target = main_traits.detach()
        if target.dim() == 1:
            target = target.view(1, -1)
        target = target.to(device=self.agent.traits.device, dtype=self.agent.traits.dtype)
        with torch.no_grad():
            if self.agent.traits.dim() == 2 and self.agent.traits.size(0) > 1:
                self.agent.traits.data[0:1] = target[0:1]
            else:
                self.agent.traits.data = target
            self.agent.traits.data.clamp_(-2.0, 2.0)

    def _recall_lifelong_traits(
        self,
        current_traits: torch.Tensor,
        regime_name: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        regime = str(regime_name or "")
        base_key = self._regime_base_key(regime)
        current_main = current_traits
        if current_main.dim() == 1:
            current_main = current_main.view(1, -1)
        current_main = current_main[:1].detach()
        memory = self.lifelong_trait_memory.get(base_key)
        recall_weight = 0.8 if regime.endswith("_return") else 0.3
        info: Dict[str, Any] = {
            "base_key": base_key,
            "memory_used": False,
            "recall_weight": recall_weight,
            "score": self.lifelong_trait_memory_score.get(base_key),
        }
        if not isinstance(memory, torch.Tensor):
            return current_main, info
        mem = memory.detach()
        if mem.dim() == 1:
            mem = mem.view(1, -1)
        mem = mem[:1].to(device=current_main.device, dtype=current_main.dtype)
        blended = recall_weight * mem + (1.0 - recall_weight) * current_main
        info["memory_used"] = True
        return blended, info

    def _update_lifelong_trait_memory_if_better(
        self,
        regime_name: str,
        mean_return: float,
        traits_main: torch.Tensor,
    ) -> bool:
        base_key = self._regime_base_key(regime_name)
        prev = self.lifelong_trait_memory_score.get(base_key)
        if prev is not None and float(mean_return) <= float(prev):
            return False
        store = traits_main.detach()
        if store.dim() == 1:
            store = store.view(1, -1)
        self.lifelong_trait_memory[base_key] = store[:1].cpu().clone()
        self.lifelong_trait_memory_score[base_key] = float(mean_return)
        return True

    def _lifelong_reflection_schedule(
        self,
        *,
        episode_idx: int,
        episodes_per_chapter: int,
        base_steps: int,
        base_step_size: float,
        base_lambda_prior: float,
        high_safety_risk: bool = False,
    ) -> Dict[str, Any]:
        n_eps = max(1, int(episodes_per_chapter))
        ep = max(0, int(episode_idx))
        early_cut = max(1, int(math.ceil(float(n_eps) / 3.0)))
        late_start = max(0, int(math.floor((2.0 * float(n_eps)) / 3.0)))
        phase = "mid"
        steps = int(base_steps)
        step_size = float(base_step_size)
        lambda_prior = float(base_lambda_prior)
        if ep < early_cut:
            phase = "early"
            steps = int(base_steps) + int(getattr(self, "lifelong_reflect_early_step_boost", 2))
            step_size = float(base_step_size) * float(getattr(self, "lifelong_reflect_early_step_size_scale", 1.2))
            lambda_prior = float(base_lambda_prior) * float(getattr(self, "lifelong_reflect_early_lambda_prior_scale", 0.5))
        elif ep >= late_start:
            phase = "late"
            steps = int(base_steps) + int(getattr(self, "lifelong_reflect_late_step_delta", -1))
            step_size = float(base_step_size) * float(getattr(self, "lifelong_reflect_late_step_size_scale", 0.8))
            lambda_prior = float(base_lambda_prior) * float(getattr(self, "lifelong_reflect_late_lambda_prior_scale", 1.5))

        if bool(high_safety_risk):
            step_size = step_size * float(getattr(self, "lifelong_reflect_safety_step_size_scale", 0.8))
            lambda_prior = lambda_prior * float(getattr(self, "lifelong_reflect_safety_lambda_prior_scale", 1.5))

        steps = int(max(1, min(12, steps)))
        step_size = float(max(0.005, min(0.08, step_size)))
        lambda_prior = float(max(0.001, min(0.02, lambda_prior)))
        return {
            "phase": phase,
            "steps": steps,
            "step_size": step_size,
            "lambda_prior": lambda_prior,
            "high_safety_risk": bool(high_safety_risk),
        }

    def _compute_lifelong_past_regime_weights(self, eps: float = 1.0e-3) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for regime_name, stats in self.regime_stats.items():
            if not isinstance(stats, dict):
                continue
            try:
                forgetting_gap = float(stats.get("forgetting_gap", 0.0))
            except Exception:
                forgetting_gap = 0.0
            try:
                uncertainty = float(stats.get("uncertainty", 0.0))
            except Exception:
                uncertainty = 0.0
            w = float(eps + max(0.0, forgetting_gap) + 0.5 * max(0.0, uncertainty))
            if math.isfinite(w) and w > 0.0:
                weights[str(regime_name)] = w
        return weights

    # =========================
    #  Stage 1: random experience
    # =========================

    def collect_random_experience(self, n_steps: int):
        """Stage 1: collect random rollouts for world-model pretraining."""
        self.current_regime_name = "stage1_random"
        obs = self.env.reset(split="train")
        patch = obs["patch"]
        energy = obs["energy"]
        env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

        for _ in range(n_steps):
            action = self.env.rng.randint(0, self.env.n_actions)
            next_obs, _, done, info = self.env.step(int(action))

            reward_env = self.compute_preference_reward(info)
            death_flag = float(info.get("death_flag", 0.0))

            got_food = 1.0 if info.get("got_food", False) else 0.0
            took_damage = 1.0 if info.get("took_damage", False) else 0.0
            moved = 1.0 if info.get("moved", False) else 0.0
            alive = 1.0 if info.get("alive", True) else 0.0
            scenario_id = int(info.get("scenario_id", 0))
            env_id = int(info.get("env_id", env_id))

            tr = Transition(
                obs_patch=patch,
                energy=energy,
                action=int(action),
                reward=reward_env,
                done=done,
                next_obs_patch=next_obs["patch"],
                next_energy=next_obs["energy"],
                death_flag=death_flag,
                got_food=got_food,
                took_damage=took_damage,
                moved=moved,
                alive=alive,
                scenario_id=scenario_id,
                env_id=env_id,
                regime_name=self.current_regime_name or "stage1",
            )
            self.buffer.push(tr)

            patch = next_obs["patch"]
            energy = next_obs["energy"]
            env_id = int(next_obs.get("env_id", env_id))

            if done:
                obs = self.env.reset(split="train")
                patch = obs["patch"]
                energy = obs["energy"]
                env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

    def train_world_model(
        self,
        batch_size: int = 32,
        seq_len: int = 16,
        n_batches: int = 200,
    ):
        """Stage 1: supervised world-model training on buffered data."""
        self.current_regime_name = "stage1_world_model"
        self.agent.perception.train()
        self.agent.world_model.train()

        for _ in range(n_batches):
            if len(self.buffer) < seq_len:
                break
            (
                obs_seq,
                H_seq,
                a_seq,
                r_seq,
                d_seq,
                death_seq,
                scenario_seq,
                env_seq,
            ) = self._sample_sequences_from_buffer(
                self.buffer, batch_size, seq_len, with_events=False, current_regime=self.current_regime_name
            )

            patch = torch.from_numpy(obs_seq).long().to(self.device)
            H = torch.from_numpy(H_seq).float().to(self.device)
            a = torch.from_numpy(a_seq).long().to(self.device)
            scenario = torch.from_numpy(scenario_seq).long().to(self.device)
            env_ids = torch.from_numpy(env_seq).long().to(self.device)

            B, T, p, _ = patch.shape
            patch_flat = patch.view(B * T, p, p)
            H_flat = H.view(B * T, 1)
            scenario_flat = scenario.view(B * T)
            env_flat = env_ids.view(B * T)
            env_desc_flat = self._env_desc_from_ids(env_flat)

            text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
            z_flat = self.agent.perception(
                patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
            )
            z_seq = z_flat.reshape(B, T, -1)

            loss = self.agent.world_model.loss_supervised(z_seq, H, a)

            self.agent.optim_world.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.agent.perception.parameters())
                + list(self.agent.world_model.parameters()),
                1.0,
            )
            self.agent.optim_world.step()

    def train_latent_skills(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Backwards-compatible entrypoint for latent skill training.
        Currently runs supervised distillation via train_latent_skills_supervised().
        """
        self.train_latent_skills_supervised()
        return {"mode": "supervised", "n_latent_skills": int(getattr(self, "n_latent_skills", 0))}

    # =========================
    #  Multi-step planner
    # =========================

    def compute_planner_logits(
        self,
        z_obs: torch.Tensor,      # (1, obs_dim)
        H_t: torch.Tensor,        # (1, h_dim)
        h_w: torch.Tensor,        # (1, 1, w_dim)
        traits: torch.Tensor,     # (1, trait_dim)
        M: torch.Tensor,          # (1, mem_dim)
        horizon: Optional[int] = None,
        gamma: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Multi-step planner v1 ("repeat"):
          - repeats action a for horizon steps via world model rollout;
          - scores with ValueModel under main + safety traits and applies safety penalty.
        """
        device = z_obs.device
        A = self.env.n_actions
        horizon = horizon or self.planning_horizon
        gamma = gamma or self.planner_gamma

        z = z_obs.expand(A, -1).contiguous()  # (A, obs_dim)
        H = H_t.expand(A, -1).contiguous()    # (A, h_dim)
        h_w_local = h_w.expand(1, A, -1).contiguous()  # (1, A, w_dim)
        a_cand = torch.arange(A, device=device)

        scores_main = torch.zeros(A, device=device)
        scores_safety = torch.zeros(A, device=device)

        traits_main = traits if traits is not None else self._mixed_traits()
        if traits_main.dim() == 1:
            traits_main = traits_main.view(1, -1)
        traits_rep = traits_main.expand(A, -1)
        traits_safety = self._safety_traits()
        traits_rep_safety = traits_safety.expand(A, -1)
        M_rep = M.expand(A, -1)

        for t in range(horizon):
            w_t, h_w_local, z_hat, H_hat = self.agent.world_model.forward_step(
                z, H, a_cand, h_w_local
            )

            V_main = self.agent.value_model(w_t, H_hat, traits_rep, M_rep).squeeze(-1)
            V_safety = self.agent.value_model(w_t, H_hat, traits_rep_safety, M_rep).squeeze(-1)

            scores_main = scores_main + (gamma ** t) * V_main
            scores_safety = scores_safety + (gamma ** t) * V_safety

            z = z_hat.detach()
            H = H_hat.detach()

        penalized = self._apply_safety_penalty(scores_main, scores_safety)
        penalized = penalized - penalized.mean()
        std = penalized.std()
        if std > 1e-6:
            penalized = penalized / std

        return penalized.unsqueeze(0)

    def _planner_skill_logits(
        self,
        z_obs: torch.Tensor,
        H_t: torch.Tensor,
        h_w: torch.Tensor,
        traits: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plan over discrete skills by simulating each skill's macro-action sequence.
        Returns logits over skill ids (1, n_skills).
        """
        total_skills = self._total_skill_count()
        if total_skills <= 0:
            return torch.zeros(1, 1, device=self.device)
        if not self.skills:
            return torch.zeros(1, total_skills, device=self.device)
        env_desc = torch.zeros(
            (1, self.env_desc_dim if self.env_desc_dim is not None else 1),
            device=self.device,
        )
        context = SkillContext(
            w=h_w.squeeze(0),
            h=H_t,
            traits=traits,
            env_desc=env_desc,
            step_in_skill=0,
        )
        scores: List[float] = []
        for skill in self.skills:
            seq = skill.rollout(context)
            if not seq:
                scores.append(0.0)
                continue
            h_w_local = h_w.clone()
            z_cur = z_obs
            H_cur = H_t
            score = 0.0
            for a in seq:
                a_t = torch.tensor([int(a)], device=self.device)
                w_t, h_w_local, z_hat, H_hat = self.agent.world_model.forward_step(
                    z_cur, H_cur, a_t, h_w_local
                )
                V_pi = self.agent.value_model(w_t, H_cur, traits, M)
                score += float(V_pi.item())
                z_cur = z_hat.detach()
                H_cur = H_hat.detach()
            scores.append(score)
        scores_t = torch.tensor(scores, device=self.device).unsqueeze(0)
        scores_t = scores_t - scores_t.mean()
        std = scores_t.std()
        if std > 1e-6:
            scores_t = scores_t / std
        if total_skills > len(self.skills):
            pad = torch.zeros(1, total_skills - len(self.skills), device=self.device)
            scores_t = torch.cat([scores_t, pad], dim=1)
        return scores_t

    def _planner_logits_multistep(
        self,
        z_obs: torch.Tensor,   # (1, obs_dim)
        H_t: torch.Tensor,     # (1, h_dim)
        h_w: torch.Tensor,     # (1, 1, w_dim)
        traits: torch.Tensor,  # (1, trait_dim)
        M: torch.Tensor,       # (1, mem_dim)
    ) -> torch.Tensor:
        """
        Batched multi-step planner ("rollout") with safety-aware scoring.
        """
        A = self.env.n_actions
        R = self.planner_rollouts
        H = self.planning_horizon
        gamma = self.planner_gamma
        device = self.device

        with torch.no_grad():
            batch = A * R
            z = z_obs.expand(batch, -1).contiguous()
            H_cur = H_t.expand(batch, -1).contiguous()
            h_cur = h_w.expand(1, batch, -1).contiguous()

            traits_main = traits if traits is not None else self._mixed_traits()
            if traits_main.dim() == 1:
                traits_main = traits_main.view(1, -1)
            traits_rep = traits_main.expand(batch, -1)
            traits_safety = self._safety_traits()
            traits_rep_safety = traits_safety.expand(batch, -1)
            M_rep = M.expand(batch, -1)

            a = torch.arange(A, device=device).repeat_interleave(R)
            scores_main = torch.zeros(batch, device=device)
            scores_safety = torch.zeros(batch, device=device)
            disc = torch.ones(batch, device=device)

            for _ in range(H):
                W, h_cur, z_hat, H_hat = self.agent.world_model.forward_step(
                    z, H_cur, a, h_cur
                )

                V_main = self.agent.value_model(W, H_hat, traits_rep, M_rep).view(-1)
                V_safety = self.agent.value_model(W, H_hat, traits_rep_safety, M_rep).view(-1)
                scores_main = scores_main + disc * V_main
                scores_safety = scores_safety + disc * V_safety
                disc = disc * gamma

                S_zero = torch.zeros(
                    batch,
                    self.agent.self_model.gru.hidden_size,
                    device=device,
                )
                R_zero = torch.zeros(batch, 1, device=device)
                U_zero = torch.zeros(batch, 1, device=device)

                G_t = self.agent.workspace(
                    W,
                    S_zero,
                    H_hat,
                    V_main.view(batch, 1),
                    R_zero,
                    U_zero,
                    traits_rep,
                    M_rep,
                )
                logits_pi = self.agent.policy(G_t)
                dist_pi = Categorical(logits=logits_pi)
                a = dist_pi.sample()

                z = z_hat.detach()
                H_cur = H_hat.detach()

            q_main = scores_main.view(A, R).mean(dim=1)
            q_safety = scores_safety.view(A, R).mean(dim=1)
            penalized = self._apply_safety_penalty(q_main, q_safety)
            penalized = penalized - penalized.mean()
            std = penalized.std()
            if std > 1e-6:
                penalized = penalized / std

        return penalized.unsqueeze(0)

    def _planner_logits_beam(
        self,
        z_obs: torch.Tensor,   # (1, obs_dim)
        H_t: torch.Tensor,     # (1, h_dim)
        h_w: torch.Tensor,     # (1, 1, w_dim)
        traits: torch.Tensor,  # (1, trait_dim)
        M: torch.Tensor,       # (1, mem_dim)
    ) -> torch.Tensor:
        """
        Beam-search planner over primitive actions using the learned world model.

        Notes:
          - Uses `planner_rollouts` as the beam width (keeps CLI stable).
          - Ranks partial sequences by safety-penalized score but returns unpenalized
            (q_main, q_safety) so the final safety penalty is applied only once.
        """
        A = int(self.env.n_actions)
        B = int(max(1, self.planner_rollouts))
        H = int(max(1, self.planning_horizon))
        gamma = float(self.planner_gamma)
        device = self.device

        with torch.no_grad():
            traits_main = traits if traits is not None else self._mixed_traits()
            if traits_main.dim() == 1:
                traits_main = traits_main.view(1, -1)
            traits_safety = self._safety_traits()
            if traits_safety.dim() == 1:
                traits_safety = traits_safety.view(1, -1)

            q_main = torch.zeros(A, device=device, dtype=z_obs.dtype)
            q_safety = torch.zeros(A, device=device, dtype=z_obs.dtype)

            threshold = torch.as_tensor(self.safety_threshold, device=device, dtype=z_obs.dtype)
            coef = float(self.safety_penalty_coef)

            for a0 in range(A):
                a0_t = torch.tensor([int(a0)], device=device, dtype=torch.long)
                W0, h_cur, z_cur, H_cur = self.agent.world_model.forward_step(z_obs, H_t, a0_t, h_w)
                V0_main = self.agent.value_model(W0, H_cur, traits_main, M).view(-1)
                V0_safety = self.agent.value_model(W0, H_cur, traits_safety, M).view(-1)

                beam_z = z_cur.detach()
                beam_H = H_cur.detach()
                beam_h = h_cur.detach()
                beam_main = V0_main.clone()
                beam_safety = V0_safety.clone()

                disc = gamma
                for _ in range(1, H):
                    beam_size = int(beam_z.size(0))
                    z_rep = beam_z.repeat_interleave(A, dim=0)
                    H_rep = beam_H.repeat_interleave(A, dim=0)
                    h_rep = beam_h.repeat_interleave(A, dim=1)
                    a_rep = torch.arange(A, device=device, dtype=torch.long).repeat(beam_size)

                    Wn, h_next, z_next, H_next = self.agent.world_model.forward_step(z_rep, H_rep, a_rep, h_rep)
                    n = int(z_next.size(0))
                    traits_rep = traits_main.expand(n, -1)
                    traits_rep_safety = traits_safety.expand(n, -1)
                    M_rep = M.expand(n, -1)

                    V_main = self.agent.value_model(Wn, H_next, traits_rep, M_rep).view(-1)
                    V_safety = self.agent.value_model(Wn, H_next, traits_rep_safety, M_rep).view(-1)

                    main_new = beam_main.repeat_interleave(A) + disc * V_main
                    safety_new = beam_safety.repeat_interleave(A) + disc * V_safety

                    gap = torch.clamp(threshold - safety_new, min=0.0)
                    penalized = main_new - coef * gap
                    k = int(min(B, int(penalized.numel())))
                    top = torch.topk(penalized, k=k)
                    idx = top.indices

                    beam_z = z_next[idx].detach()
                    beam_H = H_next[idx].detach()
                    beam_h = h_next[:, idx, :].detach()
                    beam_main = main_new[idx]
                    beam_safety = safety_new[idx]
                    disc = disc * gamma

                gap = torch.clamp(threshold - beam_safety, min=0.0)
                penalized = beam_main - coef * gap
                best = int(torch.argmax(penalized).item()) if penalized.numel() > 0 else 0
                q_main[a0] = beam_main[best]
                q_safety[a0] = beam_safety[best]

            penalized = self._apply_safety_penalty(q_main, q_safety)
            penalized = penalized - penalized.mean()
            std = penalized.std()
            if std > 1e-6:
                penalized = penalized / std
            return penalized.unsqueeze(0)

    def _get_planner_logits(
        self,
        z_obs: torch.Tensor,
        H_t: torch.Tensor,
        h_w: torch.Tensor,
        traits: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """
        Унифицированный интерфейс планировщика:
          - planner_mode="repeat"  → compute_planner_logits (v1),
          - planner_mode="rollout" → батчевый _planner_logits_multistep.
        """
        if self.planner_mode in {"none", "", None}:
            return torch.zeros((1, self.env.n_actions), device=self.device)
        if self.use_skills and self.planner_mode == "skills":
            return self._planner_skill_logits(
                z_obs=z_obs,
                H_t=H_t,
                h_w=h_w,
                traits=traits,
                M=M,
            )

        if self.planner_mode == "repeat":
            return self.compute_planner_logits(
                z_obs=z_obs,
                H_t=H_t,
                h_w=h_w,
                traits=traits,
                M=M,
                horizon=self.planning_horizon,
                gamma=self.planner_gamma,
            )
        elif self.planner_mode == "rollout":
            return self._planner_logits_multistep(
                z_obs=z_obs,
                H_t=H_t,
                h_w=h_w,
                traits=traits,
                M=M,
            )
        elif self.planner_mode == "beam":
            return self._planner_logits_beam(
                z_obs=z_obs,
                H_t=H_t,
                h_w=h_w,
                traits=traits,
                M=M,
            )
        else:
            raise ValueError(f"Unknown planner_mode: {self.planner_mode}")

    def _planner_reliability(
        self,
        policy_logits: torch.Tensor,
        planner_logits: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        r_self: Optional[torch.Tensor] = None,
        v_pi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reliability estimate for planner guidance.
        rel = sigmoid(1.5*margin) * exp(-js) * exp(-0.7*u) * exp(-0.7*c)
          margin: top1-top2 planner logit margin
          js: Jensen-Shannon divergence between policy/planner distributions
          u: mean uncertainty
          c: mean |R_self - V_pi|
        """
        if policy_logits.dim() == 1:
            policy_logits = policy_logits.unsqueeze(0)
        if planner_logits.dim() == 1:
            planner_logits = planner_logits.unsqueeze(0)
        batch = int(policy_logits.shape[0]) if policy_logits.dim() >= 2 else 1
        device = policy_logits.device
        dtype = policy_logits.dtype

        zeros = torch.zeros(batch, device=device, dtype=dtype)
        out: Dict[str, torch.Tensor] = {
            "reliability": zeros.clone(),
            "margin": zeros.clone(),
            "js": zeros.clone(),
            "valid_mask": torch.zeros(batch, device=device, dtype=torch.bool),
        }
        if planner_logits.shape != policy_logits.shape or policy_logits.dim() != 2:
            return out

        policy_det = policy_logits.detach()
        planner_det = planner_logits.detach()
        valid_mask = torch.isfinite(policy_det).all(dim=-1) & torch.isfinite(planner_det).all(dim=-1)
        if not bool(valid_mask.any().item()):
            return out

        policy_v = policy_det[valid_mask]
        planner_v = planner_det[valid_mask]
        p = torch.softmax(policy_v, dim=-1)
        q = torch.softmax(planner_v, dim=-1)
        m = 0.5 * (p + q)
        eps = torch.tensor(1.0e-8, device=device, dtype=dtype)
        kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
        kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)
        js_v = 0.5 * (kl_pm + kl_qm)

        n_actions = int(policy_logits.shape[-1])
        if n_actions > 1:
            top_vals = torch.topk(planner_v, k=2, dim=-1).values
            margin_v = top_vals[:, 0] - top_vals[:, 1]
        else:
            margin_v = planner_v[:, 0]

        u_v = torch.zeros_like(js_v)
        if isinstance(uncertainty, torch.Tensor):
            unc_det = uncertainty.detach()
            if unc_det.dim() == 0:
                unc_det = unc_det.view(1)
            if unc_det.dim() > 1:
                unc_det = unc_det.reshape(unc_det.shape[0], -1).mean(dim=-1)
            if unc_det.shape[0] == 1 and batch > 1:
                unc_det = unc_det.expand(batch)
            if unc_det.shape[0] == batch:
                unc_valid = unc_det[valid_mask].to(device=device, dtype=dtype)
                unc_valid = torch.nan_to_num(unc_valid, nan=0.0, posinf=0.0, neginf=0.0)
                u_v = torch.clamp(unc_valid, min=0.0)

        c_v = torch.zeros_like(js_v)
        if isinstance(r_self, torch.Tensor) and isinstance(v_pi, torch.Tensor):
            conf_det = torch.abs(r_self.detach() - v_pi.detach())
            if conf_det.dim() == 0:
                conf_det = conf_det.view(1)
            if conf_det.dim() > 1:
                conf_det = conf_det.reshape(conf_det.shape[0], -1).mean(dim=-1)
            if conf_det.shape[0] == 1 and batch > 1:
                conf_det = conf_det.expand(batch)
            if conf_det.shape[0] == batch:
                conf_valid = conf_det[valid_mask].to(device=device, dtype=dtype)
                conf_valid = torch.nan_to_num(conf_valid, nan=0.0, posinf=0.0, neginf=0.0)
                c_v = torch.clamp(conf_valid, min=0.0)

        rel_v = torch.sigmoid(1.5 * margin_v) * torch.exp(-1.0 * js_v) * torch.exp(-0.7 * u_v) * torch.exp(-0.7 * c_v)
        rel_v = torch.clamp(rel_v, min=0.0, max=1.0)

        out["reliability"][valid_mask] = rel_v
        out["margin"][valid_mask] = margin_v
        out["js"][valid_mask] = js_v
        out["valid_mask"] = valid_mask
        return out

    def _blend_with_planner(
        self,
        policy_logits: torch.Tensor,
        planner_logits: Optional[torch.Tensor],
        base_planning_coef: float,
        uncertainty: Optional[torch.Tensor] = None,
        r_self: Optional[torch.Tensor] = None,
        v_pi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Reliability-adaptive blending of policy and planner logits.
        """
        try:
            base = float(base_planning_coef)
        except Exception:
            base = 0.0
        base = max(0.0, min(1.0, base))
        info = {
            "planner_alpha": 0.0,
            "planner_margin": 0.0,
            "planner_js": 0.0,
            "planner_override": 0.0,
        }
        if base <= 0.0 or not isinstance(planner_logits, torch.Tensor):
            return policy_logits, info

        reliab = self._planner_reliability(
            policy_logits=policy_logits,
            planner_logits=planner_logits,
            uncertainty=uncertainty,
            r_self=r_self,
            v_pi=v_pi,
        )
        valid_mask = reliab.get("valid_mask")
        if not isinstance(valid_mask, torch.Tensor) or not bool(valid_mask.any().item()):
            return policy_logits, info

        rel = reliab["reliability"]
        alpha = torch.clamp(rel * base, min=0.0, max=base)
        blended = policy_logits.clone()
        blended[valid_mask] = (
            (1.0 - alpha[valid_mask].unsqueeze(-1)) * policy_logits[valid_mask]
            + alpha[valid_mask].unsqueeze(-1) * planner_logits[valid_mask]
        )

        policy_argmax = torch.argmax(policy_logits.detach(), dim=-1)
        planner_argmax = torch.argmax(planner_logits.detach(), dim=-1)
        override = (
            (policy_argmax != planner_argmax).to(policy_logits.dtype)
            * (alpha > 1.0e-6).to(policy_logits.dtype)
        )

        alpha_valid = alpha[valid_mask]
        margin_valid = reliab["margin"][valid_mask]
        js_valid = reliab["js"][valid_mask]
        override_valid = override[valid_mask]
        info = {
            "planner_alpha": float(alpha_valid.mean().item()) if alpha_valid.numel() else 0.0,
            "planner_margin": float(margin_valid.mean().item()) if margin_valid.numel() else 0.0,
            "planner_js": float(js_valid.mean().item()) if js_valid.numel() else 0.0,
            "planner_override": float(override_valid.mean().item()) if override_valid.numel() else 0.0,
        }
        return blended, info

    def _planner_debug_summary(
        self,
        alpha_values: List[float],
        js_values: List[float],
        margin_values: List[float],
        override_values: List[float],
    ) -> Dict[str, float]:
        def _clean(values: List[float]) -> List[float]:
            out: List[float] = []
            for v in values:
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    out.append(float(v))
            return out

        alpha = np.asarray(_clean(alpha_values), dtype=np.float32)
        js = np.asarray(_clean(js_values), dtype=np.float32)
        margin = np.asarray(_clean(margin_values), dtype=np.float32)
        override = np.asarray(_clean(override_values), dtype=np.float32)

        alpha_mean = float(alpha.mean()) if alpha.size else 0.0
        alpha_p90 = float(np.percentile(alpha, 90.0)) if alpha.size else 0.0
        js_mean = float(js.mean()) if js.size else 0.0
        margin_mean = float(margin.mean()) if margin.size else 0.0
        override_rate = float(override.mean()) if override.size else 0.0
        return {
            "planner_alpha_mean": alpha_mean,
            "planner_alpha_p90": alpha_p90,
            "planner_js_mean": js_mean,
            "planner_margin_mean": margin_mean,
            "planner_override_rate": override_rate,
        }

    def _total_skill_count(self) -> int:
        latent_count = len(self.skill_library) if self.skill_library is not None else 0
        return len(self.skills) + latent_count

    def _select_action_with_skills(
        self,
        G_t: torch.Tensor,
        z_obs: torch.Tensor,
        H_t: torch.Tensor,
        h_w: torch.Tensor,
        traits: torch.Tensor,
        M: torch.Tensor,
        planning_coef: float,
        skill_state: Dict[str, Any],
        env_desc: Optional[torch.Tensor] = None,
        W_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Sample a skill via the high-level policy (optionally mixed with planner skill logits),
        then return the next primitive action from the skill rollout using SkillContext.
        """
        total_skills = self._total_skill_count()
        if total_skills <= 0 or self.agent.high_level_policy is None:
            return None, None, None, skill_state

        env_desc_tensor = env_desc
        if env_desc_tensor is None:
            if self.env_desc_dim is not None:
                env_desc_tensor = torch.zeros((1, self.env_desc_dim), device=self.device)
            elif self.env_descriptors is not None:
                env_desc_tensor = torch.zeros_like(self.env_descriptors[0:1]).to(self.device)
            else:
                env_desc_tensor = torch.zeros((1, 1), device=self.device)

        world_latent = W_t
        if world_latent is None:
            world_latent = h_w.squeeze(0) if h_w is not None else torch.zeros((1, getattr(self.agent, "w_dim", 1)), device=self.device)

        extra = skill_state.get("obs_history", [])[-1] if skill_state.get("obs_history") else None
        # continue existing active skill if any
        active_skill = skill_state.get("active_skill")
        if active_skill is not None:
            local_step = int(skill_state.get("step_in_skill", 0))
            ctx = SkillContext(
                w=world_latent,
                h=H_t,
                traits=traits,
                env_desc=env_desc_tensor,
                step_in_skill=local_step,
                extra=extra,
            )
            action_id = active_skill.step(ctx)
            skill_state["step_in_skill"] = local_step + 1
            if active_skill.is_complete():
                skill_state.pop("active_skill", None)
                skill_state["step_in_skill"] = 0
            logprob = skill_state.get("logprob")
            entropy = skill_state.get("entropy")
            return (
                torch.tensor([action_id], device=self.device, dtype=torch.long),
                logprob if logprob is not None else torch.tensor(0.0, device=self.device),
                entropy if entropy is not None else torch.tensor(0.0, device=self.device),
                skill_state,
            )

        logits_skill = self.agent.high_level_policy(G_t)
        if planning_coef > 0.0 and self.planner_mode == "skills":
            planner_logits = self._planner_skill_logits(
                z_obs=z_obs,
                H_t=H_t,
                h_w=h_w,
                traits=traits,
                M=M,
            )
            logits_skill = (1.0 - planning_coef) * logits_skill + planning_coef * planner_logits

        dist_skill = Categorical(logits=logits_skill)
        skill_id = dist_skill.sample()
        idx = int(skill_id.item())
        self.skill_usage_counts[idx] = self.skill_usage_counts.get(idx, 0) + 1

        if idx < len(self.skills):
            skill = self.skills[idx]
        else:
            latent_idx = idx - len(self.skills)
            if self.skill_library is None or latent_idx >= len(self.skill_library):
                return None, None, None, skill_state
            skill = self.skill_library.get_skill(latent_idx)

        skill.reset()
        ctx = SkillContext(
            w=world_latent,
            h=H_t,
            traits=traits,
            env_desc=env_desc_tensor,
            step_in_skill=0,
            extra=extra,
        )
        action_id = skill.step(ctx)

        if skill.is_complete():
            skill_state["active_skill"] = None
            skill_state["step_in_skill"] = 0
        else:
            skill_state["active_skill"] = skill
            skill_state["step_in_skill"] = skill.step_in_skill
        skill_state["logprob"] = dist_skill.log_prob(skill_id)
        skill_state["entropy"] = dist_skill.entropy()

        return (
            torch.tensor([action_id], device=self.device, dtype=torch.long),
            skill_state["logprob"],
            skill_state["entropy"],
            skill_state,
        )

    # =========================
    #  Latent skill distillation
    # =========================

    def collect_skill_demonstrations(
        self,
        skill: Skill,
        num_episodes: int,
        max_steps_per_episode: Optional[int] = None,
    ) -> List[SkillDemoTransition]:
        """
        Roll out a fixed hand-crafted skill and record (context -> action) pairs.
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = self.latent_skill_training.max_steps_per_episode
        transitions: List[SkillDemoTransition] = []
        episodes_run = 0

        for _ in range(num_episodes):
            obs = self.env.reset()
            patch = obs.get("patch")
            energy = float(obs.get("energy", 0.0))
            scenario_id = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0)))
            env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))
            if patch is None:
                continue

            h_w = torch.zeros(
                1,
                1,
                self.agent.world_model.gru.hidden_size,
                device=self.device,
            )

            skill.reset()
            done = False
            t = 0
            while not done and t < max_steps_per_episode:
                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)
                    if env_desc_t is None:
                        env_dim = getattr(self.agent, "env_desc_dim", 1) or 1
                        env_desc_t = torch.zeros((1, env_dim), device=self.device)
                    traits = self._mixed_traits().to(self.device)

                    ctx = SkillContext(
                        w=h_w.squeeze(0),
                        h=H_t,
                        traits=traits,
                        env_desc=env_desc_t,
                        step_in_skill=skill.step_in_skill,
                        extra={"patch": patch, "energy": energy},
                    )
                    ctx_vec = flatten_skill_context(ctx, max_horizon=getattr(skill, "horizon", None))
                action = int(skill.step(ctx))
                transitions.append(SkillDemoTransition(context=ctx_vec.detach(), action=action))

                next_obs, _, done, info = self.env.step(action)
                patch = next_obs.get("patch", patch)
                energy = float(next_obs.get("energy", energy))
                scenario_id = int(next_obs.get("scenario_id", scenario_id))
                env_id = int(next_obs.get("env_id", env_id))
                t += 1

                with torch.no_grad():
                    patch_next_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_next = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    a_t = torch.tensor([action], dtype=torch.long, device=self.device)
                    env_t_next = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_next = self._env_desc_from_ids(env_t_next)
                    if env_desc_next is None:
                        env_desc_next = env_desc_t
                    scenario_next_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                    text_next = self._text_tokens_from_ids(env_t_next, scenario_next_t)
                    z_next = self.agent.perception(
                        patch_next_t, H_next, scenario_next_t, env_desc_next, text_tokens=text_next
                    )
                    _, h_w, _, _ = self.agent.world_model.forward_step(
                        z_next,
                        H_next,
                        a_t,
                        h_w,
                    )

                if skill.is_complete():
                    skill.reset()

            episodes_run += 1

        if transitions:
            print(
                f"[LatentSkills] Collected {len(transitions)} transitions "
                f"from skill={skill.name} over {episodes_run} episodes."
            )
        return transitions

    def train_latent_skills_supervised(self):
        """
        Pair each hand-crafted skill with a latent skill, gather demonstrations,
        and train the latent skill via cross-entropy on primitive actions.
        """
        if self.skill_library is None or self.n_latent_skills <= 0:
            print("[LatentSkills] No latent skills to train; skipping.")
            return
        if not self.skills:
            print("[LatentSkills] No hand-crafted skills available for distillation; skipping.")
            return

        cfg = self.latent_skill_training
        pairs = list(zip(self.skills, list(self.skill_library.skills)))
        if not pairs:
            print("[LatentSkills] No skill pairs found for distillation.")
            return

        for idx, (hand_skill, latent_skill) in enumerate(pairs):
            latent_skill.train()
            transitions: List[SkillDemoTransition] = []
            episodes = 0
            while len(transitions) < cfg.demos_per_skill:
                to_run = max(1, math.ceil((cfg.demos_per_skill - len(transitions)) / max(1, cfg.max_steps_per_episode)))
                new_transitions = self.collect_skill_demonstrations(
                    hand_skill,
                    num_episodes=to_run,
                    max_steps_per_episode=cfg.max_steps_per_episode,
                )
                transitions.extend(new_transitions)
                episodes += to_run
                if not new_transitions:
                    break

            if not transitions:
                print(f"[LatentSkills] No transitions gathered for skill {hand_skill.name}; skipping.")
                continue

            if cfg.shuffle_buffer_size and len(transitions) > cfg.shuffle_buffer_size:
                perm = torch.randperm(len(transitions))[: cfg.shuffle_buffer_size].tolist()
                transitions = [transitions[i] for i in perm]

            contexts = torch.stack([t.context for t in transitions], dim=0).to(self.device)
            actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
            n_samples = contexts.shape[0]
            print(
                f"[LatentSkills] Training latent_skill_{idx} to imitate {hand_skill.name} "
                f"on {n_samples} samples (episodes ~{episodes})."
            )

            optimizer = torch.optim.Adam(latent_skill.parameters(), lr=cfg.lr)

            for epoch in range(cfg.epochs):
                perm = torch.randperm(n_samples, device=self.device)
                total_loss = 0.0
                batches = 0
                for start in range(0, n_samples, cfg.batch_size):
                    batch_idx = perm[start : start + cfg.batch_size]
                    batch_ctx = contexts[batch_idx]
                    batch_act = actions[batch_idx]
                    logits = latent_skill.forward_logits(batch_ctx)
                    loss = F.cross_entropy(logits, batch_act)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    batches += 1
                mean_loss = total_loss / max(1, batches)
                print(
                    f"[LatentSkills] skill={hand_skill.name} epoch {epoch + 1}/{cfg.epochs} "
                    f"loss={mean_loss:.4f}"
                )
            latent_skill.eval()

    # =========================
    #  On-policy experience for RL
    # =========================

    def collect_onpolicy_experience(
        self,
        n_steps: int,
        use_self: bool = False,
        curiosity_beta: float = 0.1,
        beta_conflict: float = 0.1,
        beta_uncertainty: float = 0.1,
        planning_coef: float = 0.0,
    ):
        """
        Собирает опыт текущей policy для RL.
        В буфер кладём env-reward (trait-based), без любопытства.
        Для A2C используем reward_total = env + curiosity.

        Новое:
          - self-модель (delta_self),
          - meta-curiosity (адаптивный curiosity_beta),
          - multi-step planner (через _get_planner_logits):
              * планировщик строит logits_planner(a),
              * смешиваем с logits политики:
                    logits = (1 - planning_coef) * logits + planning_coef * planner_logits
        """

        # коэффициенты для гейтинга влияния self-модели (усилены)
        alpha_conf = 0.5
        alpha_unc = 0.5
        max_delta_amp = 0.2

        self.agent.perception.eval()
        self.agent.world_model.train()
        self.agent.self_model.eval()
        self.agent.workspace.train()
        self.agent.policy.train()
        self.agent.value_model.train()

        obs = self.env.reset(split="train")
        patch_np = obs["patch"]
        energy = obs["energy"]
        scenario_id = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0)))
        env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

        h_w = torch.zeros(
            1,
            1,
            self.agent.world_model.gru.hidden_size,
            device=self.device,
        )
        h_s = torch.zeros(
            1,
            1,
            self.agent.self_model.gru.hidden_size,
            device=self.device,
        )

        last_action = torch.zeros(1, dtype=torch.long, device=self.device)
        last_reward = 0.0

        # Preallocate non-differentiable buffers on device to avoid per-step tensor churn.
        # NOTE: Do NOT store differentiable tensors (logprobs/values/entropy/etc.) into a
        # preallocated tensor, because that breaks the autograd graph and makes A2C a no-op.
        actions_buf = torch.empty(n_steps, dtype=torch.long, device=self.device)
        rewards_buf = torch.empty(n_steps, dtype=torch.float32, device=self.device)
        dones_buf = torch.empty(n_steps, dtype=torch.float32, device=self.device)

        # Keep differentiable signals as lists of tensors (preserves autograd graphs).
        values_list: List[torch.Tensor] = []
        logprobs_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []
        conflicts_list: List[torch.Tensor] = []
        uncertainties_list: List[torch.Tensor] = []
        invalid_action_mass_list: List[torch.Tensor] = []
        mask_pred_logits_list: List[torch.Tensor] = []
        mask_targets_list: List[torch.Tensor] = []
        online_bc_logits_list: List[torch.Tensor] = []
        online_bc_targets_list: List[torch.Tensor] = []
        online_bc_enabled = float(getattr(self, "repo_online_bc_coef", 0.0) or 0.0) > 0.0

        step_idx = 0
        planner_used_steps = 0
        planner_alpha_vals: List[float] = []
        planner_js_vals: List[float] = []
        planner_margin_vals: List[float] = []
        planner_override_vals: List[float] = []
        patch_shape = patch_np.shape

        skill_state = {
            "logprob": None,
            "entropy": None,
            "obs_history": [],
            "active_skill": None,
            "step_in_skill": 0,
        }

        # Reusable tensors for current/next observations (device-resident)
        patch_t = torch.empty((1, *patch_shape), dtype=torch.long, device=self.device)
        patch_t.copy_(torch.from_numpy(patch_np))
        H_t = torch.empty((1, 1), dtype=torch.float32, device=self.device)
        H_t.fill_(float(energy))
        scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
        # reusable "next" buffers to avoid per-step allocations
        H_next = torch.empty_like(H_t)
        scenario_next_t = torch.empty_like(scenario_t)
        env_next_t = torch.empty_like(env_t)

        while step_idx < n_steps:
            skill_state["obs_history"].append({"patch": patch_np.copy(), "energy": float(energy)})
            H_t.fill_(float(energy))
            scenario_t.fill_(scenario_id)
            env_t.fill_(env_id)
            env_desc_t = self._env_desc_from_ids(env_t)

            # Use clones for model inputs to avoid autograd saving mutated tensors
            patch_in = patch_t.clone()
            H_in = H_t.clone()
            scenario_in = scenario_t.clone()
            env_in = env_t.clone()
            env_desc_in = self._env_desc_from_ids(env_in)
            text_in = self._text_tokens_from_ids(env_in, scenario_in)

            # perception
            z_obs = self.agent.perception(
                patch_in, H_in, scenario_in, env_desc_in, text_tokens=text_in
            )  # (1,obs_dim)
            W_t = h_w.squeeze(0)  # (1,w_dim)

            traits = self._mixed_traits()
            M = self.agent.memory
            invalid_action_mass = torch.zeros(1, device=self.device, dtype=torch.float32)

            # ---- SelfModel ----
            if use_self:
                with torch.no_grad():
                    r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                    a_emb = self.agent.self_model.act_emb(last_action)  # (1,8)
                    env_emb = self.agent.self_model.env_desc_to_emb(env_desc_in)

                    W_in = W_t.unsqueeze(1)  # (1,1,w_dim)
                    H_in_step = H_in.unsqueeze(1)  # (1,1,h_dim)
                    r_in = r_t.unsqueeze(1)  # (1,1,1)
                    M_b = M.unsqueeze(1)     # (1,1,mem_dim)

                    x_s = torch.cat(
                        [W_in, H_in_step, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)],
                        dim=-1,
                    )
                    out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                    S_t = out_s.squeeze(1)  # (1,s_dim)

                    surv_raw = torch.sigmoid(self.agent.self_model.head_survival(S_t))  # (1,1)
                    surv_t = torch.sigmoid(
                        self.agent.self_model.head_survival_calib(surv_raw)
                    )  # (1,1)
                    food_t = self.agent.self_model.head_food(S_t)      # (1,1)
                    dmg_t = self.agent.self_model.head_damage(S_t)     # (1,1)
                    move_t = self.agent.self_model.head_move(S_t)      # (1,1)

                    w = traits_to_preference_weights(traits)  # (1,4)
                    w_survive, w_food, w_danger, w_move = w[0]

                    R_unscaled = (
                        w_survive * surv_t
                        + w_food * food_t
                        + w_danger * dmg_t
                        + w_move * move_t
                    ).view(1, 1)

                    R_self = self.agent.self_model.head_return_calib(R_unscaled)
                    U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
            else:
                S_t = torch.zeros(
                    1,
                    self.agent.self_model.gru.hidden_size,
                    device=self.device,
                )
                R_self = torch.zeros(1, 1, device=self.device)
                U_t = torch.zeros(1, 1, device=self.device)

            # critic
            V_pi = self.agent.value_model(W_t, H_in, traits, M)  # (1,1)

            if use_self:
                conf_t = torch.abs(R_self - V_pi.detach())  # (1,1)
                gate = torch.exp(
                    -alpha_conf * (conf_t ** 2) - alpha_unc * (U_t ** 2)
                )  # (1,1)
                delta_raw = torch.tanh(R_self - V_pi.detach())  # (1,1)
                delta_self = delta_raw * gate  # (1,1)
                delta_self = torch.clamp(delta_self, -max_delta_amp, max_delta_amp)
            else:
                conf_t = torch.zeros_like(R_self)
                delta_self = torch.zeros_like(R_self)

            # Workspace
            G_t = self.agent.workspace(
                W_t,
                S_t,
                H_t,
                V_pi,
                delta_self,
                U_t,
                traits,
                M,
            )
            logits_for_online_bc: Optional[torch.Tensor] = None

            if self.use_skills and self.agent.high_level_policy is not None and self._total_skill_count() > 0:
                action, logprob, entropy, skill_state = self._select_action_with_skills(
                    G_t=G_t,
                    z_obs=z_obs,
                    H_t=H_t,
                    h_w=h_w,
                    traits=traits,
                    M=M,
                    planning_coef=planning_coef,
                    skill_state=skill_state,
                    env_desc=env_desc_in,
                    W_t=W_t,
                )
                # fallback to primitive policy if skill selection failed
                if action is None:
                    logits_raw, mask_logits_pred = self._policy_forward_with_mask(G_t)
                    logits_for_online_bc = logits_raw
                    mask = self._get_action_mask_for_logits(logits_raw)
                    has_invalid = bool(mask is not None and bool(torch.any(~mask).item()))
                    if mask is not None and mask_logits_pred is not None:
                        mask_pred_logits_list.append(mask_logits_pred)
                        mask_targets_list.append(mask.to(mask_logits_pred.dtype))
                    if has_invalid:
                        probs = torch.softmax(logits_raw, dim=-1)
                        invalid_action_mass = (probs * (~mask).to(probs.dtype)).sum(dim=-1)
                    else:
                        invalid_action_mass = torch.zeros(
                            logits_raw.shape[0], device=logits_raw.device, dtype=torch.float32
                        )
                    apply_mask = True
                    dropout_p = float(getattr(self, "action_mask_dropout_prob", 0.0) or 0.0)
                    if dropout_p > 0.0 and has_invalid:
                        if torch.rand((), device=logits_raw.device).item() < dropout_p:
                            apply_mask = False
                    logits = self._compose_policy_logits_with_masks(
                        logits_raw,
                        mask,
                        mask_logits_pred,
                        apply_hard_mask=bool(apply_mask),
                    )
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    entropy = dist.entropy()
            else:
                logits_raw, mask_logits_pred = self._policy_forward_with_mask(G_t)  # (1, n_actions)

                # ---- planner (????????????? _get_planner_logits) ----
                if planning_coef > 0.0:
                    with torch.no_grad():
                        planner_logits = self._get_planner_logits(
                            z_obs=z_obs,
                            H_t=H_t,
                            h_w=h_w,
                            traits=traits,
                            M=M,
                        )
                    logits_raw, planner_debug = self._blend_with_planner(
                        policy_logits=logits_raw,
                        planner_logits=planner_logits,
                        base_planning_coef=float(planning_coef),
                        uncertainty=U_t if use_self else None,
                        r_self=R_self if use_self else None,
                        v_pi=V_pi if use_self else None,
                    )
                    planner_alpha_vals.append(float(planner_debug.get("planner_alpha", 0.0)))
                    planner_js_vals.append(float(planner_debug.get("planner_js", 0.0)))
                    planner_margin_vals.append(float(planner_debug.get("planner_margin", 0.0)))
                    planner_override_vals.append(float(planner_debug.get("planner_override", 0.0)))
                    if float(planner_debug.get("planner_alpha", 0.0)) > 1.0e-6:
                        planner_used_steps += 1
                logits_for_online_bc = logits_raw

                mask = self._get_action_mask_for_logits(logits_raw)
                has_invalid = bool(mask is not None and bool(torch.any(~mask).item()))
                if mask is not None and mask_logits_pred is not None:
                    mask_pred_logits_list.append(mask_logits_pred)
                    mask_targets_list.append(mask.to(mask_logits_pred.dtype))
                if has_invalid:
                    probs = torch.softmax(logits_raw, dim=-1)
                    invalid_action_mass = (probs * (~mask).to(probs.dtype)).sum(dim=-1)
                else:
                    invalid_action_mass = torch.zeros(
                        logits_raw.shape[0], device=logits_raw.device, dtype=torch.float32
                    )
                apply_mask = True
                dropout_p = float(getattr(self, "action_mask_dropout_prob", 0.0) or 0.0)
                if dropout_p > 0.0 and has_invalid:
                    if torch.rand((), device=logits_raw.device).item() < dropout_p:
                        apply_mask = False
                logits = self._compose_policy_logits_with_masks(
                    logits_raw,
                    mask,
                    mask_logits_pred,
                    apply_hard_mask=bool(apply_mask),
                )
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                entropy = dist.entropy()

            if online_bc_enabled and logits_for_online_bc is not None:
                expert_action = self._get_repo_expert_action()
                if (
                    expert_action is not None
                    and int(expert_action) >= 0
                    and int(expert_action) < int(logits_for_online_bc.shape[-1])
                ):
                    online_bc_logits_list.append(logits_for_online_bc)
                    online_bc_targets_list.append(
                        torch.tensor([int(expert_action)], dtype=torch.long, device=self.device)
                    )

            # world model step for curiosity
            _, h_w_new, z_hat, H_hat = self.agent.world_model.forward_step(
                z_obs, H_in, action, h_w
            )

            next_obs, _, done, info = self.env.step(action.item())
            reward_env = self.compute_preference_reward(info)
            death_flag = float(info.get("death_flag", 0.0))

            patch_next_np = next_obs["patch"]
            energy_next = next_obs["energy"]
            scenario_next_id = int(
                next_obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id))
            )
            env_next_id = int(next_obs.get("env_id", env_id))

            patch_t.copy_(torch.from_numpy(patch_next_np))
            H_next.fill_(float(energy_next))
            scenario_next_t.fill_(scenario_next_id)
            env_next_t.fill_(env_next_id)
            env_next_desc = self._env_desc_from_ids(env_next_t)

            # clones for model inputs on next state
            patch_next_in = patch_t.clone()
            H_next_in = H_next.clone()
            scenario_next_in = scenario_next_t.clone()
            env_next_in = env_next_t.clone()
            env_next_desc_in = self._env_desc_from_ids(env_next_in)
            text_next_in = self._text_tokens_from_ids(env_next_in, scenario_next_in)

            z_next = self.agent.perception(
                patch_next_in, H_next_in, scenario_next_in, env_next_desc_in, text_tokens=text_next_in
            )

            err_cur = self.agent.world_model.curiosity_error(
                z_next, H_next_in, z_hat, H_hat
            )
            env_family = info.get("env_family") if isinstance(info, dict) else None
            if curiosity_beta > 0.0 and str(env_family or "") != "repo-basic":
                cur_r = (curiosity_beta * err_cur.detach()).item()
            else:
                cur_r = 0.0

            reward_total = reward_env + cur_r

            # Log for RL.
            actions_buf[step_idx] = action
            rewards_buf[step_idx] = reward_total
            dones_buf[step_idx] = float(done)
            values_list.append(V_pi.view(-1))
            logprobs_list.append(logprob.view(-1))
            entropies_list.append(entropy.view(-1))
            conflicts_list.append(conf_t.view(-1))
            uncertainties_list.append(U_t.view(-1))
            invalid_action_mass_list.append(invalid_action_mass.view(-1))

            got_food = 1.0 if info.get("got_food", False) else 0.0
            took_damage = 1.0 if info.get("took_damage", False) else 0.0
            moved = 1.0 if info.get("moved", False) else 0.0
            alive = 1.0 if info.get("alive", True) else 0.0

            tr = Transition(
                obs_patch=patch_np,
                energy=energy,
                action=action.item(),
                reward=reward_env,
                done=done,
                next_obs_patch=next_obs["patch"],
                next_energy=next_obs["energy"],
                death_flag=death_flag,
                got_food=got_food,
                took_damage=took_damage,
                moved=moved,
                alive=alive,
                scenario_id=scenario_id,
                env_id=env_id,
                regime_name=self.current_regime_name or "stage4",
            )
            self.buffer.push(tr)

            # обновляем состояние
            patch_np = patch_next_np
            energy = energy_next
            scenario_id = scenario_next_id
            env_id = env_next_id
            # Detach recurrent state so we don't backprop through time across the whole rollout.
            # A2C here is not meant to do full BPTT through the world-model GRU.
            h_w = h_w_new.detach()

            last_reward = reward_env
            last_action = action.detach()
            step_idx += 1

            if done and step_idx < n_steps:
                obs = self.env.reset(split="train")
                patch_np = obs["patch"]
                patch_t.copy_(torch.from_numpy(patch_np))
                energy = obs["energy"]
                scenario_id = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0)))
                env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))
                h_w = torch.zeros_like(h_w)
                h_s = torch.zeros_like(h_s)
                last_reward = 0.0
                last_action = torch.zeros(1, dtype=torch.long, device=self.device)

        # Slice to actual filled length (may be < n_steps if loop breaks early, but here equals step_idx)
        actions_t = actions_buf[:step_idx]
        rewards_total_t = rewards_buf[:step_idx]
        dones_t = dones_buf[:step_idx]
        if step_idx <= 0:
            raise RuntimeError("collect_onpolicy_experience collected 0 steps; cannot train A2C.")
        values_t = torch.cat(values_list, dim=0)
        logprobs_t = torch.cat(logprobs_list, dim=0)
        entropies_t = torch.cat(entropies_list, dim=0)
        conflicts_t = torch.cat(conflicts_list, dim=0)
        uncertainties_t = torch.cat(uncertainties_list, dim=0)
        invalid_action_mass_t = torch.cat(invalid_action_mass_list, dim=0)
        mask_pred_logits_t: Optional[torch.Tensor] = (
            torch.cat(mask_pred_logits_list, dim=0) if mask_pred_logits_list else None
        )
        mask_targets_t: Optional[torch.Tensor] = (
            torch.cat(mask_targets_list, dim=0) if mask_targets_list else None
        )
        online_bc_logits_t: Optional[torch.Tensor] = (
            torch.cat(online_bc_logits_list, dim=0) if online_bc_logits_list else None
        )
        online_bc_targets_t: Optional[torch.Tensor] = (
            torch.cat(online_bc_targets_list, dim=0) if online_bc_targets_list else None
        )

        # logging/diagnostics (planner usage)
        self.last_planner_usage_steps = int(planner_used_steps)
        self.last_planner_total_steps = int(step_idx)
        self.last_planner_usage_frac = float(planner_used_steps) / float(max(1, step_idx))
        self.last_planner_debug = self._planner_debug_summary(
            alpha_values=planner_alpha_vals,
            js_values=planner_js_vals,
            margin_values=planner_margin_vals,
            override_values=planner_override_vals,
        )

        return (
            actions_t,
            rewards_total_t,
            dones_t,
            values_t,
            logprobs_t,
            entropies_t,
            conflicts_t,
            uncertainties_t,
            invalid_action_mass_t,
            mask_pred_logits_t,
            mask_targets_t,
            online_bc_logits_t,
            online_bc_targets_t,
        )

    # =========================
    #  A2C training
    # =========================

    def update_memory_from_buffer(
        self,
        batch_size: int = 32,
        seq_len: int = 32,
        memory_alpha: float = 0.1,
    ):
        """
        Обновление долговременной памяти M по свежим кускам из буфера.
        Использует SelfModel → S_last → to_memory(S_mean).
        """
        if len(self.buffer) < seq_len:
            return

        (
            obs_seq,
            H_seq,
            a_seq,
            r_seq,
            d_seq,
            death_seq,
            scenario_seq,
            env_seq,
        ) = self._sample_sequences_from_buffer(
            self.buffer, batch_size, seq_len, with_events=False, current_regime=self.current_regime_name
        )

        B, T, p, _ = obs_seq.shape

        patch = torch.from_numpy(obs_seq).long().to(self.device)
        H = torch.from_numpy(H_seq).float().to(self.device)
        a = torch.from_numpy(a_seq).long().to(self.device)
        r = torch.from_numpy(r_seq).float().to(self.device)
        scenario = torch.from_numpy(scenario_seq).long().to(self.device)
        env_ids = torch.from_numpy(env_seq).long().to(self.device)
        env_desc_seq = self._env_desc_from_ids(env_ids)

        with torch.no_grad():
            patch_flat = patch.view(B * T, p, p)
            H_flat = H.view(B * T, 1)
            scenario_flat = scenario.view(B * T)
            env_flat = env_ids.view(B * T)
            env_desc_flat = env_desc_seq.reshape(B * T, -1) if env_desc_seq is not None else None

            text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
            z_flat = self.agent.perception(
                patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
            )
            z_seq = z_flat.reshape(B, T, -1)

            a_emb = self.agent.world_model.act_emb(a)
            x_w = torch.cat([z_seq, H, a_emb], dim=-1)
            h0 = torch.zeros(
                1,
                B,
                self.agent.world_model.gru.hidden_size,
                device=self.device,
            )
            out_w, _ = self.agent.world_model.gru(x_w, h0)
            W_seq = out_w

            M = self.agent.memory
            (
                S_seq,
                S_last,
                surv_pred,
                food_pred,
                dmg_pred,
                move_pred,
                unc_pred,
                surv_raw_pred,
            ) = self.agent.self_model.forward_seq(
                W_seq,
                H,
                a,
                r,
                M=M,
                env_desc=env_desc_seq,
            )
            S_mean = S_last.mean(dim=0, keepdim=True)  # (1,s_dim)
            M_target = self.agent.self_model.to_memory(S_mean)  # (1,mem_dim)
            # обновляем эпизодическую память через слоты
            self.agent.write_memory(
                M_target, alpha_slot=memory_alpha, alpha_summary=memory_alpha
            )

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Generalized Advantage Estimation (GAE-λ) для улучшенного кредит-ассайнмента.
        rewards: (T,)
        values:  (T,)  — V(s_t)
        dones:   (T,)  — 1.0 если эпизод закончился на этом шаге, 0.0 иначе

        Возвращает:
          advantages: (T,)
          returns:    (T,)  где returns = advantages + values
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        # расширяем values, чтобы иметь V_{T} для bootstrap'а
        values_ext = torch.cat(
            [values, torch.zeros(1, device=self.device, dtype=values.dtype)], dim=0
        )

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # 0 если эпизод закончился, 1 иначе
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def _fast_param_l2_penalty(self, coef: float) -> torch.Tensor:
        if coef <= 0.0:
            return torch.tensor(0.0, device=self.device)
        penalty: Optional[torch.Tensor] = None
        for p_init, p in zip(self.fast_params_initial, self.agent.get_fast_params()):
            if p_init.shape != p.shape:
                continue
            diff = p - p_init.to(p.device)
            term = (diff * diff).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            return torch.tensor(0.0, device=self.device)
        return coef * penalty

    def _fast_param_distance_from_init(self) -> float:
        total = 0.0
        for p_init, p in zip(self.fast_params_initial, self.agent.get_fast_params()):
            if p_init.shape != p.shape:
                continue
            diff = p.detach() - p_init.to(p.device)
            total += float(torch.norm(diff).item() ** 2)
        return float(total ** 0.5)

    def _run_a2c_update(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        logprobs: torch.Tensor,
        entropies: torch.Tensor,
        conflicts: Optional[torch.Tensor],
        uncertainties: Optional[torch.Tensor],
        gamma: float,
        entropy_coef: float,
        beta_conflict: float,
        beta_uncertainty: float,
        invalid_action_mass: Optional[torch.Tensor] = None,
        mask_pred_logits: Optional[torch.Tensor] = None,
        mask_targets: Optional[torch.Tensor] = None,
        online_bc_logits: Optional[torch.Tensor] = None,
        online_bc_targets: Optional[torch.Tensor] = None,
        regularization_coef: float = 0.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """
        Shared A2C update used by standard training and lifelong fine-tuning.
        Expects flattened tensors (T,) on the correct device.
        """
        conflicts_t = conflicts if conflicts is not None else torch.zeros_like(rewards)
        uncertainties_t = uncertainties if uncertainties is not None else torch.zeros_like(rewards)

        # Compute GAE targets from a detached critic to avoid backprop-through-time
        # across the whole rollout (stabilizes A2C and keeps graphs small).
        gae_lambda_base = float(getattr(self, "gae_lambda_base", 0.95) or 0.95)
        gae_lambda_eff = self.get_adaptive_gae_lambda(gae_lambda_base)
        advantages, returns_t = self._compute_gae(
            rewards=rewards,
            values=values.detach(),
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda_eff,
        )

        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        adv_eps = float(getattr(self.safety, "std_eps", 1e-6))
        if not torch.isfinite(adv_std):
            adv_std = torch.tensor(adv_eps, device=advantages.device, dtype=advantages.dtype)
        adv_std = torch.clamp(adv_std, min=adv_eps)
        advantages_norm = (advantages - adv_mean) / adv_std

        policy_loss = -(advantages_norm.detach() * logprobs).mean()
        value_loss = (returns_t - values).pow(2).mean()
        entropy = entropies.mean()

        aux_penalty = beta_conflict * conflicts_t.mean() + beta_uncertainty * uncertainties_t.mean()
        entropy_coef_eff = self.get_adaptive_entropy_coef(entropy_coef)

        loss = policy_loss + 0.5 * value_loss - entropy_coef_eff * entropy + aux_penalty
        invalid_coef = float(getattr(self, "action_mask_internalization_coef", 0.0) or 0.0)
        if invalid_action_mass is not None and invalid_coef > 0.0:
            # Encourage the policy to internalize the action mask even when actions are sampled
            # from the masked distribution. Using a KL-style barrier is stronger than a linear
            # penalty on invalid mass when valid_mass is small:
            #   KL(masked||raw) = -log(sum_a p_raw(a) * mask(a)) = -log(valid_mass)
            valid_mass = (1.0 - invalid_action_mass).clamp(min=1.0e-6, max=1.0)
            loss = loss + invalid_coef * (-torch.log(valid_mass)).mean()
        mask_pred_coef = float(getattr(self, "action_mask_prediction_coef", 0.0) or 0.0)
        mask_pred_loss = torch.tensor(0.0, device=self.device, dtype=loss.dtype)
        mask_pred_f1: Optional[float] = None
        mask_pred_auc: Optional[float] = None
        if (
            mask_pred_logits is not None
            and mask_targets is not None
            and mask_pred_coef > 0.0
            and mask_pred_logits.numel() > 0
            and mask_targets.numel() > 0
        ):
            targets = mask_targets.to(mask_pred_logits.dtype)
            mask_pred_loss = F.binary_cross_entropy_with_logits(mask_pred_logits, targets)
            loss = loss + mask_pred_coef * mask_pred_loss

            with torch.no_grad():
                probs = torch.sigmoid(mask_pred_logits)
                preds = probs >= 0.5
                trues = targets >= 0.5
                tp = float((preds & trues).sum().item())
                fp = float((preds & (~trues)).sum().item())
                fn = float(((~preds) & trues).sum().item())
                precision = tp / max(1.0, tp + fp)
                recall = tp / max(1.0, tp + fn)
                if precision + recall > 0.0:
                    mask_pred_f1 = float((2.0 * precision * recall) / (precision + recall))
                else:
                    mask_pred_f1 = 0.0

                scores_np = probs.detach().reshape(-1).cpu().numpy()
                labels_np = trues.detach().reshape(-1).to(torch.int64).cpu().numpy()
                n_pos = int(labels_np.sum())
                n_total = int(labels_np.size)
                n_neg = int(n_total - n_pos)
                if n_pos > 0 and n_neg > 0:
                    order = np.argsort(scores_np, kind="mergesort")
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(n_total) + 1
                    sum_pos = float(ranks[labels_np == 1].sum())
                    mask_pred_auc = float(
                        (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(max(1, n_pos * n_neg))
                    )
                    try:
                        prev_auc_ema = float(getattr(self, "mask_pred_auc_ema", float("nan")))
                    except Exception:
                        prev_auc_ema = float("nan")
                    try:
                        decay = float(getattr(self, "mask_pred_auc_ema_decay", 0.95) or 0.95)
                    except Exception:
                        decay = 0.95
                    decay = max(0.0, min(0.999, decay))
                    if math.isfinite(prev_auc_ema):
                        self.mask_pred_auc_ema = decay * prev_auc_ema + (1.0 - decay) * mask_pred_auc
                    else:
                        self.mask_pred_auc_ema = mask_pred_auc
        online_bc_coef = float(getattr(self, "repo_online_bc_coef", 0.0) or 0.0)
        online_bc_loss = torch.tensor(0.0, device=self.device, dtype=loss.dtype)
        online_bc_samples = 0.0
        if (
            online_bc_logits is not None
            and online_bc_targets is not None
            and online_bc_coef > 0.0
            and online_bc_logits.numel() > 0
            and online_bc_targets.numel() > 0
        ):
            targets_long = online_bc_targets.to(dtype=torch.long).view(-1)
            online_bc_samples = float(targets_long.numel())
            online_bc_loss = F.cross_entropy(online_bc_logits, targets_long)
            loss = loss + online_bc_coef * online_bc_loss
        if regularization_coef > 0.0:
            loss = loss + self._fast_param_l2_penalty(regularization_coef)

        opt = optimizer or self.agent.optim_policy
        opt.zero_grad()
        loss.backward()
        params_to_clip = [p for group in opt.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(params_to_clip, 1.0)
        opt.step()


        return {
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "entropy_coef_eff": float(entropy_coef_eff),
            "mean_conflict": float(conflicts_t.mean().item()),
            "mean_uncertainty": float(uncertainties_t.mean().item()),
            "gae_lambda_eff": float(gae_lambda_eff),
            "mean_invalid_action_mass": float(invalid_action_mass.mean().item())
            if invalid_action_mass is not None
            else 0.0,
            "invalid_action_coef": float(invalid_coef),
            "mask_pred_coef": float(mask_pred_coef),
            "mask_pred_loss": float(mask_pred_loss.detach().item()),
            "mask_pred_f1": float(mask_pred_f1) if mask_pred_f1 is not None else float("nan"),
            "mask_pred_auc": float(mask_pred_auc) if mask_pred_auc is not None else float("nan"),
            "online_bc_coef": float(online_bc_coef),
            "online_bc_loss": float(online_bc_loss.detach().item()),
            "online_bc_samples": float(online_bc_samples),
        }

    def train_repo_policy_bc(
        self,
        n_episodes: int = 64,
        max_steps: int = 24,
        planning_coef: float = 0.0,
        regime_name: str = "repo_bc_pretrain",
    ) -> Dict[str, Any]:
        """
        Behavior-cloning pretrain on RepoToolEnv via scripted expert actions.
        """
        episodes_req = int(max(0, n_episodes))
        if episodes_req <= 0:
            return {
                "used": False,
                "episodes_requested": 0.0,
                "episodes_used": 0.0,
                "steps": 0.0,
                "success_rate": 0.0,
                "mean_loss": 0.0,
                "mean_action_loss": 0.0,
                "mean_mask_pred_loss": 0.0,
            }

        self.current_regime_name = str(regime_name or "repo_bc_pretrain")
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.eval()
        self.agent.workspace.train()
        self.agent.policy.train()
        self.agent.value_model.train()

        opt = self.agent.optim_policy
        planning_coef_eff = max(0.0, float(planning_coef))
        mask_pred_coef = float(getattr(self, "action_mask_prediction_coef", 0.0) or 0.0)

        episodes_used = 0
        steps_total = 0
        solved = 0
        loss_sum = 0.0
        action_loss_sum = 0.0
        mask_loss_sum = 0.0

        for _ in range(episodes_req):
            obs = self.env.reset(split="train")
            expert_action = self._get_repo_expert_action()
            if expert_action is None:
                continue

            episodes_used += 1
            patch_np = obs["patch"]
            energy = float(obs["energy"])
            scenario_id = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0)))
            env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

            h_w = torch.zeros(
                1,
                1,
                self.agent.world_model.gru.hidden_size,
                device=self.device,
            )
            max_steps_ep = int(max(1, max_steps))

            for _step in range(max_steps_ep):
                patch_t = torch.from_numpy(patch_np).long().unsqueeze(0).to(self.device)
                H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                env_desc_t = self._env_desc_from_ids(env_t)
                text_t = self._text_tokens_from_ids(env_t, scenario_t)

                with torch.no_grad():
                    z_obs = self.agent.perception(
                        patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t
                    )
                W_t = h_w.squeeze(0)
                traits = self._mixed_traits()
                M = self.agent.memory
                S_t = torch.zeros(1, self.agent.self_model.gru.hidden_size, device=self.device)
                V_pi = self.agent.value_model(W_t, H_t, traits, M)
                delta_self = torch.zeros(1, 1, device=self.device)
                U_t = torch.zeros(1, 1, device=self.device)
                G_t = self.agent.workspace(
                    W_t,
                    S_t,
                    H_t,
                    V_pi,
                    delta_self,
                    U_t,
                    traits,
                    M,
                )

                logits_raw, mask_logits_pred = self._policy_forward_with_mask(G_t)
                if planning_coef_eff > 0.0:
                    with torch.no_grad():
                        planner_logits = self._get_planner_logits(
                            z_obs=z_obs,
                            H_t=H_t,
                            h_w=h_w,
                            traits=traits,
                            M=M,
                        )
                    logits_raw = (1.0 - planning_coef_eff) * logits_raw + planning_coef_eff * planner_logits

                expert_action = self._get_repo_expert_action()
                if expert_action is None:
                    break
                target = torch.tensor([int(expert_action)], dtype=torch.long, device=self.device)
                action_loss = F.cross_entropy(logits_raw, target)
                mask_loss = torch.tensor(0.0, device=self.device, dtype=action_loss.dtype)
                mask = self._get_action_mask_for_logits(logits_raw)
                if (
                    mask is not None
                    and mask_logits_pred is not None
                    and mask_pred_coef > 0.0
                    and mask_logits_pred.numel() > 0
                ):
                    mask_targets = mask.to(mask_logits_pred.dtype)
                    mask_loss = F.binary_cross_entropy_with_logits(mask_logits_pred, mask_targets)

                loss = action_loss + mask_pred_coef * mask_loss

                opt.zero_grad()
                loss.backward()
                params_to_clip = [p for group in opt.param_groups for p in group["params"]]
                nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                opt.step()

                loss_sum += float(loss.detach().item())
                action_loss_sum += float(action_loss.detach().item())
                mask_loss_sum += float(mask_loss.detach().item())
                steps_total += 1

                action_t = torch.tensor([int(expert_action)], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    _, h_w_new, _, _ = self.agent.world_model.forward_step(z_obs, H_t, action_t, h_w)
                h_w = h_w_new.detach()

                next_obs, _reward, done, info = self.env.step(int(expert_action))
                patch_np = next_obs["patch"]
                energy = float(next_obs["energy"])
                scenario_id = int(next_obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id)))
                env_id = int(next_obs.get("env_id", env_id))

                if done:
                    solved_flag = self._infer_episode_success_from_info(info)
                    if solved_flag is None:
                        env_inst = self._get_active_env_instance()
                        fallback_info = {
                            "last_test_passed": getattr(env_inst, "last_test_passed", None),
                            "reason": info.get("reason", ""),
                            "reward_env": info.get("reward_env", None),
                        }
                        solved_flag = self._infer_episode_success_from_info(fallback_info)
                    if solved_flag is True:
                        solved += 1
                    break

        used = episodes_used > 0
        mean_loss = float(loss_sum / max(1, steps_total))
        mean_action_loss = float(action_loss_sum / max(1, steps_total))
        mean_mask_loss = float(mask_loss_sum / max(1, steps_total))
        success_rate = float(solved) / float(max(1, episodes_used))

        print(
            f"[Repo-BC] episodes={episodes_used}/{episodes_req}, steps={steps_total}, "
            f"success_rate={success_rate:.3f}, mean_loss={mean_loss:.4f}"
        )
        return {
            "used": bool(used),
            "episodes_requested": float(episodes_req),
            "episodes_used": float(episodes_used),
            "steps": float(steps_total),
            "success_rate": float(success_rate),
            "mean_loss": float(mean_loss),
            "mean_action_loss": float(mean_action_loss),
            "mean_mask_pred_loss": float(mean_mask_loss),
        }

    def train_policy_a2c(
        self,
        n_steps: int = 1024,
        gamma: float = 0.99,
        entropy_coef: float = 0.003,
        use_self: bool = False,
        curiosity_beta: float = 0.1,
        beta_conflict: float = 0.0,
        beta_uncertainty: float = 0.0,
        planning_coef: float = 0.0,
        regime_name: Optional[str] = None,
    ):
        """Stage 2 or 4: A2C policy/value training (self/planner toggled by args)."""
        self.current_regime_name = str(regime_name or ("stage4" if use_self else "stage2"))
        # Optional: enable autograd anomaly detection when debugging inplace issues
        if getattr(self, "debug_autograd", False):
            torch.autograd.set_detect_anomaly(True)
        # мета-адаптивный коэффициент любопытства
        cur_beta_eff = self.get_adaptive_curiosity_beta(curiosity_beta)
        # мета-адаптивный коэффициент планирования
        planning_coef_eff = self.get_adaptive_planning_coef(planning_coef)

        (
            actions,
            rewards,
            dones,
            values,
            logprobs,
            entropies,
            conflicts,
            uncertainties,
            invalid_action_mass,
            mask_pred_logits,
            mask_targets,
            online_bc_logits,
            online_bc_targets,
        ) = self.collect_onpolicy_experience(
            n_steps=n_steps,
            use_self=use_self,
            curiosity_beta=cur_beta_eff,
            beta_conflict=beta_conflict,
            beta_uncertainty=beta_uncertainty,
            planning_coef=planning_coef_eff,
        )

        stats = self._run_a2c_update(
            rewards=rewards,
            dones=dones,
            values=values,
            logprobs=logprobs,
            entropies=entropies,
            conflicts=conflicts,
            uncertainties=uncertainties,
            invalid_action_mass=invalid_action_mass,
            mask_pred_logits=mask_pred_logits,
            mask_targets=mask_targets,
            online_bc_logits=online_bc_logits,
            online_bc_targets=online_bc_targets,
            gamma=gamma,
            entropy_coef=entropy_coef,
            beta_conflict=beta_conflict,
            beta_uncertainty=beta_uncertainty,
            regularization_coef=0.0,
        )

        self.update_memory_from_buffer(
            batch_size=32,
            seq_len=32,
            memory_alpha=0.05,
        )

        # обновляем мета-статистики (медленное состояние "я запутан / я уверен")

        # отладочные метрики по self-модели / мета-контролю
        mean_conflict = float(conflicts.mean().item())
        mean_uncertainty = float(uncertainties.mean().item())
        if use_self:
            self._update_meta_stats(conflicts, uncertainties)
        meta_conflict_ma = float(self.meta_conflict_ma)
        meta_uncertainty_ma = float(self.meta_uncertainty_ma)
        planner_usage_frac = float(getattr(self, "last_planner_usage_frac", 0.0))
        planner_debug = getattr(self, "last_planner_debug", {})
        if not isinstance(planner_debug, dict):
            planner_debug = {}
        planner_alpha_mean = float(planner_debug.get("planner_alpha_mean", 0.0) or 0.0)
        planner_alpha_p90 = float(planner_debug.get("planner_alpha_p90", 0.0) or 0.0)
        planner_js_mean = float(planner_debug.get("planner_js_mean", 0.0) or 0.0)
        planner_margin_mean = float(planner_debug.get("planner_margin_mean", 0.0) or 0.0)
        planner_override_rate = float(planner_debug.get("planner_override_rate", 0.0) or 0.0)
        entropy_eff = stats.get("entropy_coef_eff", entropy_coef)
        mean_invalid_mass = float(stats.get("mean_invalid_action_mass", 0.0))
        mask_pred_f1 = stats.get("mask_pred_f1", float("nan"))
        mask_pred_auc = stats.get("mask_pred_auc", float("nan"))
        online_bc_loss = float(stats.get("online_bc_loss", 0.0))
        online_bc_samples = float(stats.get("online_bc_samples", 0.0))
        gae_lambda_eff = float(stats.get("gae_lambda_eff", getattr(self, "gae_lambda_base", 0.95)))
        print(
            f"[A2C] use_self={use_self} | "
            f"mean_conflict={mean_conflict:.4f}, mean_uncertainty={mean_uncertainty:.4f}, "
            f"mean_invalid_mass={mean_invalid_mass:.4f} | "
            f"meta_conflict_ma={meta_conflict_ma:.4f}, meta_uncertainty_ma={meta_uncertainty_ma:.4f} | "
            f"entropy_coef_eff={entropy_eff:.5f}, curiosity_beta_eff={cur_beta_eff:.5f}, "
            f"planning_coef_eff={planning_coef_eff:.5f}, gae_lambda_eff={gae_lambda_eff:.4f}, "
            f"planner_usage_frac={planner_usage_frac:.3f}, planner_alpha_mean={planner_alpha_mean:.3f}, "
            f"planner_alpha_p90={planner_alpha_p90:.3f}, planner_js_mean={planner_js_mean:.3f}, "
            f"planner_margin_mean={planner_margin_mean:.3f}, planner_override_rate={planner_override_rate:.3f} | "
            f"beta_conflict={beta_conflict:.3f}, beta_uncertainty={beta_uncertainty:.3f}, "
            f"mask_pred_f1={mask_pred_f1:.3f}, mask_pred_auc={mask_pred_auc:.3f}, "
            f"online_bc_loss={online_bc_loss:.4f}, online_bc_samples={online_bc_samples:.0f}"
        )
        return stats

    # =========================
    #  Stage 3: SelfModel offline
    # =========================

    def _collect_self_model_data_for_current_traits(
        self,
        n_episodes: int,
        max_steps: int = 200,
        use_self: bool = True,
        planning_coef: float = 0.0,
        split: str = "train",
        reward_profile: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Roll out episodes with the current traits/policy and push them into replay for SelfModel fine-tuning.
        Lightweight stats are returned for logging; models stay frozen during collection.
        """
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.eval()
        self.agent.workspace.eval()
        self.agent.policy.eval()
        self.agent.value_model.eval()

        alpha_conf = 0.2
        alpha_unc = 0.2

        returns: List[float] = []
        lengths: List[int] = []
        foods: List[int] = []
        damages: List[int] = []

        for _ in range(max(0, int(n_episodes))):
            obs = self.env.reset(split=split)
            patch = obs["patch"]
            energy = obs["energy"]
            scenario_id = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0)))
            env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))
            env_name = obs.get("env_name", getattr(self.env, "env_name", "env"))
            scenario_name = obs.get(
                "scenario_name", getattr(self.env, "current_scenario_name", f"scenario_{scenario_id}")
            )

            h_w = torch.zeros(1, 1, self.agent.world_model.gru.hidden_size, device=self.device)
            h_s = torch.zeros(1, 1, self.agent.self_model.gru.hidden_size, device=self.device)
            last_action = torch.zeros(1, dtype=torch.long, device=self.device)
            last_reward = 0.0
            traits = self._mixed_traits()
            M = self.agent.memory

            total_r = 0.0
            t = 0
            done = False
            food_count = 0
            damage_count = 0

            while not done and t < max_steps:
                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)

                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    W_t = h_w.squeeze(0)

                    if use_self:
                        r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                        a_emb = self.agent.self_model.act_emb(last_action)
                        env_emb = self.agent.self_model.env_desc_to_emb(env_desc_t)

                        W_in = W_t.unsqueeze(1)
                        H_in = H_t.unsqueeze(1)
                        r_in = r_t.unsqueeze(1)
                        M_b = M.unsqueeze(1)
                        x_s = torch.cat([W_in, H_in, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)], dim=-1)
                        out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                        S_t = out_s.squeeze(1)

                        surv_t = torch.sigmoid(self.agent.self_model.head_survival(S_t))
                        food_t = self.agent.self_model.head_food(S_t)
                        dmg_t = self.agent.self_model.head_damage(S_t)
                        move_t = self.agent.self_model.head_move(S_t)

                        w = traits_to_preference_weights(traits)
                        w_survive, w_food, w_danger, w_move = w[0]
                        R_unscaled = (
                            w_survive * surv_t
                            + w_food * food_t
                            + w_danger * dmg_t
                            + w_move * move_t
                        ).view(1, 1)
                        R_self = self.agent.self_model.head_return_calib(R_unscaled)
                        U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
                    else:
                        S_t = torch.zeros(1, self.agent.self_model.gru.hidden_size, device=self.device)
                        R_self = torch.zeros(1, 1, device=self.device)
                        U_t = torch.zeros(1, 1, device=self.device)

                    V_pi = self.agent.value_model(W_t, H_t, traits, M)

                    if use_self:
                        conf_t = torch.abs(R_self - V_pi)
                        gate = torch.exp(-alpha_conf * conf_t - alpha_unc * U_t)
                        delta_raw = torch.tanh(R_self - V_pi)
                        delta_self = delta_raw * gate
                    else:
                        delta_self = torch.zeros_like(R_self)

                    G_t = self.agent.workspace(
                        W_t,
                        S_t,
                        H_t,
                        V_pi,
                        delta_self,
                        U_t,
                        traits,
                        M,
                    )
                    logits = self.agent.policy(G_t)

                    if planning_coef > 0.0:
                        planner_logits = self._get_planner_logits(
                            z_obs=z_obs,
                            H_t=H_t,
                            h_w=h_w,
                            traits=traits,
                            M=M,
                        )
                        logits = (1.0 - planning_coef) * logits + planning_coef * planner_logits

                    if not torch.isfinite(logits).all():
                        bad = ~torch.isfinite(logits)
                        logger.warning(
                            "[PhaseC][online] Non-finite logits detected; "
                            f"replacing with zeros. "
                            f"min={logits[torch.isfinite(logits)].min().item() if torch.isfinite(logits).any() else 'n/a'}, "
                            f"max={logits[torch.isfinite(logits)].max().item() if torch.isfinite(logits).any() else 'n/a'}, "
                            f"nan_count={bad.sum().item()}"
                        )
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

                    logits = self._apply_action_mask(logits)
                    dist = Categorical(logits=logits)
                    action = dist.sample()

                next_obs, _, done, info = self.env.step(action.item())
                reward_env = self.compute_preference_reward(info, reward_profile=reward_profile)
                total_r += reward_env
                t += 1

                if info.get("got_food", False):
                    food_count += 1
                if info.get("took_damage", False):
                    damage_count += 1

                got_food = 1.0 if info.get("got_food", False) else 0.0
                took_damage = 1.0 if info.get("took_damage", False) else 0.0
                moved = 1.0 if info.get("moved", False) else 0.0
                alive = 1.0 if info.get("alive", True) else 0.0
                death_flag = float(info.get("death_flag", 0.0))
                scenario_id = int(info.get("scenario_id", scenario_id))
                env_id = int(info.get("env_id", env_id))

                tr = Transition(
                    obs_patch=patch,
                    energy=energy,
                    action=int(action.item()),
                    reward=reward_env,
                    done=done,
                    next_obs_patch=next_obs["patch"],
                    next_energy=next_obs["energy"],
                    death_flag=death_flag,
                    got_food=got_food,
                    took_damage=took_damage,
                    moved=moved,
                    alive=alive,
                    scenario_id=scenario_id,
                    env_id=env_id,
                    regime_name=self.current_regime_name or "stage4",
                )
                self.buffer.push(tr)

                patch = next_obs["patch"]
                energy = next_obs["energy"]

                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)
                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs_next = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    _, h_w, _, _ = self.agent.world_model.forward_step(z_obs_next, H_t, action, h_w)

                last_reward = reward_env
                last_action = action

            returns.append(total_r)
            lengths.append(t)
            foods.append(food_count)
            damages.append(damage_count)
            if self.logger is not None:
                env_desc_np = None
                if self.env_descriptors is not None:
                    try:
                        env_desc_np = self.env_descriptors[env_id].detach().cpu().numpy().tolist()
                    except Exception:
                        env_desc_np = None
                self.logger.log_scalar(
                    stage=f"collect_{split}_{'self' if use_self else 'no_self'}",
                    metric="episode_return",
                    value=float(total_r),
                    env_id=int(env_id),
                    env_name=str(env_name),
                    scenario_id=int(scenario_id),
                    scenario_name=str(scenario_name),
                    env_descriptor=env_desc_np,
                )

        mean_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns)) if returns else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        mean_food = float(np.mean(foods)) if foods else 0.0
        mean_damage = float(np.mean(damages)) if damages else 0.0

        return {
            "n_episodes": int(len(returns)),
            "mean_return": mean_ret,
            "std_return": std_ret,
            "mean_length": mean_len,
            "mean_food": mean_food,
            "mean_damage": mean_damage,
            "returns": [float(x) for x in returns],
            "lengths": [int(x) for x in lengths],
        }

    def _train_self_model_from_buffer(
        self,
        batch_size: int = 32,
        seq_len: int = 32,
        n_batches: int = 200,
        gamma: float = 0.99,
        lr_scale: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Shared supervised SelfModel training loop used by Stage 3a and Stage 3c.
        Returns a small dict with the number of batches and last loss for logging.
        """
        self.agent.self_model.train()

        opt = self.agent.optim_self
        base_lrs = [g["lr"] for g in opt.param_groups]
        if lr_scale != 1.0:
            for g, base_lr in zip(opt.param_groups, base_lrs):
                g["lr"] = base_lr * lr_scale

        steps = 0
        last_loss_val: Optional[float] = None
        loss_w = loss_weights or {}
        # loss weights (kept separate from preference weights)
        lw_surv = float(loss_w.get("surv", loss_w.get("survival", 1.5)))
        lw_food = float(loss_w.get("food", 1.0))
        lw_dmg = float(loss_w.get("dmg", loss_w.get("damage", 1.0)))
        lw_move = float(loss_w.get("move", 1.0))
        lw_unc = float(loss_w.get("unc", loss_w.get("uncertainty", 0.2)))
        lw_ret = float(loss_w.get("ret", loss_w.get("return", 1.0)))
        try:
            for _ in range(n_batches):
                if len(self.buffer) < seq_len:
                    break

                (
                    obs_seq,
                    H_seq,
                    a_seq,
                    r_seq,
                    d_seq,
                    death_seq,
                    food_seq,
                    dmg_seq,
                    move_seq,
                    alive_seq,
                    scenario_seq,
                    env_seq,
                ) = self._sample_sequences_from_buffer(
                    self.buffer,
                    batch_size,
                    seq_len,
                    with_events=True,
                    current_regime=self.current_regime_name,
                )

                B, T, p, _ = obs_seq.shape

                patch = torch.from_numpy(obs_seq).long().to(self.device)
                H = torch.from_numpy(H_seq).float().to(self.device)
                a = torch.from_numpy(a_seq).long().to(self.device)
                r = torch.from_numpy(r_seq).float().to(self.device)
                d = torch.from_numpy(d_seq).float().to(self.device)
                death = torch.from_numpy(death_seq).float().to(self.device)
                food = torch.from_numpy(food_seq).float().to(self.device)
                dmg = torch.from_numpy(dmg_seq).float().to(self.device)
                move = torch.from_numpy(move_seq).float().to(self.device)
                alive = torch.from_numpy(alive_seq).float().to(self.device)
                scenario = torch.from_numpy(scenario_seq).long().to(self.device)
                env_ids = torch.from_numpy(env_seq).long().to(self.device)
                env_desc_seq = self._env_desc_from_ids(env_ids)

                with torch.no_grad():
                    patch_flat = patch.view(B * T, p, p)
                    H_flat = H.view(B * T, 1)
                    scenario_flat = scenario.view(B * T)
                    env_flat = env_ids.view(B * T)
                    env_desc_flat = env_desc_seq.reshape(B * T, -1) if env_desc_seq is not None else None

                    text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
                    z_flat = self.agent.perception(
                        patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
                    )
                    z_seq = z_flat.reshape(B, T, -1)

                    a_emb = self.agent.world_model.act_emb(a)
                    x_w = torch.cat([z_seq, H, a_emb], dim=-1)
                    h0 = torch.zeros(
                        1,
                        B,
                        self.agent.world_model.gru.hidden_size,
                        device=self.device,
                    )
                    out_w, _ = self.agent.world_model.gru(x_w, h0)
                    W_seq = out_w  # (B,T,w_dim)

                done_mask = d > 0.5
                done_any = done_mask.any(dim=1)
                done_idx = torch.argmax(done_mask.int(), dim=1)
                death_at_done = torch.gather(death, 1, done_idx.unsqueeze(1)).squeeze(1)
                ep_len = done_idx + 1
                frac = torch.ones(B, device=self.device, dtype=torch.float32)
                env_at_done = torch.gather(env_ids, 1, done_idx.unsqueeze(1)).squeeze(1)
                max_steps = self._env_max_steps_from_ids(env_at_done).clamp(min=1.0)
                frac = torch.where(done_any & (death_at_done > 0.5), ep_len.float() / max_steps, frac)
                frac = torch.clamp(frac, min=0.0, max=1.0)
                S_alive = frac.view(B, 1).expand(-1, T)  # (B,T)

                def discounted_sums(x: torch.Tensor, done_mask: torch.Tensor, gamma: float) -> torch.Tensor:
                    """
                    x, done_mask: (B,T), done_mask=1.0 where terminal at t.
                    Computes G_t = sum_{k>=t} gamma^{k-t} x_k with termination where done_mask flips on.
                    """
                    B_, T_ = x.shape
                    idx = torch.arange(T_, device=x.device)
                    gamma_mat = torch.triu(gamma ** (idx[None, :] - idx[:, None]))  # (T,T)
                    not_done = 1.0 - done_mask
                    prefix = torch.cumprod(torch.cat([torch.ones(B_, 1, device=x.device, dtype=x.dtype), not_done], dim=1), dim=1)
                    prefix_t = prefix[:, :-1]  # (B,T)
                    prefix_k = prefix[:, 1:]   # (B,T)
                    prefix_t_safe = torch.clamp(prefix_t, min=1e-8)
                    ratio = prefix_k.unsqueeze(1) / prefix_t_safe.unsqueeze(2)  # (B,T,T)
                    weights = gamma_mat.unsqueeze(0) * ratio
                    return (weights * x.unsqueeze(1)).sum(dim=2)  # (B,T)

                G_food = discounted_sums(food.squeeze(-1), d, gamma)
                G_dmg = discounted_sums(dmg.squeeze(-1), d, gamma)
                G_move = discounted_sums(move.squeeze(-1), d, gamma)
                G_ret = discounted_sums(r.squeeze(-1), d, gamma)

                surv_target = S_alive.unsqueeze(-1)   # (B,T,1)
                food_target = G_food.unsqueeze(-1)    # (B,T,1)
                dmg_target = G_dmg.unsqueeze(-1)      # (B,T,1)
                move_target = G_move.unsqueeze(-1)    # (B,T,1)
                ret_target = G_ret.unsqueeze(-1)      # (B,T,1)

                M = self.agent.memory  # (1,mem_dim)

                (
                    S_seq,
                    S_last,
                    surv_pred,
                    food_pred,
                    dmg_pred,
                    move_pred,
                    unc_pred,
                    surv_raw_pred,
                ) = self.agent.self_model.forward_seq(
                    W_seq,
                    H,
                    a,
                    r,
                    M=M,
                    env_desc=env_desc_seq,
                )

                loss_surv_main = F.binary_cross_entropy(surv_pred, surv_target)
                loss_surv_raw = F.binary_cross_entropy(surv_raw_pred, surv_target)
                loss_surv = 1.5 * loss_surv_main + 0.5 * loss_surv_raw
                loss_food = F.mse_loss(food_pred, food_target)
                loss_dmg = F.mse_loss(dmg_pred, dmg_target)
                loss_move = F.mse_loss(move_pred, move_target)

                with torch.no_grad():
                    pref = self._combined_preference_weights().detach()
                    pref_surv, pref_food, pref_danger, pref_move = pref.view(-1)
                R_unscaled_pred = (
                    pref_surv * surv_pred
                    + pref_food * food_pred
                    + pref_danger * dmg_pred
                    + pref_move * move_pred
                )
                R_calib_pred = self.agent.self_model.head_return_calib(R_unscaled_pred)
                loss_ret = F.mse_loss(R_calib_pred, ret_target)

                with torch.no_grad():
                    err_food = torch.abs(food_target - food_pred)
                    err_dmg = torch.abs(dmg_target - dmg_pred)
                    err_move = torch.abs(move_target - move_pred)
                    err_surv = torch.abs(surv_target - surv_pred)
                    err_total = err_food + err_dmg + err_move + err_surv
                    unc_target = err_total

                loss_unc = F.mse_loss(unc_pred, unc_target)

                loss = (
                    lw_surv * loss_surv
                    + lw_food * loss_food
                    + lw_dmg * loss_dmg
                    + lw_move * loss_move
                    + lw_unc * loss_unc
                    + lw_ret * loss_ret
                )

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.self_model.parameters(), 1.0)
                opt.step()

                steps += 1
                last_loss_val = float(loss.detach().item())
        finally:
            if lr_scale != 1.0:
                for g, base_lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = base_lr

        return {"steps": steps, "last_loss": last_loss_val, "lr_scale": float(lr_scale)}

    def train_self_model_offline(
        self,
        batch_size: int = 32,
        seq_len: int = 32,
        n_batches: int = 200,
        gamma: float = 0.99,
    ):
        """Stage 3: offline self-model training on replay sequences."""
        return self._train_self_model_from_buffer(
            batch_size=batch_size,
            seq_len=seq_len,
            n_batches=n_batches,
            gamma=gamma,
            lr_scale=1.0,
        )

    # =========================
    #  Stage 3b: self-reflection on traits
    # =========================

    def self_reflect_on_traits(
        self,
        n_batches: int = 200,
        batch_size: int = 32,
        seq_len: int = 32,
        gamma: float = 0.99,
        lr: float = 5e-2,
        memory_alpha: float = 0.1,
        lambda_reg: float = 1e-2,
        lambda_conflict: float = 0.0,
        lambda_self: float = 0.3,
    ):
        """
        Stage 3b self-reflection on traits (only traits updated).
        Uses small gradient-like updates on traits guided by the SelfModel.
        """
        print("Stage 3b: self-reflection on traits...")
        print("Traits before:")
        self.print_traits()

        self.agent.self_model.eval()
        step_size = float(lr)
        if self.is_minigrid:
            step_size = float(self.trait_reflection_lr)
        probe_stats = self.last_self_probe
        if probe_stats is None:
            probe_stats = self.run_selfmodel_probe(use_self=False)
        cfg = self.safety
        corr_ret = probe_stats.corr_return if math.isfinite(probe_stats.corr_return) else 0.0
        corr_surv = probe_stats.corr_survival if math.isfinite(probe_stats.corr_survival) else 0.0
        corr_surv_defined = bool(getattr(probe_stats, "corr_survival_defined", True))
        corr_ret_defined = bool(getattr(probe_stats, "corr_return_defined", True))
        if (
            cfg.skip_reflection_if_very_low_quality
            and abs(corr_ret) < cfg.very_low_corr_threshold
            and abs(corr_surv) < cfg.very_low_corr_threshold
        ):
            if self.logger is not None:
                self.logger.log(
                    {
                        "event": "self_reflection_skipped",
                        "reason": "self_model_very_low_quality",
                        "corr_return": corr_ret,
                        "corr_survival": corr_surv,
                    }
                )
            return
        if not probe_stats.good_enough_for_reflection(cfg):
            if self.logger is not None:
                self.logger.log(
                    {
                        "event": "self_reflection_skipped",
                        "reason": "probe_not_good_enough",
                        "corr_return": corr_ret,
                        "corr_survival": corr_surv,
                        "num_episodes": int(probe_stats.num_samples),
                    }
                )
            logger.info(
                "[Self-reflection][stage3b] Skipping traits update: "
                f"n={probe_stats.num_samples}, "
                f"corr_return={corr_ret:.3f}, "
                f"corr_survival={corr_surv:.3f}, "
                f"min_r={cfg.min_return_corr}, min_s={cfg.min_survival_corr}"
            )
            return
        reflection_lr = step_size
        if (
            (corr_ret_defined and abs(corr_ret) < cfg.min_return_corr)
            or (corr_surv_defined and abs(corr_surv) < cfg.min_survival_corr)
        ):
            reflection_lr *= cfg.reflection_lr_scale_when_low_quality
            if self.logger is not None:
                self.logger.log(
                    {
                        "event": "self_reflection_lr_scaled",
                        "scale": cfg.reflection_lr_scale_when_low_quality,
                        "corr_return": corr_ret,
                        "corr_survival": corr_surv,
                    }
                )
        step_size = reflection_lr
        base_traits_main = self._main_traits().detach().clone()
        self._start_trait_safety_phase("stage3b", base_traits_main)
        base_weights = get_faction_preference_weights(self.agent)
        w_survive_base = float(base_weights[0, 0].item()) if base_weights.numel() > 0 else 0.0
        w_danger_base = float(base_weights[0, 2].item()) if base_weights.numel() > 0 else 0.0
        stage_anchor = self.agent.traits.detach().clone()
        reflection_steps_total = 0
        reflection_step_norms: List[float] = []

        n_batches_eff = n_batches
        if self.is_minigrid:
            n_batches_eff = int(n_batches * 1.5)
        reward_profile = None
        if self.is_minigrid:
            reward_profile = self._reward_profile_for_regime("R2")

        for _ in range(n_batches_eff):
            if len(self.buffer) < seq_len:
                break

            batch = self._sample_sequences_from_buffer(
                self.buffer,
                batch_size,
                seq_len,
                with_events=True,
                current_regime=self.current_regime_name,
            )
            info = self._run_trait_reflection(
                batch=batch,
                reward_profile=reward_profile,
                step_size=step_size,
                regime_name="stage3b_offline",
                lambda_l2=lambda_reg,
                n_steps=self.trait_reflection_steps_per_batch,
                trait_anchor=stage_anchor,
                init_traits=stage_anchor,
                safety_phase="stage3b",
                safety_baseline=base_traits_main,
                log_progress=False,
            )
            reflection_steps_total += int(info.get("steps", 0))
            reflection_step_norms.extend(info.get("step_norms", []))

        # final clamp/priors adjustments identical to previous behavior
        with torch.no_grad():
            self.agent.traits.clamp_(-2.0, 2.0)
            w = traits_to_preference_weights(self.agent.traits)
            w_survive, w_food, w_danger, w_move = w[0]

            if w_survive < 0.6:
                delta = 0.6 - w_survive
                self.agent.traits[0, 0] += delta

            if w_danger > -0.5:
                delta = w_danger + 0.5
                self.agent.traits[0, 2] -= delta

            if w_move < -0.2:
                delta = -0.2 - w_move
                self.agent.traits[0, 3] += delta
            elif w_move > 0.9:
                delta = w_move - 0.9
                self.agent.traits[0, 3] -= delta

            self.agent.traits.clamp_(-2.0, 2.0)
        self.apply_safe_trait_update(self.agent, self._main_traits().detach().clone(), phase="stage3b", finalize=True)
        weights_after = get_faction_preference_weights(self.agent)
        w_survive_after = float(weights_after[0, 0].item()) if weights_after.numel() > 0 else 0.0
        w_danger_after = float(weights_after[0, 2].item()) if weights_after.numel() > 0 else 0.0
        delta_main = float(torch.norm(self._main_traits() - base_traits_main).item())
        if self.logger is not None:
            self.logger.log(
                {
                    "event": "traits_reflection_summary",
                    "phase": "stage3b",
                    "traits_main_before": base_traits_main.cpu().tolist(),
                    "traits_main_after": self._main_traits().detach().cpu().tolist(),
                    "w_survive_before": w_survive_base,
                    "w_survive_after": w_survive_after,
                    "w_danger_before": w_danger_base,
                    "w_danger_after": w_danger_after,
                    "delta_norm": delta_main,
                }
            )

        final_dist = float(torch.norm(self.agent.traits - stage_anchor).item())
        mean_step_norm = float(np.mean(reflection_step_norms)) if reflection_step_norms else 0.0
        self.trait_reflection_debug["stage3b_offline"] = {
            "trait_reflection_steps": int(reflection_steps_total),
            "trait_reflection_final_dist_from_init": final_dist,
            "trait_reflection_final_dist_from_anchor": final_dist,
            "trait_reflection_mean_step_norm": mean_step_norm,
        }
        self._trait_safety_ctx.pop("stage3b", None)

        decision_summary = self.describe_trait_delta(
            base_traits_main.detach().cpu().numpy(), self._main_traits().detach().cpu().numpy()
        )
        summary_msg = (
            "[Self-reflection][stage3b_offline] "
            f"Reflection finished: decision='{decision_summary}', "
            f"steps={int(reflection_steps_total)}, lr={float(step_size):.4f}, "
            f"||delta||={delta_main:.4f}, mean_step_norm={mean_step_norm:.4f}"
        )
        logger.info(summary_msg)
        print(summary_msg)
        print("Traits after:")
        self.print_traits()

    def run_stage3c_self_model_trait_co_learning(
        self,
        n_collect_episodes: int = 10,
        max_steps: int = 200,
        use_self_for_collection: bool = True,
        planning_coef: float = 0.0,
        split: str = "train",
        train_kwargs: Optional[Dict[str, Any]] = None,
        lr_scale: float = 0.5,
        probe_gamma: float = 0.99,
    ) -> Dict[str, Any]:
        """
        Stage 3c: self/value co-learning.

        1) Collect a few fresh episodes under the current traits (post-reflection).
        2) Fine-tune the SelfModel on the combined buffer.
        3) Probe the SelfModel again to track calibration.
        """
        if not hasattr(self.agent, "self_model"):
            return {"collected": None, "train_stats": None, "probe_after": None}

        print("Stage 3c: self/value co-learning (collect -> fine-tune self-model -> probe)...")
        train_cfg = dict(train_kwargs or {})
        default_n_batches = 50
        if self.env_family == "minigrid" and "n_batches" not in train_cfg:
            default_n_batches = 200
        n_batches = int(train_cfg.pop("n_batches", default_n_batches))
        batch_size = int(train_cfg.pop("batch_size", 32))
        seq_len = int(train_cfg.pop("seq_len", 32))
        gamma = float(train_cfg.pop("gamma", probe_gamma))

        n_collect = n_collect_episodes
        if self.env_family == "minigrid" and n_collect == 10:
            n_collect = 30

        lr_scale_eff = lr_scale
        if self.env_family == "minigrid" and lr_scale_eff <= 0.6:
            lr_scale_eff = 1.5

        collected_stats: Optional[Dict[str, Any]] = None
        if n_collect > 0:
            collected_stats = self._collect_self_model_data_for_current_traits(
                n_episodes=n_collect,
                max_steps=max_steps,
                use_self=use_self_for_collection,
                planning_coef=planning_coef,
                split=split,
            )

        loss_weights = None
        if self.env_family == "minigrid":
            loss_weights = {
                "surv": 1.0,
                "food": 1.0,
                "dmg": 1.0,
                "move": 1.0,
                "unc": 0.2,
                "ret": 2.0,
            }

        train_stats = self._train_self_model_from_buffer(
            batch_size=batch_size,
            seq_len=seq_len,
            n_batches=n_batches,
            gamma=gamma,
            lr_scale=lr_scale_eff,
            loss_weights=loss_weights,
        )
        probe_after = self.probe_self_model(n_episodes=30, max_steps=max_steps, gamma=probe_gamma)

        return {
            "collected": collected_stats,
            "train_stats": train_stats,
            "probe_after": probe_after,
            "use_self_for_collection": bool(use_self_for_collection),
            "planning_coef_for_collection": float(planning_coef),
            "n_collect_episodes": int(n_collect),
        }
    # =========================
    # Evaluation
    # =========================
    def evaluate(
        self,
        n_episodes: int = 20,
        max_steps: int = 200,
        use_self: bool = False,
        planning_coef: float = 0.0,
        eval_policy: str = "sample",
    ):
        """
        Оценка политики с учётом multi-env, self-модели и планировщика.
        Для каждого эпизода логируем сценарии/среды и считаем отдельные train/test метрики.
        """
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.eval()
        self.agent.workspace.eval()
        self.agent.policy.eval()
        self.agent.value_model.eval()

        alpha_conf = 0.2
        alpha_unc = 0.2

        def _classify_env(env_name: str, env_id: int) -> str:
            if isinstance(env_name, str):
                if env_name.startswith("train_"):
                    return "train"
                if env_name.startswith("test_"):
                    return "test"
            if env_id in self.train_env_ids:
                return "train"
            if env_id in self.test_env_ids:
                return "test"
            return "other"

        def _get_env_desc(env_tensor: torch.Tensor) -> torch.Tensor:
            env_desc = self._env_desc_from_ids(env_tensor)
            if env_desc is None:
                in_dim = self.agent.perception.env_desc_to_emb[0].in_features
                env_desc = torch.zeros(
                    env_tensor.shape[0], in_dim, device=self.device, dtype=torch.float32
                )
            return env_desc

        eval_policy_norm = (eval_policy or "sample").lower()
        if eval_policy_norm not in {"sample", "greedy"}:
            eval_policy_norm = "sample"

        def _supports_action_mask(env_obj: Any) -> bool:
            envs_list = getattr(env_obj, "envs", None)
            if envs_list is not None:
                for sub_env in envs_list:
                    get_mask_fn = getattr(sub_env, "get_action_mask", None)
                    set_mask_fn = getattr(sub_env, "set_action_mask_enabled", None)
                    if callable(get_mask_fn) or callable(set_mask_fn):
                        return True
                return False
            get_mask_fn = getattr(env_obj, "get_action_mask", None)
            set_mask_fn = getattr(env_obj, "set_action_mask_enabled", None)
            return bool(callable(get_mask_fn) or callable(set_mask_fn))

        mask_supported = _supports_action_mask(self.env)
        prev_mask_enabled: Optional[bool] = None
        if mask_supported and hasattr(self.env, "get_action_mask_enabled"):
            try:
                prev_mask_enabled = bool(self.env.get_action_mask_enabled())
            except Exception:
                prev_mask_enabled = None

        def _set_mask_enabled(enabled: bool) -> None:
            if not mask_supported:
                return
            try:
                self.env.set_action_mask_enabled(bool(enabled))
            except Exception:
                pass

        def _run_eval(split_use_self: bool, mask_label: str) -> Dict[str, Any]:
            returns = []
            lengths = []
            foods = []
            damages = []
            timeout_episodes = 0
            death_episodes = 0
            constraint_violation_episodes = 0
            catastrophic_fail_episodes = 0
            damage_episodes = 0
            reason_counts: Dict[str, int] = {}
            scenario_counts: Dict[str, int] = {}
            env_counts: Dict[str, int] = {}
            train_returns = []
            test_returns = []
            per_task_returns: Dict[str, List[float]] = {} if self.is_minigrid else {}
            repo_pass_flags: List[bool] = []
            repo_steps_to_pass: List[int] = []
            repo_pass_flags_train: List[bool] = []
            repo_pass_flags_test: List[bool] = []
            repo_steps_to_pass_train: List[int] = []
            repo_steps_to_pass_test: List[int] = []
            instruction_success_flags: List[bool] = []
            instruction_success_flags_train: List[bool] = []
            instruction_success_flags_test: List[bool] = []
            social_success_flags: List[bool] = []
            social_success_flags_train: List[bool] = []
            social_success_flags_test: List[bool] = []
            planner_alpha_vals: List[float] = []
            planner_js_vals: List[float] = []
            planner_margin_vals: List[float] = []
            planner_override_vals: List[float] = []

            for _ in range(n_episodes):
                obs = self.env.reset()
                scenario_name = getattr(
                    self.env, "current_scenario_name", obs.get("scenario_name", "single")
                )
                scenario_counts[scenario_name] = scenario_counts.get(scenario_name, 0) + 1

                env_name = obs.get("env_name", getattr(self.env, "env_name", "env"))
                env_counts[env_name] = env_counts.get(env_name, 0) + 1

                patch = obs["patch"]
                energy = obs["energy"]
                scenario_id = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0)))
                env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

                h_w = torch.zeros(
                    1,
                    1,
                    self.agent.world_model.gru.hidden_size,
                    device=self.device,
                )
                h_s = torch.zeros(
                    1,
                    1,
                    self.agent.self_model.gru.hidden_size,
                    device=self.device,
                )
                last_action = torch.zeros(1, dtype=torch.long, device=self.device)
                last_reward = 0.0
                traits = self._mixed_traits()
                M = self.agent.memory
                skill_state = {
                    "logprob": None,
                    "entropy": None,
                    "obs_history": [],
                    "active_skill": None,
                    "step_in_skill": 0,
                }

                total_r = 0.0
                t = 0
                done = False
                food_count = 0
                damage_count = 0
                episode_had_damage = False
                episode_had_death = False
                episode_had_violation = False
                episode_had_catastrophic = False
                unc_episode = 0.0
                unc_steps = 0
                env_desc_np = None
                info: Dict[str, Any] = {}

                while not done and t < max_steps:
                    skill_state["obs_history"].append({"patch": patch.copy(), "energy": float(energy)})
                    with torch.no_grad():
                        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                        H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                        scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        env_desc_t = _get_env_desc(env_t)
                        if env_desc_t is not None:
                            env_desc_np = env_desc_t[0].detach().cpu().numpy().reshape(-1).tolist()

                        text_t = self._text_tokens_from_ids(env_t, scenario_t)
                        z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                        W_t = h_w.squeeze(0)

                        if split_use_self:
                            r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                            a_emb = self.agent.self_model.act_emb(last_action)
                            env_emb = self.agent.self_model.env_desc_to_emb(env_desc_t)

                            W_in = W_t.unsqueeze(1)
                            H_in = H_t.unsqueeze(1)
                            r_in = r_t.unsqueeze(1)
                            M_b = M.unsqueeze(1)
                            x_s = torch.cat(
                                [W_in, H_in, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)],
                                dim=-1,
                            )
                            out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                            S_t = out_s.squeeze(1)

                            surv_t = torch.sigmoid(self.agent.self_model.head_survival(S_t))
                            food_t = self.agent.self_model.head_food(S_t)
                            dmg_t = self.agent.self_model.head_damage(S_t)
                            move_t = self.agent.self_model.head_move(S_t)

                            w = traits_to_preference_weights(traits)
                            w_survive, w_food, w_danger, w_move = w[0]
                            R_unscaled = (
                                w_survive * surv_t
                                + w_food * food_t
                                + w_danger * dmg_t
                                + w_move * move_t
                            ).view(1, 1)
                            R_self = self.agent.self_model.head_return_calib(R_unscaled)
                            U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
                        else:
                            S_t = torch.zeros(
                                1,
                                self.agent.self_model.gru.hidden_size,
                                device=self.device,
                            )
                            R_self = torch.zeros(1, 1, device=self.device)
                            U_t = torch.zeros(1, 1, device=self.device)

                        unc_episode += float(U_t.mean().item())
                        unc_steps += 1

                        V_pi = self.agent.value_model(W_t, H_t, traits, M)

                        if split_use_self:
                            conf_t = torch.abs(R_self - V_pi)
                            gate = torch.exp(-alpha_conf * conf_t - alpha_unc * U_t)
                            delta_raw = torch.tanh(R_self - V_pi)
                            delta_self = delta_raw * gate
                        else:
                            delta_self = torch.zeros_like(R_self)

                        G_t = self.agent.workspace(
                            W_t,
                            S_t,
                            H_t,
                            V_pi,
                            delta_self,
                            U_t,
                            traits,
                            M,
                        )
                        if self.use_skills and self.agent.high_level_policy is not None and self._total_skill_count() > 0:
                            action, _, _, skill_state = self._select_action_with_skills(
                                G_t=G_t,
                                z_obs=z_obs,
                                H_t=H_t,
                                h_w=h_w,
                                traits=traits,
                                M=M,
                                planning_coef=planning_coef,
                                skill_state=skill_state,
                                env_desc=env_desc_t,
                                W_t=W_t,
                            )
                            if action is None:
                                logits_raw, mask_logits_pred = self._policy_forward_with_mask(G_t)
                                mask = self._get_action_mask_for_logits(logits_raw)
                                logits = self._compose_policy_logits_with_masks(
                                    logits_raw,
                                    mask,
                                    mask_logits_pred,
                                )
                                dist = Categorical(logits=logits)
                                action = dist.sample()
                        else:
                            logits_raw, mask_logits_pred = self._policy_forward_with_mask(G_t)
                            logits = logits_raw

                            if planning_coef > 0.0:
                                planner_logits = self._get_planner_logits(
                                    z_obs=z_obs,
                                    H_t=H_t,
                                    h_w=h_w,
                                    traits=traits,
                                    M=M,
                                )
                                logits, planner_debug = self._blend_with_planner(
                                    policy_logits=logits,
                                    planner_logits=planner_logits,
                                    base_planning_coef=float(planning_coef),
                                    uncertainty=U_t if split_use_self else None,
                                    r_self=R_self if split_use_self else None,
                                    v_pi=V_pi if split_use_self else None,
                                )
                                planner_alpha_vals.append(float(planner_debug.get("planner_alpha", 0.0)))
                                planner_js_vals.append(float(planner_debug.get("planner_js", 0.0)))
                                planner_margin_vals.append(float(planner_debug.get("planner_margin", 0.0)))
                                planner_override_vals.append(float(planner_debug.get("planner_override", 0.0)))

                            mask = self._get_action_mask_for_logits(logits)
                            logits = self._compose_policy_logits_with_masks(
                                logits,
                                mask,
                                mask_logits_pred,
                            )
                            if eval_policy_norm == "greedy":
                                action = torch.argmax(logits, dim=-1)
                            else:
                                dist = Categorical(logits=logits)
                                action = dist.sample()

                    next_obs, _, done, info = self.env.step(action.item())
                    reward_env = self.compute_preference_reward(info)
                    total_r += reward_env
                    t += 1

                    if info.get("got_food", False):
                        food_count += 1
                    if info.get("took_damage", False):
                        damage_count += 1
                        episode_had_damage = True
                        episode_had_violation = True

                    death_flag_raw = info.get("death_flag")
                    if isinstance(death_flag_raw, (int, float)) and float(death_flag_raw) > 0.0:
                        episode_had_death = True

                    alive_raw = info.get("alive")
                    if isinstance(alive_raw, bool):
                        if not alive_raw:
                            episode_had_death = True
                    elif isinstance(alive_raw, (int, float)):
                        if float(alive_raw) <= 0.0:
                            episode_had_death = True

                    patch = next_obs["patch"]
                    energy = next_obs["energy"]
                    scenario_id = int(
                        next_obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id))
                    )
                    scenario_name = next_obs.get(
                        "scenario_name",
                        getattr(self.env, "current_scenario_name", scenario_name),
                    )
                    env_id = int(next_obs.get("env_id", env_id))
                    env_name = next_obs.get("env_name", env_name)

                    with torch.no_grad():
                        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                        H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                        scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        env_desc_t = _get_env_desc(env_t)
                        if env_desc_t is not None:
                            env_desc_np = env_desc_t[0].detach().cpu().numpy().reshape(-1).tolist()
                        text_t = self._text_tokens_from_ids(env_t, scenario_t)
                        z_obs_next = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                        _, h_w, _, _ = self.agent.world_model.forward_step(z_obs_next, H_t, action, h_w)

                    last_reward = reward_env
                    last_action = action

                if not done and t >= max_steps:
                    timeout_episodes += 1

                if isinstance(info, dict) and "last_test_passed" in info:
                    passed_flag = bool(info.get("last_test_passed"))
                    repo_pass_flags.append(passed_flag)
                    if passed_flag:
                        repo_steps_to_pass.append(int(info.get("steps_taken", t)))

                reason_raw = ""
                if isinstance(info, dict):
                    reason_raw = str(info.get("reason", "") or "").strip()
                    if not reason_raw:
                        status_raw = str(info.get("status", "") or "").strip().lower()
                        if status_raw == "timeout":
                            reason_raw = "pytest_timeout"
                if not reason_raw:
                    if not done and t >= max_steps:
                        reason_raw = "eval_max_steps_cap"
                    elif done:
                        reason_raw = "done"
                    else:
                        reason_raw = "unknown"
                reason_counts[reason_raw] = reason_counts.get(reason_raw, 0) + 1

                reason_norm = reason_raw.lower()
                if reason_norm in {"terminated_danger", "pytest_timeout"}:
                    episode_had_violation = True
                    episode_had_catastrophic = True
                if episode_had_death and episode_had_damage:
                    episode_had_catastrophic = True

                if episode_had_damage:
                    damage_episodes += 1
                if episode_had_death:
                    death_episodes += 1
                if episode_had_violation:
                    constraint_violation_episodes += 1
                if episode_had_catastrophic:
                    catastrophic_fail_episodes += 1

                returns.append(total_r)
                lengths.append(t)
                foods.append(food_count)
                damages.append(damage_count)

                if self.logger is not None:
                    self.logger.log_scalar(
                        stage=f"eval_{'self' if split_use_self else 'no_self'}",
                        metric="episode_return",
                        value=float(total_r),
                        env_id=int(env_id),
                        env_name=str(env_name),
                        scenario_id=int(scenario_id),
                        scenario_name=str(scenario_name),
                        env_descriptor=env_desc_np,
                    )

                split = _classify_env(env_name, env_id)
                if split == "train":
                    train_returns.append(total_r)
                elif split == "test":
                    test_returns.append(total_r)

                if isinstance(info, dict) and "last_test_passed" in info:
                    passed_flag = bool(info.get("last_test_passed"))
                    if split == "train":
                        repo_pass_flags_train.append(passed_flag)
                        if passed_flag:
                            repo_steps_to_pass_train.append(int(info.get("steps_taken", t)))
                    elif split == "test":
                        repo_pass_flags_test.append(passed_flag)
                        if passed_flag:
                            repo_steps_to_pass_test.append(int(info.get("steps_taken", t)))

                if self.is_minigrid:
                    task_key = scenario_name or env_name
                    per_task_returns.setdefault(task_key, []).append(float(total_r))

                family = str(info.get("env_family", "")).lower() if isinstance(info, dict) else ""
                if not family:
                    env_name_l = str(env_name).lower()
                    if "instruction" in env_name_l:
                        family = "instruction-basic"
                    elif "social" in env_name_l:
                        family = "social-basic"
                success_flag = self._infer_episode_success_from_info(info)
                if success_flag is not None:
                    if "instruction" in family:
                        instruction_success_flags.append(bool(success_flag))
                        if split == "train":
                            instruction_success_flags_train.append(bool(success_flag))
                        elif split == "test":
                            instruction_success_flags_test.append(bool(success_flag))
                    elif "social" in family:
                        social_success_flags.append(bool(success_flag))
                        if split == "train":
                            social_success_flags_train.append(bool(success_flag))
                        elif split == "test":
                            social_success_flags_test.append(bool(success_flag))

            mean_ret = float(np.mean(returns)) if returns else float("nan")
            std_ret = float(np.std(returns)) if returns else float("nan")
            mean_len = float(np.mean(lengths)) if lengths else float("nan")
            mean_food = float(np.mean(foods)) if foods else 0.0
            mean_damage = float(np.mean(damages)) if damages else 0.0
            denom_eps = float(max(1, int(n_episodes)))
            death_rate = float(death_episodes) / denom_eps
            damage_episode_rate = float(damage_episodes) / denom_eps
            catastrophic_fail_rate = float(catastrophic_fail_episodes) / denom_eps
            constraint_compliance = float(
                max(0.0, min(1.0, 1.0 - (float(constraint_violation_episodes) / denom_eps)))
            )
            timeout_rate = float(timeout_episodes) / denom_eps

            print(
                f"Eval[{mask_label}] (use_self={split_use_self}, planning_coef={planning_coef:.2f}): "
                f"mean return = {mean_ret:.3f} ± {std_ret:.3f}, "
                f"mean length = {mean_len:.1f}, "
                f"mean food = {mean_food:.2f}, "
                f"mean damage = {mean_damage:.2f}"
            )

            print(
                f"  Safety: compliance={constraint_compliance:.3f}, "
                f"catastrophic_fail_rate={catastrophic_fail_rate:.3f}, death_rate={death_rate:.3f}"
            )

            if scenario_counts:
                print("  Scenario usage:")
                for name, cnt in scenario_counts.items():
                    print(f"    {name}: {cnt} episodes")

            if env_counts:
                print("  Env usage:")
                for name, cnt in env_counts.items():
                    print(f"    {name}: {cnt} episodes")

            if repo_pass_flags:
                pass_rate = float(np.mean(repo_pass_flags))
                mean_steps_pass = float(np.mean(repo_steps_to_pass)) if repo_steps_to_pass else float("nan")
                print(f"  Repo pass rate: {pass_rate:.3f}, mean steps to pass: {mean_steps_pass:.1f}")
                if repo_pass_flags_train:
                    pr = float(np.mean(repo_pass_flags_train))
                    ms = float(np.mean(repo_steps_to_pass_train)) if repo_steps_to_pass_train else float("nan")
                    print(f"  Repo train pass rate: {pr:.3f}, mean steps to pass: {ms:.1f}")
                if repo_pass_flags_test:
                    pr = float(np.mean(repo_pass_flags_test))
                    ms = float(np.mean(repo_steps_to_pass_test)) if repo_steps_to_pass_test else float('nan')
                    print(f"  Repo test pass rate:  {pr:.3f}, mean steps to pass: {ms:.1f}")

            train_mean = float(np.mean(train_returns)) if train_returns else float("nan")
            train_std = float(np.std(train_returns)) if train_returns else float("nan")
            test_mean = float(np.mean(test_returns)) if test_returns else float("nan")
            test_std = float(np.std(test_returns)) if test_returns else float("nan")

            if train_returns:
                print(
                    f"  Train envs mean return (use_self={split_use_self}): "
                    f"{train_mean:.3f} ± {train_std:.3f}"
                )
            if test_returns:
                print(
                    f"  Test envs mean return (use_self={split_use_self}):  "
                    f"{test_mean:.3f} ± {test_std:.3f}"
                )

            per_task_stats: Dict[str, Any] = {}
            if self.is_minigrid and per_task_returns:
                for task_name, vals in per_task_returns.items():
                    if not vals:
                        continue
                    m = float(statistics.mean(vals)) if vals else float("nan")
                    s = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
                    per_task_stats[task_name] = {
                        "returns": [float(x) for x in vals],
                        "mean_return": m,
                        "std_return": s,
                        "n_episodes": len(vals),
                    }

            metrics = {
                "use_self": bool(split_use_self),
                "planning_coef": float(planning_coef),
                "n_episodes": int(n_episodes),
                "max_steps": int(max_steps),
                "timeout_episodes": int(timeout_episodes),
                "timeout_rate": timeout_rate,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "mean_length": mean_len,
                "mean_food": mean_food,
                "mean_damage": mean_damage,
                "damage_episode_rate": damage_episode_rate,
                "death_rate": death_rate,
                "catastrophic_fail_rate": catastrophic_fail_rate,
                "constraint_compliance": constraint_compliance,
                "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
                "scenario_counts": scenario_counts,
                "env_counts": env_counts,
                "returns": [float(x) for x in returns],
                "lengths": [int(x) for x in lengths],
                "foods": [int(x) for x in foods],
                "damages": [int(x) for x in damages],
                "train_mean_return": train_mean,
                "train_std_return": train_std,
                "test_mean_return": test_mean,
                "test_std_return": test_std,
            }
            planner_summary = self._planner_debug_summary(
                alpha_values=planner_alpha_vals,
                js_values=planner_js_vals,
                margin_values=planner_margin_vals,
                override_values=planner_override_vals,
            )
            metrics.update(planner_summary)
            if per_task_stats:
                metrics["per_task"] = per_task_stats
            if repo_pass_flags:
                metrics["repo_pass_rate"] = float(np.mean(repo_pass_flags))
                metrics["repo_steps_to_pass"] = [int(x) for x in repo_steps_to_pass]
                if repo_pass_flags_train:
                    metrics["repo_train_pass_rate"] = float(np.mean(repo_pass_flags_train))
                    metrics["repo_train_steps_to_pass"] = [int(x) for x in repo_steps_to_pass_train]
                if repo_pass_flags_test:
                    metrics["repo_test_pass_rate"] = float(np.mean(repo_pass_flags_test))
                    metrics["repo_test_steps_to_pass"] = [int(x) for x in repo_steps_to_pass_test]
            if instruction_success_flags:
                metrics["instruction_success_rate"] = float(np.mean(instruction_success_flags))
                if instruction_success_flags_train:
                    metrics["instruction_train_success_rate"] = float(np.mean(instruction_success_flags_train))
                if instruction_success_flags_test:
                    metrics["instruction_test_success_rate"] = float(np.mean(instruction_success_flags_test))
            if social_success_flags:
                metrics["social_success_rate"] = float(np.mean(social_success_flags))
                if social_success_flags_train:
                    metrics["social_train_success_rate"] = float(np.mean(social_success_flags_train))
                if social_success_flags_test:
                    metrics["social_test_success_rate"] = float(np.mean(social_success_flags_test))
            return metrics

        # Primary eval: respect/enable the environment's action-mask UI (if present).
        _set_mask_enabled(True)
        results = _run_eval(bool(use_self), mask_label="masked" if mask_supported else "default")

        # Secondary eval: measure "internalization" without mask (robustness).
        if mask_supported:
            _set_mask_enabled(False)
            unmasked = _run_eval(bool(use_self), mask_label="unmasked")
            results["unmasked"] = unmasked
            for key in (
                "mean_return",
                "std_return",
                "mean_length",
                "train_mean_return",
                "test_mean_return",
                "repo_pass_rate",
                "repo_train_pass_rate",
                "repo_test_pass_rate",
                "instruction_success_rate",
                "instruction_train_success_rate",
                "instruction_test_success_rate",
                "social_success_rate",
                "social_train_success_rate",
                "social_test_success_rate",
                "timeout_rate",
                "damage_episode_rate",
                "death_rate",
                "catastrophic_fail_rate",
                "constraint_compliance",
                "reason_counts",
            ):
                if key in unmasked:
                    results[f"unmasked_{key}"] = unmasked[key]

        # Restore prior mask state if we could read it; otherwise default to enabled.
        if prev_mask_enabled is not None:
            _set_mask_enabled(prev_mask_enabled)
        else:
            _set_mask_enabled(True)
        return results

    # =========================
    #  Online Phase C adaptation (Phase D)
    # =========================
    def run_online_phaseC_adaptation(
        self,
        env,
        n_episodes: int = 50,
        use_self: bool = True,
        planning_coef: float = 0.5,
        phase_label: str = "phaseC_online",
        allow_reflection: bool = True,
        lr: float = 5e-2,
        lambda_reg: float = 1e-2,
        lambda_conflict: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Online adaptation on test envs: traits can update between episodes via self-reflection.
        Policy/world-model/SelfModel weights stay frozen.
        """
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.eval()
        self.agent.workspace.eval()
        self.agent.policy.eval()
        self.agent.value_model.eval()

        alpha_conf = 0.2
        alpha_unc = 0.2

        traits_before = self.agent.traits.detach().cpu().numpy().tolist()
        base_traits_main = self._main_traits().detach().clone()
        self._start_trait_safety_phase("phaseC", base_traits_main)
        base_weights_phaseC = get_faction_preference_weights(self.agent)
        w_survive_base_phaseC = float(base_weights_phaseC[0, 0].item()) if base_weights_phaseC.numel() > 0 else 0.0
        w_danger_base_phaseC = float(base_weights_phaseC[0, 2].item()) if base_weights_phaseC.numel() > 0 else 0.0

        returns: List[float] = []
        lengths: List[int] = []
        foods: List[int] = []
        damages: List[int] = []
        episode_returns: List[float] = []
        planner_alpha_vals: List[float] = []
        planner_js_vals: List[float] = []
        planner_margin_vals: List[float] = []
        planner_override_vals: List[float] = []
        reflection_step_norms: List[float] = []
        reflection_steps_total = 0
        step_size_reflect_base = float(lr)
        if self.is_minigrid:
            step_size_reflect_base = float(self.trait_reflection_lr)
        n_steps_reflect_base = self.trait_reflection_steps_per_batch if not self.is_minigrid else max(
            self.trait_reflection_steps_per_batch, self.trait_reflection_steps_per_batch
        )

        for _ in range(n_episodes):
            obs = env.reset()
            patch = obs["patch"]
            energy = obs["energy"]
            scenario_id = int(obs.get("scenario_id", getattr(env, "current_scenario_id", 0)))
            env_id = int(obs.get("env_id", getattr(env, "env_id", 0)))

            h_w = torch.zeros(1, 1, self.agent.world_model.gru.hidden_size, device=self.device)
            h_s = torch.zeros(1, 1, self.agent.self_model.gru.hidden_size, device=self.device)
            last_action = torch.zeros(1, dtype=torch.long, device=self.device)
            last_reward = 0.0
            traits = self._mixed_traits()
            M = self.agent.memory

            total_r = 0.0
            t = 0
            done = False
            food_count = 0
            damage_count = 0
            trajectory = []

            while not done and t < env.max_steps:
                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)

                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    W_t = h_w.squeeze(0)

                    if use_self:
                        r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                        a_emb = self.agent.self_model.act_emb(last_action)
                        env_emb = self.agent.self_model.env_desc_to_emb(env_desc_t)

                        W_in = W_t.unsqueeze(1)
                        H_in = H_t.unsqueeze(1)
                        r_in = r_t.unsqueeze(1)
                        M_b = M.unsqueeze(1)
                        x_s = torch.cat([W_in, H_in, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)], dim=-1)
                        out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                        S_t = out_s.squeeze(1)

                        surv_t = torch.sigmoid(self.agent.self_model.head_survival(S_t))
                        food_t = self.agent.self_model.head_food(S_t)
                        dmg_t = self.agent.self_model.head_damage(S_t)
                        move_t = self.agent.self_model.head_move(S_t)

                        w = traits_to_preference_weights(traits)
                        w_survive, w_food, w_danger, w_move = w[0]
                        R_unscaled = (
                            w_survive * surv_t
                            + w_food * food_t
                            + w_danger * dmg_t
                            + w_move * move_t
                        ).view(1, 1)
                        R_self = self.agent.self_model.head_return_calib(R_unscaled)
                        U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
                    else:
                        S_t = torch.zeros(1, self.agent.self_model.gru.hidden_size, device=self.device)
                        R_self = torch.zeros(1, 1, device=self.device)
                        U_t = torch.zeros(1, 1, device=self.device)

                    V_pi = self.agent.value_model(W_t, H_t, traits, M)

                    if use_self:
                        conf_t = torch.abs(R_self - V_pi)
                        gate = torch.exp(-alpha_conf * conf_t - alpha_unc * U_t)
                        delta_raw = torch.tanh(R_self - V_pi)
                        delta_self = delta_raw * gate
                    else:
                        delta_self = torch.zeros_like(R_self)

                    G_t = self.agent.workspace(
                        W_t,
                        S_t,
                        H_t,
                        V_pi,
                        delta_self,
                        U_t,
                        traits,
                        M,
                    )
                    logits = self.agent.policy(G_t)

                    if planning_coef > 0.0:
                        planner_logits = self._get_planner_logits(
                            z_obs=z_obs,
                            H_t=H_t,
                            h_w=h_w,
                            traits=traits,
                            M=M,
                        )
                        logits, planner_debug = self._blend_with_planner(
                            policy_logits=logits,
                            planner_logits=planner_logits,
                            base_planning_coef=float(planning_coef),
                            uncertainty=U_t if use_self else None,
                            r_self=R_self if use_self else None,
                            v_pi=V_pi if use_self else None,
                        )
                        planner_alpha_vals.append(float(planner_debug.get("planner_alpha", 0.0)))
                        planner_js_vals.append(float(planner_debug.get("planner_js", 0.0)))
                        planner_margin_vals.append(float(planner_debug.get("planner_margin", 0.0)))
                        planner_override_vals.append(float(planner_debug.get("planner_override", 0.0)))

                    logits = self._apply_action_mask(logits)
                    dist = Categorical(logits=logits)
                    action = dist.sample()

                next_obs, _, done, info = env.step(action.item())
                reward_env = self.compute_preference_reward(info)
                total_r += reward_env
                t += 1

                if info.get("got_food", False):
                    food_count += 1
                if info.get("took_damage", False):
                    damage_count += 1

                trajectory.append(
                    {
                        "obs_patch": patch.copy(),
                        "energy": float(energy),
                        "action": int(action.item()),
                        "reward": float(reward_env),
                        "done": done,
                        "next_obs_patch": next_obs["patch"].copy(),
                        "next_energy": float(next_obs["energy"]),
                        "death_flag": float(info.get("death_flag", 0.0)),
                        "got_food": float(info.get("got_food", False)),
                        "took_damage": float(info.get("took_damage", False)),
                        "moved": float(info.get("moved", False)),
                        "alive": float(info.get("alive", True)),
                        "scenario_id": int(info.get("scenario_id", scenario_id)),
                        "env_id": int(info.get("env_id", env_id)),
                    }
                )

                patch = next_obs["patch"]
                energy = next_obs["energy"]
                scenario_id = int(next_obs.get("scenario_id", getattr(env, "current_scenario_id", scenario_id)))
                env_id = int(next_obs.get("env_id", env_id))

                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                    H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                    scenario_t = torch.tensor([scenario_id], dtype=torch.long, device=self.device)
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)
                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs_next = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    _, h_w, _, _ = self.agent.world_model.forward_step(z_obs_next, H_t, action, h_w)

                last_reward = reward_env
                last_action = action

            returns.append(total_r)
            lengths.append(t)
            foods.append(food_count)
            damages.append(damage_count)
            episode_returns.append(total_r)

            if use_self and allow_reflection:
                obs_seq = np.stack([tr["obs_patch"] for tr in trajectory], axis=0)[None, ...]
                H_seq = np.array([[tr["energy"]] for tr in trajectory], dtype=np.float32)[None, ...]
                a_seq = np.array([tr["action"] for tr in trajectory], dtype=np.int64)[None, ...]
                r_seq = np.array([tr["reward"] for tr in trajectory], dtype=np.float32)[None, ...]
                d_seq = np.array([float(tr["done"]) for tr in trajectory], dtype=np.float32)[None, ...]
                death_seq = np.array([tr["death_flag"] for tr in trajectory], dtype=np.float32)[None, ...]
                food_seq = np.array([tr["got_food"] for tr in trajectory], dtype=np.float32)[None, ...]
                dmg_seq = np.array([tr["took_damage"] for tr in trajectory], dtype=np.float32)[None, ...]
                move_seq = np.array([tr["moved"] for tr in trajectory], dtype=np.float32)[None, ...]
                alive_seq = np.array([tr["alive"] for tr in trajectory], dtype=np.float32)[None, ...]
                scenario_seq = np.array([tr["scenario_id"] for tr in trajectory], dtype=np.int64)[None, ...]
                env_seq = np.array([tr["env_id"] for tr in trajectory], dtype=np.int64)[None, ...]

                anchor_traits = self._main_traits().detach().clone()
                step_size_reflect = step_size_reflect_base
                n_steps_reflect = n_steps_reflect_base
                info_reflect = self._run_trait_reflection(
                    batch=(
                        obs_seq,
                        H_seq,
                        a_seq,
                        r_seq,
                        d_seq,
                        death_seq,
                        food_seq,
                        dmg_seq,
                        move_seq,
                        alive_seq,
                        scenario_seq,
                        env_seq,
                    ),
                    reward_profile=None,
                    step_size=step_size_reflect,
                    regime_name=phase_label,
                    lambda_l2=lambda_reg,
                    n_steps=n_steps_reflect,
                    trait_anchor=anchor_traits,
                    init_traits=anchor_traits,
                    safety_phase="phaseC",
                    safety_baseline=base_traits_main,
                )
                reflection_step_norms.extend(info_reflect.get("step_norms", []))
                reflection_steps_total += int(info_reflect.get("steps", 0))
                updated = bool(info_reflect.get("updated", False))
                steps = int(info_reflect.get("steps", 0))
                step_norms = info_reflect.get("step_norms", []) or []
                stats = self.trait_reflection_debug.setdefault(
                    phase_label,
                    {
                        "trait_reflection_n_calls": 0,
                        "trait_reflection_n_updates": 0,
                        "trait_reflection_total_steps": 0,
                        "trait_reflection_step_norm_sum": 0.0,
                    },
                )
                stats["trait_reflection_n_calls"] += 1
                stats["trait_reflection_total_steps"] += steps
                if updated:
                    stats["trait_reflection_n_updates"] += 1
                    stats["trait_reflection_step_norm_sum"] += float(sum(step_norms))

        mean_ret = float(np.mean(returns)) if returns else 0.0
        std_ret = float(np.std(returns)) if returns else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        mean_food = float(np.mean(foods)) if foods else 0.0
        mean_damage = float(np.mean(damages)) if damages else 0.0
        traits_after = self.agent.traits.detach().cpu().numpy().tolist()
        mean_step_norm = float(np.mean(reflection_step_norms)) if reflection_step_norms else 0.0
        online_debug = {
            "steps": int(reflection_steps_total),
            "mean_step_norm": mean_step_norm,
        }
        planner_summary = self._planner_debug_summary(
            alpha_values=planner_alpha_vals,
            js_values=planner_js_vals,
            margin_values=planner_margin_vals,
            override_values=planner_override_vals,
        )
        self.trait_reflection_debug[f"online_{phase_label}"] = online_debug
        self.apply_safe_trait_update(self.agent, self._main_traits().detach().clone(), phase="phaseC", finalize=True)
        weights_after = get_faction_preference_weights(self.agent)
        w_survive_after = float(weights_after[0, 0].item()) if weights_after.numel() > 0 else 0.0
        w_danger_after = float(weights_after[0, 2].item()) if weights_after.numel() > 0 else 0.0
        delta_main = float(torch.norm(self._main_traits() - base_traits_main).item())
        if self.logger is not None:
            self.logger.log(
                {
                    "event": "traits_reflection_summary",
                    "phase": phase_label,
                    "traits_main_before": base_traits_main.cpu().tolist(),
                    "traits_main_after": self._main_traits().detach().cpu().tolist(),
                    "w_survive_before": w_survive_base_phaseC,
                    "w_survive_after": w_survive_after,
                    "w_danger_before": w_danger_base_phaseC,
                    "w_danger_after": w_danger_after,
                    "delta_norm": delta_main,
                }
            )
        self._trait_safety_ctx.pop("phaseC", None)

        return {
            "phase": phase_label,
            "use_self": use_self,
            "planning_coef": planning_coef,
            "n_episodes": n_episodes,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "mean_length": mean_len,
            "mean_food": mean_food,
            "mean_damage": mean_damage,
            "episode_returns": [float(x) for x in episode_returns],
            "traits_before": traits_before,
            "traits_after": traits_after,
            "trait_reflection_debug": online_debug,
            **planner_summary,
        }

    # =========================
    #  Lifelong regimes (Phase D+)
    # =========================

    def _sample_mixed_sequences(
        self,
        new_buffer: ReplayBuffer,
        old_buffer: Optional[ReplayBuffer],
        batch_size: int,
        seq_len: int,
        replay_ratio_old: float = 0.5,
        with_events: bool = False,
    ) -> Optional[Tuple[np.ndarray, ...]]:
        n_old = int(batch_size * replay_ratio_old)
        n_new = batch_size - n_old

        def _can(buf: Optional[ReplayBuffer], count: int) -> bool:
            return buf is not None and count > 0 and len(buf) >= seq_len

        if not _can(new_buffer, n_new):
            n_old += n_new
            n_new = 0
        if not _can(old_buffer, n_old):
            n_new += n_old
            n_old = 0

        samples: List[Tuple[np.ndarray, ...]] = []
        if _can(new_buffer, n_new):
            fn_new = new_buffer.sample_sequences_with_events if with_events else new_buffer.sample_sequences
            samples.append(fn_new(n_new, seq_len))
        if _can(old_buffer, n_old):
            past_regime_weights = self._compute_lifelong_past_regime_weights()
            mix_cfg = {
                "current_regime": str(self.current_regime_name or ""),
                "frac_current": 0.0,
                "past_regime_weights": past_regime_weights,
                "sampling_temperature": 1.0,
            }
            samples.append(
                old_buffer.sample_mixed(
                    batch_size=n_old,
                    seq_len=seq_len,
                    mix_config=mix_cfg,
                    with_events=with_events,
                )
            )

        if not samples:
            return None

        merged: List[np.ndarray] = []
        n_fields = len(samples[0])
        for idx in range(n_fields):
            merged.append(np.concatenate([s[idx] for s in samples], axis=0))
        return tuple(merged)

    def _policy_update_from_sequences(
        self,
        batch: Tuple[np.ndarray, ...],
        use_self: bool,
        gamma: float,
        entropy_coef: float,
        beta_conflict: float,
        beta_uncertainty: float,
        regularization_coef: float,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Optional[Dict[str, float]]:
        (
            obs_seq,
            H_seq,
            a_seq,
            r_seq,
            d_seq,
            death_seq,
            scenario_seq,
            env_seq,
        ) = batch

        B, T, p, _ = obs_seq.shape
        if B == 0 or T == 0:
            return None

        patch = torch.from_numpy(obs_seq).long().to(self.device)
        H = torch.from_numpy(H_seq).float().to(self.device)
        actions = torch.from_numpy(a_seq).long().to(self.device)
        rewards = torch.from_numpy(r_seq).float().to(self.device)
        dones = torch.from_numpy(d_seq).float().to(self.device)
        scenario = torch.from_numpy(scenario_seq).long().to(self.device)
        env_ids = torch.from_numpy(env_seq).long().to(self.device)
        env_desc_seq = self._env_desc_from_ids(env_ids)

        with torch.no_grad():
            patch_flat = patch.reshape(B * T, p, p)
            H_flat = H.reshape(B * T, 1)
            scenario_flat = scenario.reshape(B * T)
            env_desc_flat = env_desc_seq.reshape(B * T, -1) if env_desc_seq is not None else None

            env_flat = env_ids.reshape(B * T)
            text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
            z_flat = self.agent.perception(
                patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
            )
            z_seq = z_flat.reshape(B, T, -1)

            a_emb = self.agent.world_model.act_emb(actions)
            x_w = torch.cat([z_seq, H, a_emb], dim=-1)
            h0 = torch.zeros(1, B, self.agent.world_model.gru.hidden_size, device=self.device)
            W_seq, _ = self.agent.world_model.gru(x_w, h0)

        traits = self._mixed_traits()
        M = self.agent.memory

        if use_self:
            (
                S_seq,
                _,
                surv_pred,
                food_pred,
                dmg_pred,
                move_pred,
                unc_pred,
                _,
            ) = self.agent.self_model.forward_seq(
                W_seq.detach(),
                H,
                actions,
                rewards,
                M=M,
                env_desc=env_desc_seq,
            )
            w_eff = traits_to_preference_weights(traits)
            w_survive, w_food, w_danger, w_move = w_eff[0]
            R_unscaled = (
                w_survive * surv_pred
                + w_food * food_pred
                + w_danger * dmg_pred
                + w_move * move_pred
            )
            R_self = self.agent.self_model.head_return_calib(R_unscaled)
            U_seq = torch.abs(unc_pred)
        else:
            S_seq = torch.zeros(
                B,
                T,
                self.agent.self_model.gru.hidden_size,
                device=self.device,
            )
            R_self = torch.zeros(B, T, 1, device=self.device)
            U_seq = torch.zeros_like(R_self)

        W_flat = W_seq.detach().reshape(B * T, -1)
        H_flat = H.reshape(B * T, 1)
        V_flat = self.agent.value_model(W_flat, H_flat, traits, M)
        V_seq = V_flat.reshape(B, T, 1)

        alpha_conf = 0.5
        alpha_unc = 0.5
        max_delta_amp = 0.2
        if use_self:
            conf_seq = torch.abs(R_self - V_seq.detach())
            gate = torch.exp(-alpha_conf * (conf_seq ** 2) - alpha_unc * (U_seq ** 2))
            delta_raw = torch.tanh(R_self - V_seq.detach())
            delta_self = torch.clamp(delta_raw * gate, -max_delta_amp, max_delta_amp)
        else:
            conf_seq = torch.zeros_like(R_self)
            delta_self = torch.zeros_like(R_self)

        G_flat = self.agent.workspace(
            W_flat,
            S_seq.reshape(B * T, -1),
            H_flat,
            V_flat,
            delta_self.reshape(B * T, 1),
            U_seq.reshape(B * T, 1),
            traits,
            M,
        )

        logits = self.agent.policy(G_flat)
        dist = Categorical(logits=logits)
        actions_flat = actions.reshape(-1)
        logprobs = dist.log_prob(actions_flat)
        entropies = dist.entropy()

        rewards_flat = rewards.reshape(-1)
        dones_flat = dones.reshape(-1)
        values_flat = V_flat.reshape(-1)
        conflicts_flat = conf_seq.reshape(-1)
        uncertainties_flat = U_seq.reshape(-1)

        if use_self and conflicts_flat.numel() > 0:
            self._update_meta_stats(conflicts_flat, uncertainties_flat)

        return self._run_a2c_update(
            rewards=rewards_flat,
            dones=dones_flat,
            values=values_flat,
            logprobs=logprobs,
            entropies=entropies,
            conflicts=conflicts_flat,
            uncertainties=uncertainties_flat,
            gamma=gamma,
            entropy_coef=entropy_coef,
            beta_conflict=beta_conflict,
            beta_uncertainty=beta_uncertainty,
            regularization_coef=regularization_coef,
            optimizer=optimizer,
        )

    def _self_model_update_from_batch(
        self,
        batch: Tuple[np.ndarray, ...],
        gamma: float,
        regularization_coef: float,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Optional[float]:
        (
            obs_seq,
            H_seq,
            a_seq,
            r_seq,
            d_seq,
            death_seq,
            food_seq,
            dmg_seq,
            move_seq,
            alive_seq,
            scenario_seq,
            env_seq,
        ) = batch

        B, T, p, _ = obs_seq.shape
        if B == 0 or T == 0:
            return None

        patch = torch.from_numpy(obs_seq).long().to(self.device)
        H = torch.from_numpy(H_seq).float().to(self.device)
        a = torch.from_numpy(a_seq).long().to(self.device)
        r = torch.from_numpy(r_seq).float().to(self.device)
        d = torch.from_numpy(d_seq).float().to(self.device)
        death = torch.from_numpy(death_seq).float().to(self.device)
        food = torch.from_numpy(food_seq).float().to(self.device)
        dmg = torch.from_numpy(dmg_seq).float().to(self.device)
        move = torch.from_numpy(move_seq).float().to(self.device)
        alive = torch.from_numpy(alive_seq).float().to(self.device)
        scenario = torch.from_numpy(scenario_seq).long().to(self.device)
        env_ids = torch.from_numpy(env_seq).long().to(self.device)
        env_desc_seq = self._env_desc_from_ids(env_ids)

        with torch.no_grad():
            patch_flat = patch.reshape(B * T, p, p)
            H_flat = H.reshape(B * T, 1)
            scenario_flat = scenario.reshape(B * T)
            env_desc_flat = env_desc_seq.reshape(B * T, -1) if env_desc_seq is not None else None

            env_flat = env_ids.reshape(B * T)
            text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
            z_flat = self.agent.perception(
                patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
            )
            z_seq = z_flat.reshape(B, T, -1)

            a_emb = self.agent.world_model.act_emb(a)
            x_w = torch.cat([z_seq, H, a_emb], dim=-1)
            h0 = torch.zeros(1, B, self.agent.world_model.gru.hidden_size, device=self.device)
            W_seq, _ = self.agent.world_model.gru(x_w, h0)

        done_mask = d > 0.5
        done_any = done_mask.any(dim=1)
        done_idx = torch.argmax(done_mask.int(), dim=1)
        death_at_done = torch.gather(death, 1, done_idx.unsqueeze(1)).squeeze(1)
        alive_final = alive[:, -1] if alive.numel() > 0 else torch.ones_like(death_at_done)
        survived_flag = torch.where(done_any, 1.0 - torch.clamp(death_at_done, 0.0, 1.0), alive_final)
        S_alive = survived_flag.view(B, 1).expand(-1, T)

        def discounted_sums(x: torch.Tensor, done_mask: torch.Tensor, gamma: float) -> torch.Tensor:
            B_, T_ = x.shape
            idx = torch.arange(T_, device=x.device)
            gamma_mat = torch.triu(gamma ** (idx[None, :] - idx[:, None]))
            not_done = 1.0 - done_mask
            prefix = torch.cumprod(torch.cat([torch.ones(B_, 1, device=x.device, dtype=x.dtype), not_done], dim=1), dim=1)
            prefix_t = prefix[:, :-1]
            prefix_k = prefix[:, 1:]
            prefix_t_safe = torch.clamp(prefix_t, min=1e-8)
            ratio = prefix_k.unsqueeze(1) / prefix_t_safe.unsqueeze(2)
            return (gamma_mat.unsqueeze(0) * x.unsqueeze(1) * ratio).sum(dim=2)

        G_food = discounted_sums(food.squeeze(-1), d, gamma)
        G_dmg = discounted_sums(dmg.squeeze(-1), d, gamma)
        G_move = discounted_sums(move.squeeze(-1), d, gamma)
        G_ret = discounted_sums(r.squeeze(-1), d, gamma)

        surv_target = S_alive.unsqueeze(-1)
        food_target = G_food.unsqueeze(-1)
        dmg_target = G_dmg.unsqueeze(-1)
        move_target = G_move.unsqueeze(-1)
        ret_target = G_ret.unsqueeze(-1)

        M = self.agent.memory

        (
            S_seq,
            S_last,
            surv_pred,
            food_pred,
            dmg_pred,
            move_pred,
            unc_pred,
            surv_raw_pred,
        ) = self.agent.self_model.forward_seq(
            W_seq,
            H,
            a,
            r,
            M=M,
            env_desc=env_desc_seq,
        )

        loss_surv_main = F.binary_cross_entropy(surv_pred, surv_target)
        loss_surv_raw = F.binary_cross_entropy(surv_raw_pred, surv_target)
        loss_surv = 1.5 * loss_surv_main + 0.5 * loss_surv_raw
        loss_food = F.mse_loss(food_pred, food_target)
        loss_dmg = F.mse_loss(dmg_pred, dmg_target)
        loss_move = F.mse_loss(move_pred, move_target)

        with torch.no_grad():
            w = self._combined_preference_weights().detach()
            w_survive, w_food, w_danger, w_move = w.view(-1)
        R_unscaled_pred = (
            w_survive * surv_pred
            + w_food * food_pred
            + w_danger * dmg_pred
            + w_move * move_pred
        )
        R_calib_pred = self.agent.self_model.head_return_calib(R_unscaled_pred)
        loss_ret = F.mse_loss(R_calib_pred, ret_target)

        with torch.no_grad():
            err_food = torch.abs(food_target - food_pred)
            err_dmg = torch.abs(dmg_target - dmg_pred)
            err_move = torch.abs(move_target - move_pred)
            err_surv = torch.abs(surv_target - surv_pred)
            err_total = err_food + err_dmg + err_move + err_surv
            unc_target = err_total

        loss_unc = F.mse_loss(unc_pred, unc_target)

        w_surv = 1.5
        w_food = 1.0
        w_dmg = 1.0
        w_move = 1.0
        w_unc = 0.2
        w_ret = 1.0

        loss = (
            w_surv * loss_surv
            + w_food * loss_food
            + w_dmg * loss_dmg
            + w_move * loss_move
            + w_unc * loss_unc
            + w_ret * loss_ret
        )
        if regularization_coef > 0.0:
            loss = loss + self._fast_param_l2_penalty(regularization_coef)

        opt = optimizer or self.lifelong_optimizer
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.self_model.parameters(), 1.0)
        opt.step()

        return float(loss.detach().item())

    def run_lifelong_train(
        self,
        episodes_per_chapter: int = 50,
        planning_coef: float = 0.5,
        agent_variant: str = "full",
        regime_configs: Optional[List[RegimeConfig]] = None,
        allow_online_reflection: bool = True,
        allow_online_model_updates: bool = True,
        lr_policy: float = 1e-4,
        lr_models: float = 1e-4,
        replay_ratio_old: float = 0.5,
        regularization_coef: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Lifelong training mode: update traits plus fast parameters (policy/value/workspace/self-model)
        between regime chapters while replaying past experience and anchoring to initial fast params.
        """
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.train()
        self.agent.workspace.train()
        self.agent.policy.train()
        self.agent.value_model.train()

        use_self_flag = agent_variant != "no_self"
        allow_trait_updates = agent_variant == "full" and use_self_flag
        reflect_enabled = allow_online_reflection and allow_trait_updates
        entropy_coef = 0.003

        if self.lifelong_optimizer is None:
            self.lifelong_optimizer = torch.optim.Adam(self.agent.get_fast_params(), lr=lr_policy)
        if len(self.lifelong_optimizer.param_groups) >= 2:
            self.lifelong_optimizer.param_groups[0]["lr"] = lr_policy
            self.lifelong_optimizer.param_groups[1]["lr"] = lr_models
        elif self.lifelong_optimizer.param_groups:
            self.lifelong_optimizer.param_groups[0]["lr"] = lr_policy

        alpha_conf = 0.2
        alpha_unc = 0.2
        prior_traits = self.agent.traits.detach().clone()
        current_traits = prior_traits.clone()
        regime_step_norms: Dict[str, List[float]] = {}
        regime_step_counts: Dict[str, int] = {}
        step_size_reflect_base = float(self.trait_reflection_lr)
        n_steps_reflect_base = self.trait_reflection_steps_per_batch
        lambda_reg = 0.0
        lambda_prior_base = 5.0e-3
        if self.is_minigrid:
            n_steps_reflect_base = max(self.trait_reflection_steps_per_batch, self.trait_reflection_steps_per_batch)

        regimes = regime_configs or self._default_regimes()
        self.regimes = {rc.name: rc for rc in regimes}
        self.regime_priorities = {name: 1.0 for name in self.regimes.keys()}
        schedule_base: List[RegimeConfig] = regimes if regimes else [RegimeConfig("R1", {"balanced": 1.0})]
        schedule_return: List[RegimeConfig] = [
            RegimeConfig(
                name=f"{reg.name}_return",
                scenario_weights=dict(reg.scenario_weights),
                reward_profile=dict(reg.reward_profile or {}),
                description=reg.description,
            )
            for reg in schedule_base
        ]
        schedule: List[RegimeConfig] = schedule_base + schedule_return

        scenario_map = self._get_scenario_name_to_id()
        baseline_regime_perf: Dict[str, float] = {}
        per_chapter: List[Dict[str, Any]] = []
        trait_stats: Dict[str, Dict[str, Any]] = {}
        planner_alpha_vals_all: List[float] = []
        planner_js_vals_all: List[float] = []
        planner_margin_vals_all: List[float] = []
        planner_override_vals_all: List[float] = []

        def _adaptation_delta(values: List[float]) -> Optional[float]:
            if len(values) < 2:
                return None
            half = max(1, len(values) // 2)
            head = float(np.mean(values[:half]))
            tail = float(np.mean(values[-half:]))
            return tail - head

        def _collect_trait_stats(trait_head: torch.Tensor, trait_tail: torch.Tensor) -> Dict[str, Any]:
            trait_head_list = trait_head.detach().cpu().numpy().flatten().tolist()
            trait_tail_list = trait_tail.detach().cpu().numpy().flatten().tolist()
            dist_head = float(torch.norm(trait_head - prior_traits).item())
            dist_tail = float(torch.norm(trait_tail - prior_traits).item())
            change_within = float(torch.norm(trait_tail - trait_head).item())
            return {
                "trait_head": trait_head_list,
                "trait_tail": trait_tail_list,
                "traits_final": trait_tail_list,
                "trait_dist_from_init_head": dist_head,
                "trait_dist_from_init_tail": dist_tail,
                "trait_change_within_regime": change_within,
                "trait_change_norm": change_within,
            }

        for chapter_idx, regime in enumerate(schedule):
            self.current_regime_name = regime.name
            if chapter_idx == 0:
                self._set_main_traits(prior_traits)
            else:
                self._set_main_traits(current_traits)
            recalled_traits, trait_memory_recall = self._recall_lifelong_traits(
                current_traits=self._main_traits().detach().clone(),
                regime_name=regime.name,
            )
            self._set_main_traits(recalled_traits)

            reward_profile = regime.reward_profile or {}
            trait_head = self.agent.traits.detach().clone()

            chapter_returns: List[float] = []
            chapter_lengths: List[int] = []
            chapter_food: List[int] = []
            chapter_damage: List[int] = []
            chapter_survival: List[float] = []
            chapter_uncertainty: List[float] = []
            scenario_counts: Dict[str, int] = {}
            chapter_transitions: List[Transition] = []
            planner_alpha_vals_ch: List[float] = []
            planner_js_vals_ch: List[float] = []
            planner_margin_vals_ch: List[float] = []
            planner_override_vals_ch: List[float] = []

            for ep_idx in range(episodes_per_chapter):
                scenario_id = self._sample_scenario_for_regime(regime, scenario_map)
                obs = self.env.reset(scenario_id=scenario_id)
                scenario_counts[str(scenario_id)] = scenario_counts.get(str(scenario_id), 0) + 1
                patch = obs["patch"]
                energy = obs["energy"]
                scenario_id_ep = int(obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id)))
                env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

                h_w = torch.zeros(
                    1,
                    1,
                    self.agent.world_model.gru.hidden_size,
                    device=self.device,
                )
                h_s = torch.zeros(
                    1,
                    1,
                    self.agent.self_model.gru.hidden_size,
                    device=self.device,
                )
                last_action = torch.zeros(1, dtype=torch.long, device=self.device)
                last_reward = 0.0
                traits = self._mixed_traits()
                M = self.agent.memory
                skill_state = {
                    "logprob": None,
                    "entropy": None,
                    "obs_history": [],
                    "active_skill": None,
                    "step_in_skill": 0,
                }

                total_r = 0.0
                t = 0
                done = False
                food_count = 0
                damage_count = 0
                unc_episode = 0.0
                unc_steps = 0
                last_info: Dict[str, Any] = {}
                trajectory: List[Dict[str, Any]] = []

                while not done and t < getattr(self.env, "max_steps", 200):
                    skill_state["obs_history"].append({"patch": patch.copy(), "energy": float(energy)})
                    with torch.no_grad():
                        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                        H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                        scenario_t = torch.tensor([scenario_id_ep], dtype=torch.long, device=self.device)
                        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        env_desc_t = self._env_desc_from_ids(env_t)

                        text_t = self._text_tokens_from_ids(env_t, scenario_t)
                        z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                        W_t = h_w.squeeze(0)

                        if use_self_flag:
                            r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                            a_emb = self.agent.self_model.act_emb(last_action)
                            env_emb = self.agent.self_model.env_desc_to_emb(env_desc_t)
                            W_in = W_t.unsqueeze(1)
                            H_in = H_t.unsqueeze(1)
                            r_in = r_t.unsqueeze(1)
                            M_b = M.unsqueeze(1)
                            x_s = torch.cat(
                                [W_in, H_in, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)],
                                dim=-1,
                            )
                            out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                            S_t = out_s.squeeze(1)

                            surv_t = torch.sigmoid(self.agent.self_model.head_survival(S_t))
                            food_t = self.agent.self_model.head_food(S_t)
                            dmg_t = self.agent.self_model.head_damage(S_t)
                            move_t = self.agent.self_model.head_move(S_t)

                            w = traits_to_preference_weights(traits)
                            w_survive, w_food, w_danger, w_move = w[0]
                            R_unscaled = (
                                w_survive * surv_t
                                + w_food * food_t
                                + w_danger * dmg_t
                                + w_move * move_t
                            ).view(1, 1)
                            R_self = self.agent.self_model.head_return_calib(R_unscaled)
                            U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
                        else:
                            S_t = torch.zeros(
                                1,
                                self.agent.self_model.gru.hidden_size,
                                device=self.device,
                            )
                            R_self = torch.zeros(1, 1, device=self.device)
                            U_t = torch.zeros(1, 1, device=self.device)

                        unc_episode += float(U_t.mean().item())
                        unc_steps += 1

                        V_pi = self.agent.value_model(W_t, H_t, traits, M)

                        if use_self_flag:
                            conf_t = torch.abs(R_self - V_pi)
                            gate = torch.exp(-alpha_conf * conf_t - alpha_unc * U_t)
                            delta_raw = torch.tanh(R_self - V_pi)
                            delta_self = delta_raw * gate
                        else:
                            delta_self = torch.zeros_like(R_self)

                        G_t = self.agent.workspace(
                            W_t,
                            S_t,
                            H_t,
                            V_pi,
                            delta_self,
                            U_t,
                            traits,
                            M,
                        )
                        if self.use_skills and self.agent.high_level_policy is not None and self._total_skill_count() > 0:
                            action, _, _, skill_state = self._select_action_with_skills(
                                G_t=G_t,
                                z_obs=z_obs,
                                H_t=H_t,
                                h_w=h_w,
                                traits=traits,
                                M=M,
                                planning_coef=planning_coef,
                                skill_state=skill_state,
                                env_desc=env_desc_t,
                                W_t=W_t,
                            )
                            if action is None:
                                logits = self.agent.policy(G_t)
                                logits = self._apply_action_mask(logits)
                                dist = Categorical(logits=logits)
                                action = dist.sample()
                        else:
                            logits = self.agent.policy(G_t)

                            if planning_coef > 0.0:
                                planner_logits = self._get_planner_logits(
                                    z_obs=z_obs,
                                    H_t=H_t,
                                    h_w=h_w,
                                    traits=traits,
                                    M=M,
                                )
                                logits, planner_debug = self._blend_with_planner(
                                    policy_logits=logits,
                                    planner_logits=planner_logits,
                                    base_planning_coef=float(planning_coef),
                                    uncertainty=U_t if use_self_flag else None,
                                    r_self=R_self if use_self_flag else None,
                                    v_pi=V_pi if use_self_flag else None,
                                )
                                planner_alpha_vals_ch.append(float(planner_debug.get("planner_alpha", 0.0)))
                                planner_js_vals_ch.append(float(planner_debug.get("planner_js", 0.0)))
                                planner_margin_vals_ch.append(float(planner_debug.get("planner_margin", 0.0)))
                                planner_override_vals_ch.append(float(planner_debug.get("planner_override", 0.0)))

                            logits = self._apply_action_mask(logits)
                            dist = Categorical(logits=logits)
                            action = dist.sample()

                    next_obs, _, done, info = self.env.step(action.item())
                    reward_env = self.compute_preference_reward(info, reward_profile=reward_profile)
                    total_r += reward_env
                    t += 1
                    last_info = info

                    if info.get("got_food", False):
                        food_count += 1
                    if info.get("took_damage", False):
                        damage_count += 1

                    trajectory.append(
                        {
                            "obs_patch": patch.copy(),
                            "energy": float(energy),
                            "action": int(action.item()),
                            "reward": float(reward_env),
                            "done": done,
                            "next_obs_patch": next_obs["patch"].copy(),
                            "next_energy": float(next_obs["energy"]),
                            "death_flag": float(info.get("death_flag", 0.0)),
                            "got_food": float(info.get("got_food", False)),
                            "took_damage": float(info.get("took_damage", False)),
                            "moved": float(info.get("moved", False)),
                            "alive": float(info.get("alive", True)),
                            "scenario_id": int(info.get("scenario_id", scenario_id_ep)),
                            "env_id": int(info.get("env_id", env_id)),
                        }
                    )

                    tr = Transition(
                        obs_patch=patch,
                        energy=float(energy),
                        action=int(action.item()),
                        reward=float(reward_env),
                        done=bool(done),
                        next_obs_patch=next_obs["patch"],
                        next_energy=float(next_obs["energy"]),
                        death_flag=float(info.get("death_flag", 0.0)),
                        got_food=float(info.get("got_food", False)),
                        took_damage=float(info.get("took_damage", False)),
                        moved=float(info.get("moved", False)),
                        alive=float(info.get("alive", True)),
                        scenario_id=int(info.get("scenario_id", scenario_id_ep)),
                        env_id=int(info.get("env_id", env_id)),
                        regime_name=regime.name,
                    )
                    chapter_transitions.append(tr)

                    patch = next_obs["patch"]
                    energy = next_obs["energy"]
                    scenario_id_ep = int(next_obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id_ep)))
                    env_id = int(next_obs.get("env_id", env_id))

                    with torch.no_grad():
                        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                        H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                        scenario_t = torch.tensor([scenario_id_ep], dtype=torch.long, device=self.device)
                        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        env_desc_t = self._env_desc_from_ids(env_t)
                        text_t = self._text_tokens_from_ids(env_t, scenario_t)
                        z_obs_next = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                        _, h_w, _, _ = self.agent.world_model.forward_step(z_obs_next, H_t, action, h_w)

                    last_reward = reward_env
                    last_action = action

                chapter_returns.append(total_r)
                chapter_lengths.append(t)
                chapter_food.append(food_count)
                chapter_damage.append(damage_count)
                avg_unc = float(unc_episode / max(1, unc_steps))
                chapter_uncertainty.append(avg_unc)
                survival_flag = 1.0 if (last_info.get("alive", True) if last_info else True) else 0.0
                chapter_survival.append(survival_flag)

                if reflect_enabled and trajectory:
                    obs_seq = np.stack([tr["obs_patch"] for tr in trajectory], axis=0)[None, ...]
                    H_seq = np.array([[tr["energy"] for tr in trajectory]], dtype=np.float32)[..., None]
                    a_seq = np.array([tr["action"] for tr in trajectory], dtype=np.int64)[None, ...]
                    r_seq = np.array([tr["reward"] for tr in trajectory], dtype=np.float32)[None, ...]
                    d_seq = np.array([float(tr["done"]) for tr in trajectory], dtype=np.float32)[None, ...]
                    death_seq = np.array([tr["death_flag"] for tr in trajectory], dtype=np.float32)[None, ...]
                    food_seq = np.array([tr["got_food"] for tr in trajectory], dtype=np.float32)[None, ...]
                    dmg_seq = np.array([tr["took_damage"] for tr in trajectory], dtype=np.float32)[None, ...]
                    move_seq = np.array([tr["moved"] for tr in trajectory], dtype=np.float32)[None, ...]
                    alive_seq = np.array([tr["alive"] for tr in trajectory], dtype=np.float32)[None, ...]
                    scenario_seq = np.array([tr["scenario_id"] for tr in trajectory], dtype=np.int64)[None, ...]
                    env_seq = np.array([tr["env_id"] for tr in trajectory], dtype=np.int64)[None, ...]

                    anchor_traits = self._main_traits().detach().clone()
                    death_flag_last = float(last_info.get("death_flag", 0.0)) if isinstance(last_info, dict) else 0.0
                    alive_last = bool(last_info.get("alive", True)) if isinstance(last_info, dict) else True
                    high_damage = int(damage_count) >= max(1, int(0.15 * max(1, int(t))))
                    high_safety_risk = bool(high_damage or death_flag_last > 0.0 or (not alive_last))
                    reflect_cfg = self._lifelong_reflection_schedule(
                        episode_idx=ep_idx,
                        episodes_per_chapter=int(episodes_per_chapter),
                        base_steps=int(n_steps_reflect_base),
                        base_step_size=float(step_size_reflect_base),
                        base_lambda_prior=float(lambda_prior_base),
                        high_safety_risk=high_safety_risk,
                    )
                    step_size_reflect = float(reflect_cfg["step_size"])
                    n_steps_reflect = int(reflect_cfg["steps"])
                    lambda_prior_cur = float(reflect_cfg["lambda_prior"])
                    info_reflect = self._run_trait_reflection(
                        batch=(
                            obs_seq,
                            H_seq,
                            a_seq,
                            r_seq,
                            d_seq,
                            death_seq,
                            food_seq,
                            dmg_seq,
                            move_seq,
                            alive_seq,
                            scenario_seq,
                            env_seq,
                        ),
                        reward_profile=reward_profile,
                        step_size=step_size_reflect,
                        regime_name=regime.name,
                        lambda_l2=lambda_reg,
                        n_steps=n_steps_reflect,
                        trait_anchor=anchor_traits,
                        init_traits=anchor_traits,
                        safety_phase="stage3b",
                        safety_baseline=anchor_traits,
                    )
                    regime_step_norms.setdefault(regime.name, []).extend(info_reflect.get("step_norms", []))
                    regime_step_counts[regime.name] = regime_step_counts.get(regime.name, 0) + int(info_reflect.get("steps", 0))
                    with torch.no_grad():
                        self.agent.traits.add_(lambda_prior_cur * (prior_traits - self.agent.traits))
                        self.agent.traits.clamp_(-2.0, 2.0)

            current_traits = self.agent.traits.detach().clone()
            stats = _collect_trait_stats(trait_head, current_traits)
            trait_stats.setdefault(regime.name, {})[agent_variant] = {**stats, "reward_profile": reward_profile}
            mean_ret = float(np.mean(chapter_returns)) if chapter_returns else 0.0
            std_ret = float(np.std(chapter_returns)) if chapter_returns else 0.0
            mean_len = float(np.mean(chapter_lengths)) if chapter_lengths else 0.0
            mean_food = float(np.mean(chapter_food)) if chapter_food else 0.0
            mean_damage = float(np.mean(chapter_damage)) if chapter_damage else 0.0
            trait_memory_updated = self._update_lifelong_trait_memory_if_better(
                regime_name=regime.name,
                mean_return=mean_ret,
                traits_main=current_traits,
            )
            baseline_val = baseline_regime_perf.setdefault(regime.name, mean_ret)
            self._record_regime_stats(
                regime_name=regime.name,
                returns=chapter_returns,
                survival_flags=chapter_survival,
                food_counts=chapter_food,
                damage_counts=chapter_damage,
                uncertainty_vals=chapter_uncertainty,
                baseline_return=baseline_val,
                move_counts=chapter_lengths,
            )
            if self.regime_generator is not None:
                proposals = self.regime_generator.propose_regimes(self.regimes, self.regime_stats)
                self._apply_regime_proposals(proposals)

            policy_losses: List[float] = []
            self_losses: List[float] = []
            if allow_online_model_updates and chapter_transitions:
                chapter_buffer = ReplayBuffer(max(len(chapter_transitions), 1))
                for tr in chapter_transitions:
                    chapter_buffer.push(tr)
                old_buffer = self.lifelong_buffer if len(self.lifelong_buffer) >= 1 else None
                updates_per_chapter = max(1, episodes_per_chapter // 10)
                for _ in range(updates_per_chapter):
                    batch_policy = self._sample_mixed_sequences(
                        new_buffer=chapter_buffer,
                        old_buffer=old_buffer,
                        batch_size=16,
                        seq_len=16,
                        replay_ratio_old=replay_ratio_old,
                        with_events=False,
                    )
                    if batch_policy is not None:
                        stats_loss = self._policy_update_from_sequences(
                            batch_policy,
                            use_self=use_self_flag,
                            gamma=0.99,
                            entropy_coef=entropy_coef,
                            beta_conflict=0.0,
                            beta_uncertainty=0.0,
                            regularization_coef=regularization_coef,
                            optimizer=self.lifelong_optimizer,
                        )
                        if stats_loss is not None:
                            policy_losses.append(float(stats_loss.get("policy_loss", 0.0)))

                    batch_self = self._sample_mixed_sequences(
                        new_buffer=chapter_buffer,
                        old_buffer=old_buffer,
                        batch_size=16,
                        seq_len=16,
                        replay_ratio_old=replay_ratio_old,
                        with_events=True,
                    )
                    if batch_self is not None:
                        loss_val = self._self_model_update_from_batch(
                            batch_self,
                            gamma=0.99,
                            regularization_coef=regularization_coef,
                            optimizer=self.lifelong_optimizer,
                        )
                        if loss_val is not None:
                            self_losses.append(loss_val)

                for tr in chapter_transitions:
                    self.lifelong_buffer.push(tr)
            else:
                for tr in chapter_transitions:
                    self.lifelong_buffer.push(tr)

            chapter_planner_summary = self._planner_debug_summary(
                alpha_values=planner_alpha_vals_ch,
                js_values=planner_js_vals_ch,
                margin_values=planner_margin_vals_ch,
                override_values=planner_override_vals_ch,
            )
            planner_alpha_vals_all.extend(planner_alpha_vals_ch)
            planner_js_vals_all.extend(planner_js_vals_ch)
            planner_margin_vals_all.extend(planner_margin_vals_ch)
            planner_override_vals_all.extend(planner_override_vals_ch)
            per_chapter.append(
                {
                    "regime": regime.name,
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "mean_length": mean_len,
                    "mean_food": mean_food,
                    "mean_damage": mean_damage,
                    "n_episodes": int(len(chapter_returns)),
                    **stats,
                    "reward_profile": reward_profile,
                    "description": regime.description,
                    "returns": [float(x) for x in chapter_returns],
                    "scenario_counts": scenario_counts,
                    "trait_memory_recall": dict(trait_memory_recall),
                    "trait_memory_updated": bool(trait_memory_updated),
                    "train_info": {
                        "policy_updates": int(len(policy_losses)),
                        "self_model_updates": int(len(self_losses)),
                        "last_policy_loss": float(policy_losses[-1]) if policy_losses else None,
                        "last_self_model_loss": float(self_losses[-1]) if self_losses else None,
                    },
                    "planner_debug": chapter_planner_summary,
                }
            )

        r2_delta = _adaptation_delta(per_chapter[1]["returns"]) if len(per_chapter) > 1 else None
        r3_delta = _adaptation_delta(per_chapter[2]["returns"]) if len(per_chapter) > 2 else None
        forgetting_gap = None
        retain_score = None
        # regime-specific forgetting using "_return" chapters when available
        eval_current: Dict[str, float] = {}
        for regime in schedule:
            try:
                eval_current[regime.name] = self._evaluate_regime_performance(
                    regime, scenario_map=scenario_map, episodes=5, max_steps=getattr(self.env, "max_steps", 200)
                )
            except Exception:
                eval_current[regime.name] = None  # type: ignore
        forgetting_per_regime: Dict[str, float] = {}
        for name, base_val in baseline_regime_perf.items():
            return_name = f"{name}_return"
            chapter_entry = next((c for c in per_chapter if c.get("regime") == return_name), None)
            cur = None
            if chapter_entry is not None:
                cur = chapter_entry.get("mean_return")
            if cur is None:
                cur = eval_current.get(name)
            if cur is None:
                continue
            gap_val = float(cur) - float(base_val)
            forgetting_per_regime[name] = gap_val
        if forgetting_per_regime:
            first_name = next(iter(forgetting_per_regime))
            forgetting_gap = forgetting_per_regime.get(first_name)
            retain_score = forgetting_gap
        self.current_regime_name = ""

        metrics = {
            "agent_variant": agent_variant,
            "use_self": use_self_flag,
            "online_reflection": reflect_enabled,
            "episodes_per_chapter": int(episodes_per_chapter),
            "lifelong_regimes": [rc.name for rc in schedule],
            "lifelong_reward_profiles": {rc.name: rc.reward_profile or {} for rc in schedule},
            "lifelong_per_chapter": per_chapter,
            "lifelong_trait_stats": trait_stats,
            "lifelong_adaptation_R2_delta": float(r2_delta) if r2_delta is not None else None,
            "lifelong_adaptation_R3_delta": float(r3_delta) if r3_delta is not None else None,
            "lifelong_forgetting_R1_gap": float(forgetting_gap) if forgetting_gap is not None else None,
            "lifelong_forgetting_per_regime": forgetting_per_regime,
            "retain_score": float(retain_score) if retain_score is not None else None,
            "prior_traits": prior_traits.detach().cpu().numpy().flatten().tolist(),
            "final_traits": self.agent.traits.detach().cpu().numpy().flatten().tolist(),
            "fast_param_l2_from_init": float(self._fast_param_distance_from_init()),
            "lifelong_trait_memory_keys": sorted(self.lifelong_trait_memory.keys()),
            "lifelong_trait_memory_scores": {
                str(k): float(v) for k, v in self.lifelong_trait_memory_score.items()
            },
            "lifelong_forgetting": {
                "baseline": baseline_regime_perf,
                "current": eval_current,
                "gap": forgetting_per_regime,
            },
        }
        metrics.update(
            self._planner_debug_summary(
                alpha_values=planner_alpha_vals_all,
                js_values=planner_js_vals_all,
                margin_values=planner_margin_vals_all,
                override_values=planner_override_vals_all,
            )
        )
        trait_reflection_debug: Dict[str, Any] = {}
        for regime in schedule:
            regime_name = regime.name
            norms = regime_step_norms.get(regime_name, [])
            mean_norm = float(np.mean(norms)) if norms else 0.0
            steps_count = int(regime_step_counts.get(regime_name, 0))
            metrics[f"lifelong_trait_reflection_mean_step_norm_{regime_name}"] = mean_norm
            trait_reflection_debug[regime_name] = {
                "mean_step_norm": mean_norm,
                "steps": steps_count,
            }
        metrics["trait_reflection_debug"] = trait_reflection_debug
        self.trait_reflection_debug.update({f"lifelong_train_{k}": v for k, v in trait_reflection_debug.items()})
        if self.logger is not None and forgetting_per_regime:
            for reg_name, gap_val in forgetting_per_regime.items():
                self.logger.log_scalar(
                    stage="lifelong_forgetting",
                    metric=reg_name,
                    value=float(gap_val),
                    baseline=float(baseline_regime_perf.get(reg_name, 0.0)),
                    current=float(eval_current.get(reg_name, 0.0) if eval_current.get(reg_name) is not None else 0.0),
                )
        return metrics

    def run_lifelong_eval(
        self,
        episodes_per_chapter: int = 50,
        planning_coef: float = 0.5,
        agent_variant: str = "full",
        regime_configs: Optional[List[RegimeConfig]] = None,
        allow_online_reflection: bool = True,
        eval_policy: str = "sample",
        stratified_scenarios: bool = False,
        lr: float = 5e-2,
        lambda_reg: float = 1e-2,
        lambda_conflict: float = 0.0,
        lambda_prior: float = 5e-3,
    ) -> Dict[str, Any]:
        """
        Evaluate sequential non-stationary regimes with optional online trait updates.
        Policy/world-model/self-model stay frozen; only traits may adapt (full variant).
        """
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.eval()
        self.agent.workspace.eval()
        self.agent.policy.eval()
        self.agent.value_model.eval()

        use_self_flag = agent_variant != "no_self"
        allow_trait_updates = agent_variant == "full" and use_self_flag
        reflect_enabled = allow_online_reflection and allow_trait_updates
        eval_policy_norm = (eval_policy or "sample").lower()
        if eval_policy_norm not in {"sample", "greedy"}:
            eval_policy_norm = "sample"

        alpha_conf = 0.2
        alpha_unc = 0.2
        prior_traits = self.agent.traits.detach().clone()
        initial_traits = prior_traits.clone()
        current_traits = prior_traits.clone()
        regime_step_norms: Dict[str, List[float]] = {}
        regime_step_counts: Dict[str, int] = {}
        step_size_reflect_base = float(lr)
        lambda_prior_base = float(lambda_prior)
        if self.is_minigrid:
            step_size_reflect_base = float(self.trait_reflection_lr)
        n_steps_reflect_base = self.trait_reflection_steps_per_batch if not self.is_minigrid else max(
            self.trait_reflection_steps_per_batch, self.trait_reflection_steps_per_batch
        )

        regimes = regime_configs or self._default_regimes()
        schedule_base: List[RegimeConfig] = regimes if regimes else [RegimeConfig("R1", {"balanced": 1.0})]
        schedule_return: List[RegimeConfig] = [
            RegimeConfig(
                name=f"{reg.name}_return",
                scenario_weights=dict(reg.scenario_weights),
                reward_profile=dict(reg.reward_profile or {}),
                description=reg.description,
            )
            for reg in schedule_base
        ]
        schedule: List[RegimeConfig] = schedule_base + schedule_return

        scenario_map = self._get_scenario_name_to_id()
        baseline_regime_perf: Dict[str, float] = {}
        per_chapter: List[Dict[str, Any]] = []
        trait_stats: Dict[str, Dict[str, Any]] = {}
        planner_alpha_vals_all: List[float] = []
        planner_js_vals_all: List[float] = []
        planner_margin_vals_all: List[float] = []
        planner_override_vals_all: List[float] = []

        def _adaptation_delta(values: List[float]) -> Optional[float]:
            if len(values) < 2:
                return None
            half = max(1, len(values) // 2)
            head = float(np.mean(values[:half]))
            tail = float(np.mean(values[-half:]))
            return tail - head

        def _collect_trait_stats(trait_head: torch.Tensor, trait_tail: torch.Tensor) -> Dict[str, Any]:
            trait_head_list = trait_head.detach().cpu().numpy().flatten().tolist()
            trait_tail_list = trait_tail.detach().cpu().numpy().flatten().tolist()
            dist_head = float(torch.norm(trait_head - initial_traits).item())
            dist_tail = float(torch.norm(trait_tail - initial_traits).item())
            change_within = float(torch.norm(trait_tail - trait_head).item())
            return {
                "trait_head": trait_head_list,
                "trait_tail": trait_tail_list,
                "traits_final": trait_tail_list,
                "trait_dist_from_init_head": dist_head,
                "trait_dist_from_init_tail": dist_tail,
                "trait_change_within_regime": change_within,
                "trait_change_norm": change_within,
            }

        for chapter_idx, regime in enumerate(schedule):
            # Reset traits for chapter start
            if chapter_idx == 0:
                self._set_main_traits(prior_traits)
            else:
                self._set_main_traits(current_traits)
            recalled_traits, trait_memory_recall = self._recall_lifelong_traits(
                current_traits=self._main_traits().detach().clone(),
                regime_name=regime.name,
            )
            self._set_main_traits(recalled_traits)

            reward_profile = regime.reward_profile or {}
            trait_head = self.agent.traits.detach().clone()

            chapter_returns: List[float] = []
            chapter_lengths: List[int] = []
            chapter_food: List[int] = []
            chapter_damage: List[int] = []
            chapter_survival: List[float] = []
            chapter_uncertainty: List[float] = []
            scenario_counts: Dict[str, int] = {}
            planner_alpha_vals_ch: List[float] = []
            planner_js_vals_ch: List[float] = []
            planner_margin_vals_ch: List[float] = []
            planner_override_vals_ch: List[float] = []
            scenario_plan = self._build_episode_scenario_plan(
                regime=regime,
                name_to_id=scenario_map,
                n_episodes=int(episodes_per_chapter),
                stratified=bool(stratified_scenarios),
            )
            for ep_idx in range(episodes_per_chapter):
                scenario_id = scenario_plan[ep_idx] if ep_idx < len(scenario_plan) else None
                if scenario_id is None:
                    scenario_id = self._sample_scenario_for_regime(regime, scenario_map)
                obs = self.env.reset(scenario_id=scenario_id)
                patch = obs["patch"]
                energy = obs["energy"]
                scenario_id_ep = int(
                    obs.get("scenario_id", getattr(self.env, "current_scenario_id", 0))
                )
                env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))

                scenario_name = obs.get(
                    "scenario_name",
                    getattr(self.env, "current_scenario_name", str(scenario_id_ep)),
                )
                scenario_counts[scenario_name] = scenario_counts.get(scenario_name, 0) + 1

                h_w = torch.zeros(
                    1,
                    1,
                    self.agent.world_model.gru.hidden_size,
                    device=self.device,
                )
                h_s = torch.zeros(
                    1,
                    1,
                    self.agent.self_model.gru.hidden_size,
                    device=self.device,
                )
                last_action = torch.zeros(1, dtype=torch.long, device=self.device)
                last_reward = 0.0
                traits = self._mixed_traits()
                M = self.agent.memory
                skill_state = {
                    "logprob": None,
                    "entropy": None,
                    "obs_history": [],
                    "active_skill": None,
                    "step_in_skill": 0,
                }

                total_r = 0.0
                t = 0
                done = False
                food_count = 0
                damage_count = 0
                trajectory = []
                unc_episode = 0.0
                unc_steps = 0
                last_info: Dict[str, Any] = {}

                while not done and t < self.env.max_steps:
                    skill_state["obs_history"].append({"patch": patch.copy(), "energy": float(energy)})
                    with torch.no_grad():
                        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                        H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                        scenario_t = torch.tensor([scenario_id_ep], dtype=torch.long, device=self.device)
                        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        env_desc_t = self._env_desc_from_ids(env_t)

                        text_t = self._text_tokens_from_ids(env_t, scenario_t)
                        z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                        W_t = h_w.squeeze(0)

                        if use_self_flag:
                            r_t = torch.tensor([[last_reward]], dtype=torch.float32, device=self.device)
                            a_emb = self.agent.self_model.act_emb(last_action)
                            env_emb = self.agent.self_model.env_desc_to_emb(env_desc_t)

                            W_in = W_t.unsqueeze(1)
                            H_in = H_t.unsqueeze(1)
                            r_in = r_t.unsqueeze(1)
                            M_b = M.unsqueeze(1)
                            x_s = torch.cat(
                                [W_in, H_in, r_in, a_emb.unsqueeze(1), M_b, env_emb.unsqueeze(1)],
                                dim=-1,
                            )
                            out_s, h_s = self.agent.self_model.gru(x_s, h_s)
                            S_t = out_s.squeeze(1)

                            surv_t = torch.sigmoid(self.agent.self_model.head_survival(S_t))
                            food_t = self.agent.self_model.head_food(S_t)
                            dmg_t = self.agent.self_model.head_damage(S_t)
                            move_t = self.agent.self_model.head_move(S_t)

                            w = traits_to_preference_weights(traits)
                            w_survive, w_food, w_danger, w_move = w[0]
                            R_unscaled = (
                                w_survive * surv_t
                                + w_food * food_t
                                + w_danger * dmg_t
                                + w_move * move_t
                            ).view(1, 1)
                            R_self = self.agent.self_model.head_return_calib(R_unscaled)
                            U_t = torch.abs(self.agent.self_model.head_uncertainty(S_t)).view(1, 1)
                        else:
                            S_t = torch.zeros(
                                1,
                                self.agent.self_model.gru.hidden_size,
                                device=self.device,
                            )
                            R_self = torch.zeros(1, 1, device=self.device)
                            U_t = torch.zeros(1, 1, device=self.device)

                        unc_episode += float(U_t.mean().item())
                        unc_steps += 1

                        V_pi = self.agent.value_model(W_t, H_t, traits, M)

                        if use_self_flag:
                            conf_t = torch.abs(R_self - V_pi)
                            gate = torch.exp(-alpha_conf * conf_t - alpha_unc * U_t)
                            delta_raw = torch.tanh(R_self - V_pi)
                            delta_self = delta_raw * gate
                        else:
                            delta_self = torch.zeros_like(R_self)

                        G_t = self.agent.workspace(
                            W_t,
                            S_t,
                            H_t,
                            V_pi,
                            delta_self,
                            U_t,
                            traits,
                            M,
                        )
                        logits = self.agent.policy(G_t)

                        if planning_coef > 0.0:
                            planner_logits = self._get_planner_logits(
                                z_obs=z_obs,
                                H_t=H_t,
                                h_w=h_w,
                                traits=traits,
                                M=M,
                            )
                            logits, planner_debug = self._blend_with_planner(
                                policy_logits=logits,
                                planner_logits=planner_logits,
                                base_planning_coef=float(planning_coef),
                                uncertainty=U_t if use_self_flag else None,
                                r_self=R_self if use_self_flag else None,
                                v_pi=V_pi if use_self_flag else None,
                            )
                            planner_alpha_vals_ch.append(float(planner_debug.get("planner_alpha", 0.0)))
                            planner_js_vals_ch.append(float(planner_debug.get("planner_js", 0.0)))
                            planner_margin_vals_ch.append(float(planner_debug.get("planner_margin", 0.0)))
                            planner_override_vals_ch.append(float(planner_debug.get("planner_override", 0.0)))

                        logits = self._apply_action_mask(logits)
                        if eval_policy_norm == "greedy":
                            action = torch.argmax(logits, dim=-1)
                        else:
                            dist = Categorical(logits=logits)
                            action = dist.sample()

                    next_obs, _, done, info = self.env.step(action.item())
                    reward_env = self.compute_preference_reward(
                        info, reward_profile=reward_profile
                    )
                    total_r += reward_env
                    t += 1
                    last_info = info

                    if info.get("got_food", False):
                        food_count += 1
                    if info.get("took_damage", False):
                        damage_count += 1

                    trajectory.append(
                        {
                            "obs_patch": patch.copy(),
                            "energy": float(energy),
                            "action": int(action.item()),
                            "reward": float(reward_env),
                            "done": done,
                            "next_obs_patch": next_obs["patch"].copy(),
                            "next_energy": float(next_obs["energy"]),
                            "death_flag": float(info.get("death_flag", 0.0)),
                            "got_food": float(info.get("got_food", False)),
                            "took_damage": float(info.get("took_damage", False)),
                            "moved": float(info.get("moved", False)),
                            "alive": float(info.get("alive", True)),
                            "scenario_id": int(info.get("scenario_id", scenario_id_ep)),
                            "env_id": int(info.get("env_id", env_id)),
                        }
                    )

                    patch = next_obs["patch"]
                    energy = next_obs["energy"]
                    scenario_id_ep = int(
                        next_obs.get("scenario_id", getattr(self.env, "current_scenario_id", scenario_id_ep))
                    )
                    env_id = int(next_obs.get("env_id", env_id))

                    with torch.no_grad():
                        patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(self.device)
                        H_t = torch.tensor([[energy]], dtype=torch.float32, device=self.device)
                        scenario_t = torch.tensor([scenario_id_ep], dtype=torch.long, device=self.device)
                        env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                        env_desc_t = self._env_desc_from_ids(env_t)
                        text_t = self._text_tokens_from_ids(env_t, scenario_t)
                        z_obs_next = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                        _, h_w, _, _ = self.agent.world_model.forward_step(z_obs_next, H_t, action, h_w)

                    last_reward = reward_env
                    last_action = action

                chapter_returns.append(total_r)
                chapter_lengths.append(t)
                chapter_food.append(food_count)
                chapter_damage.append(damage_count)
                avg_unc = float(unc_episode / max(1, unc_steps))
                chapter_uncertainty.append(avg_unc)
                survival_flag = 1.0 if (last_info.get("alive", True) if last_info else True) else 0.0
                chapter_survival.append(survival_flag)

                if reflect_enabled and trajectory:
                    obs_seq = np.stack([tr["obs_patch"] for tr in trajectory], axis=0)[None, ...]
                    H_seq = np.array([[tr["energy"]] for tr in trajectory], dtype=np.float32)[None, ...]
                    a_seq = np.array([tr["action"] for tr in trajectory], dtype=np.int64)[None, ...]
                    r_seq = np.array([tr["reward"] for tr in trajectory], dtype=np.float32)[None, ...]
                    d_seq = np.array([float(tr["done"]) for tr in trajectory], dtype=np.float32)[None, ...]
                    death_seq = np.array([tr["death_flag"] for tr in trajectory], dtype=np.float32)[None, ...]
                    food_seq = np.array([tr["got_food"] for tr in trajectory], dtype=np.float32)[None, ...]
                    dmg_seq = np.array([tr["took_damage"] for tr in trajectory], dtype=np.float32)[None, ...]
                    move_seq = np.array([tr["moved"] for tr in trajectory], dtype=np.float32)[None, ...]
                    alive_seq = np.array([tr["alive"] for tr in trajectory], dtype=np.float32)[None, ...]
                    scenario_seq = np.array([tr["scenario_id"] for tr in trajectory], dtype=np.int64)[None, ...]
                    env_seq = np.array([tr["env_id"] for tr in trajectory], dtype=np.int64)[None, ...]

                    anchor_traits = self._main_traits().detach().clone()
                    death_flag_last = float(last_info.get("death_flag", 0.0)) if isinstance(last_info, dict) else 0.0
                    alive_last = bool(last_info.get("alive", True)) if isinstance(last_info, dict) else True
                    high_damage = int(damage_count) >= max(1, int(0.15 * max(1, int(t))))
                    high_safety_risk = bool(high_damage or death_flag_last > 0.0 or (not alive_last))
                    reflect_cfg = self._lifelong_reflection_schedule(
                        episode_idx=ep_idx,
                        episodes_per_chapter=int(episodes_per_chapter),
                        base_steps=int(n_steps_reflect_base),
                        base_step_size=float(step_size_reflect_base),
                        base_lambda_prior=float(lambda_prior_base),
                        high_safety_risk=high_safety_risk,
                    )
                    step_size_reflect = float(reflect_cfg["step_size"])
                    n_steps_reflect = int(reflect_cfg["steps"])
                    lambda_prior_cur = float(reflect_cfg["lambda_prior"])
                    info_reflect = self._run_trait_reflection(
                        batch=(
                            obs_seq,
                            H_seq,
                            a_seq,
                            r_seq,
                            d_seq,
                            death_seq,
                            food_seq,
                            dmg_seq,
                            move_seq,
                            alive_seq,
                            scenario_seq,
                            env_seq,
                        ),
                        reward_profile=reward_profile,
                        step_size=step_size_reflect,
                        regime_name=regime.name,
                        lambda_l2=lambda_reg,
                        n_steps=n_steps_reflect,
                        trait_anchor=anchor_traits,
                        init_traits=anchor_traits,
                        safety_phase="stage3b",
                        safety_baseline=anchor_traits,
                    )
                    regime_step_norms.setdefault(regime.name, []).extend(info_reflect.get("step_norms", []))
                    regime_step_counts[regime.name] = regime_step_counts.get(regime.name, 0) + int(info_reflect.get("steps", 0))
                    updated = bool(info_reflect.get("updated", False))
                    steps = int(info_reflect.get("steps", 0))
                    step_norms = info_reflect.get("step_norms", []) or []
                    stats = self.trait_reflection_debug.setdefault(
                        regime.name,
                        {
                            "trait_reflection_n_calls": 0,
                            "trait_reflection_n_updates": 0,
                            "trait_reflection_total_steps": 0,
                            "trait_reflection_step_norm_sum": 0.0,
                        },
                    )
                    stats["trait_reflection_n_calls"] += 1
                    stats["trait_reflection_total_steps"] += steps
                    if updated:
                        stats["trait_reflection_n_updates"] += 1
                        stats["trait_reflection_step_norm_sum"] += float(sum(step_norms))
                    # pull gently toward prior traits
                    with torch.no_grad():
                        self.agent.traits.add_(lambda_prior_cur * (prior_traits - self.agent.traits))
                        self.agent.traits.clamp_(-2.0, 2.0)

            current_traits = self.agent.traits.detach().clone()
            stats = _collect_trait_stats(trait_head, current_traits)
            trait_stats.setdefault(regime.name, {})[agent_variant] = {
                **stats,
                "reward_profile": reward_profile,
            }
            mean_ret = float(np.mean(chapter_returns)) if chapter_returns else 0.0
            std_ret = float(np.std(chapter_returns)) if chapter_returns else 0.0
            mean_len = float(np.mean(chapter_lengths)) if chapter_lengths else 0.0
            mean_food = float(np.mean(chapter_food)) if chapter_food else 0.0
            mean_damage = float(np.mean(chapter_damage)) if chapter_damage else 0.0
            trait_memory_updated = self._update_lifelong_trait_memory_if_better(
                regime_name=regime.name,
                mean_return=mean_ret,
                traits_main=current_traits,
            )
            prev_stats = self.regime_stats.get(regime.name)
            baseline_val = mean_ret if prev_stats is None else float(
                prev_stats["avg_return"] + prev_stats["forgetting_gap"]
            )
            baseline_regime_perf.setdefault(regime.name, mean_ret)
            self._record_regime_stats(
                regime_name=regime.name,
                returns=chapter_returns,
                survival_flags=chapter_survival,
                food_counts=chapter_food,
                damage_counts=chapter_damage,
                uncertainty_vals=chapter_uncertainty,
                baseline_return=baseline_val,
                move_counts=chapter_lengths,
            )
            chapter_planner_summary = self._planner_debug_summary(
                alpha_values=planner_alpha_vals_ch,
                js_values=planner_js_vals_ch,
                margin_values=planner_margin_vals_ch,
                override_values=planner_override_vals_ch,
            )
            planner_alpha_vals_all.extend(planner_alpha_vals_ch)
            planner_js_vals_all.extend(planner_js_vals_ch)
            planner_margin_vals_all.extend(planner_margin_vals_ch)
            planner_override_vals_all.extend(planner_override_vals_ch)
            per_chapter.append(
                {
                    "regime": regime.name,
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "mean_length": mean_len,
                    "mean_food": mean_food,
                    "mean_damage": mean_damage,
                    "n_episodes": int(len(chapter_returns)),
                    **stats,
                    "reward_profile": reward_profile,
                    "description": regime.description,
                    "returns": [float(x) for x in chapter_returns],
                    "scenario_counts": scenario_counts,
                    "trait_memory_recall": dict(trait_memory_recall),
                    "trait_memory_updated": bool(trait_memory_updated),
                    "planner_debug": chapter_planner_summary,
                }
            )

        # Continual-learning indicators
        r2_delta = _adaptation_delta(per_chapter[1]["returns"]) if len(per_chapter) > 1 else None
        r3_delta = _adaptation_delta(per_chapter[2]["returns"]) if len(per_chapter) > 2 else None
        forgetting_gap = None
        forgetting_per_regime: Dict[str, float] = {}
        for name, base_val in baseline_regime_perf.items():
            return_name = f"{name}_return"
            entry = next((c for c in per_chapter if c.get("regime") == return_name), None)
            cur = entry.get("mean_return") if entry is not None else None
            if cur is None:
                continue
            gap_val = float(cur) - float(base_val)
            forgetting_per_regime[name] = gap_val
        if forgetting_per_regime:
            forgetting_gap = forgetting_per_regime.get(next(iter(forgetting_per_regime)))

        metrics = {
            "agent_variant": agent_variant,
            "use_self": use_self_flag,
            "online_reflection": reflect_enabled,
            "eval_policy": eval_policy_norm,
            "stratified_scenarios": bool(stratified_scenarios),
            "episodes_per_chapter": int(episodes_per_chapter),
            "lifelong_regimes": [rc.name for rc in schedule],
            "lifelong_reward_profiles": {
                rc.name: rc.reward_profile or {} for rc in schedule
            },
            "lifelong_per_chapter": per_chapter,
            "lifelong_trait_stats": trait_stats,
            "lifelong_adaptation_R2_delta": float(r2_delta) if r2_delta is not None else None,
            "lifelong_adaptation_R3_delta": float(r3_delta) if r3_delta is not None else None,
            "lifelong_forgetting_R1_gap": float(forgetting_gap) if forgetting_gap is not None else None,
            "lifelong_forgetting_per_regime": forgetting_per_regime,
            "prior_traits": prior_traits.detach().cpu().numpy().flatten().tolist(),
            "final_traits": current_traits.detach().cpu().numpy().flatten().tolist(),
            "lifelong_trait_memory_keys": sorted(self.lifelong_trait_memory.keys()),
            "lifelong_trait_memory_scores": {
                str(k): float(v) for k, v in self.lifelong_trait_memory_score.items()
            },
        }
        metrics.update(
            self._planner_debug_summary(
                alpha_values=planner_alpha_vals_all,
                js_values=planner_js_vals_all,
                margin_values=planner_margin_vals_all,
                override_values=planner_override_vals_all,
            )
        )
        trait_reflection_debug: Dict[str, Any] = {}
        for regime in schedule:
            regime_name = regime.name
            norms = regime_step_norms.get(regime_name, [])
            mean_norm = float(np.mean(norms)) if norms else 0.0
            steps_count = int(regime_step_counts.get(regime_name, 0))
            metrics[f"lifelong_trait_reflection_mean_step_norm_{regime_name}"] = mean_norm
            trait_reflection_debug[regime_name] = {
                "mean_step_norm": mean_norm,
                "steps": steps_count,
            }
        metrics["trait_reflection_debug"] = trait_reflection_debug
        self.trait_reflection_debug.update({f"lifelong_{k}": v for k, v in trait_reflection_debug.items()})
        return metrics

    # =========================
    #  SelfModel probe
    # =========================

    def run_selfmodel_probe(
        self,
        n_episodes: int = 30,
        max_steps: int = 200,
        gamma: float = 0.99,
        use_self: bool = False,
    ) -> SelfModelProbeStats:
        """Run self-model probe and return SelfModelProbeStats."""
        self.agent.perception.eval()
        self.agent.world_model.eval()
        self.agent.self_model.eval()

        true_returns = []
        pred_returns = []
        true_survival0 = []
        pred_survival0 = []

        for _ in range(n_episodes):
            obs = self.env.reset(split="train")
            patch = obs["patch"]
            energy = obs["energy"]
            scenario_id = getattr(self.env, "current_scenario_id", 0)
            env_id = int(obs.get("env_id", getattr(self.env, "env_id", 0)))
            env_id_ep = env_id

            patches = []
            energies = []
            actions = []
            rewards = []
            dones = []
            death_flags = []

            h_w = torch.zeros(
                1,
                1,
                self.agent.world_model.gru.hidden_size,
                device=self.device,
            )

            traits = self._mixed_traits()
            M = self.agent.memory

            t = 0
            done = False

            while not done and t < max_steps:
                patches.append(patch.copy())
                energies.append(energy)

                with torch.no_grad():
                    patch_t = torch.from_numpy(patch).long().unsqueeze(0).to(
                        self.device
                    )
                    H_t = torch.tensor(
                        [[energy]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    scenario_t = torch.tensor(
                        [scenario_id], dtype=torch.long, device=self.device
                    )
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)
                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_obs = self.agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
                    W_t = h_w.squeeze(0)

                    S_zero = torch.zeros(
                        1,
                        self.agent.self_model.gru.hidden_size,
                        device=self.device,
                    )
                    R_zero = torch.zeros(1, 1, device=self.device)
                    U_zero = torch.zeros(1, 1, device=self.device)
                    V_pi = self.agent.value_model(W_t, H_t, traits, M)

                    G_t = self.agent.workspace(
                        W_t,
                        S_zero,
                        H_t,
                        V_pi,
                        R_zero,
                        U_zero,
                        traits,
                        M,
                    )

                    logits = self.agent.policy(G_t)
                    logits = self._apply_action_mask(logits)
                    dist = Categorical(logits=logits)
                    action = dist.sample()

                next_obs, _, done, info = self.env.step(action.item())
                reward_env = self.compute_preference_reward(info)

                actions.append(action.item())
                rewards.append(reward_env)
                dones.append(float(done))
                death_flags.append(float(info.get("death_flag", 0.0)))

                with torch.no_grad():
                    patch_next_t = torch.from_numpy(next_obs["patch"]).long().unsqueeze(
                        0
                    ).to(self.device)
                    H_next = torch.tensor(
                        [[next_obs["energy"]]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    scenario_t = torch.tensor(
                        [scenario_id], dtype=torch.long, device=self.device
                    )
                    env_t = torch.tensor([env_id], dtype=torch.long, device=self.device)
                    env_desc_t = self._env_desc_from_ids(env_t)
                    text_t = self._text_tokens_from_ids(env_t, scenario_t)
                    z_next = self.agent.perception(
                        patch_next_t, H_next, scenario_t, env_desc_t, text_tokens=text_t
                    )
                    _, h_w, _, _ = self.agent.world_model.forward_step(
                        z_next, H_next, action, h_w
                    )

                patch = next_obs["patch"]
                energy = next_obs["energy"]
                env_id = int(next_obs.get("env_id", env_id))
                t += 1

            if len(rewards) == 0:
                continue

            T = len(rewards)
            patch_arr = np.stack(patches, axis=0)[None, ...]  # (1,T,p,p)
            H_arr = np.array(energies, dtype=np.float32).reshape(1, T, 1)
            a_arr = np.array(actions, dtype=np.int64).reshape(1, T)
            r_arr = np.array(rewards, dtype=np.float32).reshape(1, T)
            d_arr = np.array(dones, dtype=np.float32).reshape(1, T)
            death_arr = np.array(death_flags, dtype=np.float32).reshape(1, T)
            alive_arr = 1.0 - death_arr
            env_arr = np.array([env_id_ep] * T, dtype=np.int64).reshape(1, T)

            patch_t = torch.from_numpy(patch_arr).long().to(self.device)
            H_t = torch.from_numpy(H_arr).float().to(self.device)
            a_t = torch.from_numpy(a_arr).long().to(self.device)
            r_t = torch.from_numpy(r_arr).float().to(self.device)
            d_t = torch.from_numpy(d_arr).float().to(self.device)
            death_t = torch.from_numpy(death_arr).float().to(self.device)
            env_t_full = torch.from_numpy(env_arr).long().to(self.device)

            with torch.no_grad():
                B, T2, p, _ = patch_t.shape
                patch_flat = patch_t.view(B * T2, p, p)
                H_flat = H_t.view(B * T2, 1)
                scenario_np = np.full((1, T2), scenario_id, dtype=np.int64)
                env_np = np.full((1, T2), env_id_ep, dtype=np.int64)
                scenario_t = torch.from_numpy(scenario_np).long().to(self.device)
                env_all = torch.from_numpy(env_np).long().to(self.device)
                env_desc_all = self._env_desc_from_ids(env_all)
                scenario_flat = scenario_t.view(B * T2)
                env_flat = env_all.view(B * T2)
                env_desc_flat = (
                    env_desc_all.reshape(B * T2, -1) if env_desc_all is not None else None
                )

                text_flat = self._text_tokens_from_ids(env_flat, scenario_flat)
                z_flat = self.agent.perception(
                    patch_flat, H_flat, scenario_flat, env_desc_flat, text_tokens=text_flat
                )
                z_seq = z_flat.reshape(B, T2, -1)

                a_emb = self.agent.world_model.act_emb(a_t)
                x_w = torch.cat([z_seq, H_t, a_emb], dim=-1)
                h0 = torch.zeros(
                    1,
                    B,
                    self.agent.world_model.gru.hidden_size,
                    device=self.device,
                )
                out_w, _ = self.agent.world_model.gru(x_w, h0)
                W_seq = out_w

                M = self.agent.memory
                (
                    S_seq,
                    S_last,
                    surv_pred,
                    food_pred,
                    dmg_pred,
                    move_pred,
                    unc_pred,
                    surv_raw_pred,
                ) = self.agent.self_model.forward_seq(
                    W_seq,
                    H_t,
                    a_t,
                    r_t,
                    M=M,
                    env_desc=self._env_desc_from_ids(env_t_full),
                )

                w = self._combined_preference_weights()
                w_survive, w_food, w_danger, w_move = w.view(-1)
                R_unscaled = (
                    w_survive * surv_pred
                    + w_food * food_pred
                    + w_danger * dmg_pred
                    + w_move * move_pred
                )
                ret_pred = self.agent.self_model.head_return_calib(R_unscaled)

            # GT: survival flag (1 survived, 0 died) and discounted return
            D = d_arr[0]
            death_flags_arr = death_arr[0]

            done_idx = np.where(D > 0.5)[0]
            if done_idx.size > 0:
                t_done = int(done_idx[0])
                survived_flag = 0.0 if death_flags_arr[t_done] > 0.5 else 1.0
            else:
                survived_flag = float(alive_arr[0, -1]) if alive_arr.size > 0 else 1.0

            G_next = 0.0
            G_arr = np.zeros((T,), dtype=np.float32)
            for i in reversed(range(T)):
                G_next = r_arr[0, i] + gamma * G_next * (1.0 - d_arr[0, i])
                G_arr[i] = G_next

            true_R0 = float(G_arr[0])
            true_S0 = float(survived_flag)

            pred_R0 = float(ret_pred[0, 0, 0].cpu().item())
            pred_S0 = float(surv_pred[0, 0, 0].cpu().item())

            true_returns.append(true_R0)
            pred_returns.append(pred_R0)
            true_survival0.append(true_S0)
            pred_survival0.append(pred_S0)

        eps_std = float(getattr(self.safety, "std_eps", 1e-6))
        if len(true_returns) >= 2:
            tr = np.array(true_returns, dtype=np.float32)
            pr = np.array(pred_returns, dtype=np.float32)
            ts = np.array(true_survival0, dtype=np.float32)
            ps = np.array(pred_survival0, dtype=np.float32)

            def safe_corr(x: np.ndarray, y: np.ndarray, label: str) -> Tuple[float, str, bool]:
                if x.shape[0] < 2:
                    reason_local = "len<2"
                    logger.info(f"[SelfModel probe] corr_{label} undefined ({reason_local}), treating as 0.0")
                    return 0.0, reason_local, False
                sx = float(np.std(x))
                sy = float(np.std(y))
                variance_reason = f"std_x={sx:.3e}, std_y={sy:.3e}"
                if not math.isfinite(sx) or not math.isfinite(sy) or sx < eps_std or sy < eps_std:
                    note = "survival variance too low; corr_s treated as 0.0" if label == "s" else variance_reason
                    log_msg = (
                        f"[SelfModel probe] survival variance too low; corr_s treated as 0.0 ({variance_reason})"
                        if label == "s"
                        else f"[SelfModel probe] corr_{label} undefined ({variance_reason}), treating as 0.0"
                    )
                    logger.info(log_msg)
                    return 0.0, note, False
                corr_val = float(np.corrcoef(x, y)[0, 1])
                if not math.isfinite(corr_val):
                    reason_local = f"non-finite corr ({variance_reason})"
                    logger.info(
                        f"[SelfModel probe] corr_{label} non-finite ({variance_reason}), treating as 0.0"
                    )
                    return 0.0, reason_local, False
                return corr_val, "", True

            corr_ret, reason_ret, corr_ret_defined = safe_corr(tr, pr, label="r")
            corr_surv, reason_surv, corr_surv_defined = safe_corr(ts, ps, label="s")

            print("SelfModel probe:")
            print(
                f"  corr(return_true, return_pred) ~= {corr_ret:.3f}"
                + (f" (note: {reason_ret})" if reason_ret else "")
            )
            print(
                f"  corr(survival_true, survival_pred) ~= {corr_surv:.3f}"
                + (f" (note: {reason_surv})" if reason_surv else "")
            )
            print(f"  mean true R0={tr.mean():.3f}, mean pred R0={pr.mean():.3f}")
            print(f"  mean true S={ts.mean():.3f}, mean pred S={ps.mean():.3f}")

            metrics = {
                "n_episodes_used": int(len(true_returns)),
                "corr_return": float(corr_ret),
                "corr_return_defined": bool(corr_ret_defined),
                "corr_return_note": reason_ret,
                "corr_survival": float(corr_surv),
                "corr_survival_defined": bool(corr_surv_defined),
                "corr_survival_note": reason_surv,
                "mean_true_return0": float(tr.mean()),
                "mean_pred_return0": float(pr.mean()),
                "mean_true_survival": float(ts.mean()),
                "mean_pred_survival": float(ps.mean()),
            }
        else:
            reason = "len<2"
            logger.info(f"[SelfModel probe] correlations undefined ({reason}), treating as 0.0")
            print("SelfModel probe: insufficient episodes for correlations (treated as 0.0)")
            metrics = {
                "n_episodes_used": int(len(true_returns)),
                "corr_return": 0.0,
                "corr_return_defined": False,
                "corr_return_note": reason,
                "corr_survival": 0.0,
                "corr_survival_defined": False,
                "corr_survival_note": reason,
                "mean_true_return0": 0.0,
                "mean_pred_return0": 0.0,
                "mean_true_survival": 0.0,
                "mean_pred_survival": 0.0,
            }

        stats = SelfModelProbeStats(
            corr_return=float(metrics["corr_return"]),
            corr_return_defined=bool(metrics.get("corr_return_defined", True)),
            corr_survival=float(metrics["corr_survival"]),
            corr_survival_defined=bool(metrics.get("corr_survival_defined", True)),
            mean_true_return=float(metrics["mean_true_return0"]),
            mean_pred_return=float(metrics["mean_pred_return0"]),
            mean_true_survival=float(metrics["mean_true_survival"]),
            mean_pred_survival=float(metrics["mean_pred_survival"]),
            num_samples=int(metrics["n_episodes_used"]),
        )
        self.last_self_probe = stats
        if self.logger is not None:
            payload = {
                "event": "self_model_probe",
                "use_self_for_probe": bool(use_self),
                **asdict(stats),
                "corr_return_note": metrics.get("corr_return_note"),
                "corr_survival_note": metrics.get("corr_survival_note"),
            }
            self.logger.log(payload)
        return stats

    def probe_self_model(
        self,
        n_episodes: int = 30,
        max_steps: int = 200,
        gamma: float = 0.99,
        use_self: bool = False,
    ):
        stats = self.run_selfmodel_probe(
            n_episodes=n_episodes,
            max_steps=max_steps,
            gamma=gamma,
            use_self=use_self,
        )
        return asdict(stats)
