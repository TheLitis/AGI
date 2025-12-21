"""
Skill definitions for hierarchical control (hand-crafted + learnable options).

Skills now operate on a unified SkillContext and expose a consistent interface
for both hard-coded heuristics and learnable latent skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from types import SimpleNamespace
from abc import ABC, abstractmethod
import random

import torch
import torch.nn as nn


@dataclass
class SkillContext:
    """
    Structured context visible to a skill at each primitive step.

    Attributes:
        w: world-model latent (W_t)
        h: body/energy state (H_t)
        traits: current trait vector for the active faction
        env_desc: environment descriptor embedding
        step_in_skill: local step counter for this skill instance
        extra: optional raw observation payload (patch/energy/etc) for heuristics
    """

    w: torch.Tensor
    h: torch.Tensor
    traits: torch.Tensor
    env_desc: torch.Tensor
    step_in_skill: int
    extra: Optional[Dict[str, Any]] = None


@dataclass
class SkillDemoTransition:
    """
    Single (context -> action) pair captured from a hand-crafted skill rollout.
    """

    context: torch.Tensor
    action: int


@dataclass
class SkillDemoBatch:
    """Mini-batch of skill demonstrations for supervised learning."""

    contexts: torch.Tensor  # [B, D]
    actions: torch.Tensor  # [B]


def flatten_skill_context(ctx: SkillContext, max_horizon: Optional[float] = None) -> torch.Tensor:
    """
    Convert SkillContext into a flat 1D feature vector.

    Order is fixed: [w | h | env_desc | traits | step_in_skill_norm].
    This mirrors Agent.skill_context_dim calculation and LatentSkill expectations.
    """
    parts: List[torch.Tensor] = []
    device = None
    for field in (ctx.w, ctx.h, ctx.env_desc, ctx.traits):
        if isinstance(field, torch.Tensor):
            tensor_flat = field.reshape(-1).float()
            parts.append(tensor_flat)
            device = device or tensor_flat.device
    if device is None:
        device = torch.device("cpu")
    max_h = float(max_horizon) if max_horizon is not None else 1.0
    max_h = max(max_h, 1.0)
    step_norm = float(ctx.step_in_skill) / max_h
    parts.append(torch.tensor([step_norm], device=device, dtype=torch.float32))
    if not parts:
        raise ValueError("flatten_skill_context received an empty SkillContext")
    return torch.cat(parts, dim=-1)


class Skill(ABC):
    """
    Unified interface for both hand-crafted and learnable skills/options.
    Each skill maintains its own internal step counter.
    """

    def __init__(self, name: str, horizon: int = 1, skill_id: Optional[int] = None):
        self.name = name
        self.horizon = int(max(1, horizon))
        self.skill_id = int(skill_id) if skill_id is not None else None
        self.step_in_skill: int = 0

    def reset(self):
        """Reset internal counters before executing this skill."""
        self.step_in_skill = 0

    def is_complete(self) -> bool:
        """Return True when the skill exhausted its horizon."""
        return self.step_in_skill >= self.horizon

    @abstractmethod
    def step(self, context: SkillContext) -> int:
        """
        Produce the next primitive action id given the structured context.
        """

    def rollout(self, context: Optional[SkillContext]) -> List[int]:
        """Generate a short action sequence by iterating step() up to the horizon."""
        if context is None:
            return []
        self.reset()
        actions: List[int] = []
        current_ctx = context
        for _ in range(self.horizon):
            action = self.step(current_ctx)
            actions.append(int(action))
            if self.is_complete():
                break
            current_ctx = SkillContext(
                w=current_ctx.w,
                h=current_ctx.h,
                traits=current_ctx.traits,
                env_desc=current_ctx.env_desc,
                step_in_skill=self.step_in_skill,
                extra=current_ctx.extra,
            )
        return actions


class GoToFoodSkill(Skill):
    """Greedy move toward food if observed; otherwise explore randomly."""

    def __init__(self, skill_id: int, horizon: int = 3):
        super().__init__("GO_TO_FOOD", horizon=horizon, skill_id=skill_id)

    def step(self, context: SkillContext) -> int:
        patch = None
        if context.extra is not None:
            patch = context.extra.get("patch")
        if patch is None:
            action = random.randint(0, 3)
        else:
            p = patch.shape[0]
            center = p // 2
            food_positions = [(i, j) for i in range(p) for j in range(p) if patch[i, j] == 2]
            if food_positions:
                target = min(food_positions, key=lambda pos: abs(pos[0] - center) + abs(pos[1] - center))
                di = target[0] - center
                dj = target[1] - center
                if abs(di) >= abs(dj):
                    action = 0 if di < 0 else 1  # UP if food above else DOWN
                else:
                    action = 2 if dj < 0 else 3  # LEFT if food left else RIGHT
            else:
                action = random.randint(0, 3)
        self.step_in_skill += 1
        return int(action)


class ExploreSkill(Skill):
    """Random walk for the duration of the horizon."""

    def __init__(self, skill_id: int, horizon: int = 3):
        super().__init__("EXPLORE", horizon=horizon, skill_id=skill_id)

    def step(self, context: SkillContext) -> int:
        action = random.randint(0, 3)
        self.step_in_skill += 1
        return int(action)


class StaySafeSkill(Skill):
    """Move away from danger tiles if seen; otherwise stay in place."""

    def __init__(self, skill_id: int, horizon: int = 3):
        super().__init__("STAY_SAFE", horizon=horizon, skill_id=skill_id)

    def step(self, context: SkillContext) -> int:
        patch = None
        if context.extra is not None:
            patch = context.extra.get("patch")
        if patch is None:
            action = 4  # STAY
        else:
            p = patch.shape[0]
            center = p // 2
            danger_positions = [(i, j) for i in range(p) for j in range(p) if patch[i, j] == 3]
            if danger_positions:
                target = min(danger_positions, key=lambda pos: abs(pos[0] - center) + abs(pos[1] - center))
                di = target[0] - center
                dj = target[1] - center
                if abs(di) >= abs(dj):
                    action = 1 if di < 0 else 0  # move opposite vertical direction
                else:
                    action = 3 if dj < 0 else 2
            else:
                action = 4  # stay
        self.step_in_skill += 1
        return int(action)


class LatentSkill(Skill, nn.Module):
    """
    Parameterized latent skill: a small policy conditioned on context + learnable code.
    """

    def __init__(
        self,
        config: Any,
        latent_dim: int,
        num_actions: int,
        context_dim: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        Skill.__init__(
            self,
            name=getattr(config, "name", f"LATENT_{getattr(config, 'skill_id', 0)}"),
            horizon=int(getattr(config, "skill_horizon", getattr(config, "horizon", 1))),
            skill_id=getattr(config, "skill_id", None),
        )
        self.latent = nn.Parameter(torch.zeros(latent_dim))
        self.num_actions = int(num_actions)
        self.hidden_dim = int(getattr(config, "latent_skill_hidden_dim", 64))
        self.context_dim = context_dim
        self.net: Optional[nn.Sequential] = None
        if context_dim is not None:
            self._init_net(context_dim, latent_dim)

    def _init_net(self, context_dim: int, latent_dim: int):
        self.net = nn.Sequential(
            nn.Linear(context_dim + latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions),
        )

    def _flatten_context(self, context: SkillContext) -> torch.Tensor:
        """Concatenate context components into a single 1D tensor."""
        return flatten_skill_context(context, max_horizon=self.horizon)

    def step(self, context: SkillContext) -> int:
        device = context.w.device
        if self.latent.device != device:
            self.to(device)
        x_ctx = self._flatten_context(context)
        if self.net is None:
            inferred_dim = x_ctx.numel()
            self.context_dim = inferred_dim
            self._init_net(inferred_dim, self.latent.shape[-1])
            self.to(device)
        logits = self.net(torch.cat([x_ctx, self.latent.to(device)], dim=-1))
        action = torch.distributions.Categorical(logits=logits).sample().item()
        self.step_in_skill += 1
        return int(action)

    def forward_logits(self, contexts: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass over flattened contexts.

        Args:
            contexts: [B, D] float tensor of flattened SkillContext features.
        Returns:
            logits over primitive actions with shape [B, num_actions].
        """
        if self.net is None:
            inferred_dim = int(contexts.shape[-1])
            self.context_dim = inferred_dim
            self._init_net(inferred_dim, self.latent.shape[-1])
        device = contexts.device
        if self.latent.device != device:
            self.to(device)
        latent = self.latent.to(device).unsqueeze(0).expand(contexts.shape[0], -1)
        return self.net(torch.cat([contexts, latent], dim=-1))


class SkillLibrary(nn.Module):
    """Container for a learnable set of latent skills."""

    def __init__(self, config: Any, num_actions: int, context_dim: Optional[int] = None):
        super().__init__()
        num_latent = int(getattr(config, "num_latent_skills", getattr(config, "n_latent_skills", 0)))
        latent_dim = int(getattr(config, "latent_dim", getattr(config, "latent_skill_dim", 16)))
        hidden_dim = int(getattr(config, "latent_skill_hidden_dim", 64))
        horizon = int(getattr(config, "skill_horizon", getattr(config, "horizon", 1)))

        # bake minimal config template for each skill
        self.skills = nn.ModuleList()
        for idx in range(num_latent):
            skill_cfg = SimpleNamespace(
                latent_skill_hidden_dim=hidden_dim,
                skill_id=idx,
                skill_horizon=horizon,
                name=f"LATENT_{idx}",
            )
            self.skills.append(
                LatentSkill(
                    config=skill_cfg,
                    latent_dim=latent_dim,
                    num_actions=num_actions,
                    context_dim=context_dim,
                )
            )

    def __len__(self) -> int:
        return len(self.skills)

    def get_skill(self, idx: int) -> Skill:
        return self.skills[idx]


def get_default_skills(start_id: int = 0) -> List[Skill]:
    """Default set of hand-crafted skills used across experiments."""
    return [
        GoToFoodSkill(start_id + 0, horizon=3),
        ExploreSkill(start_id + 1, horizon=3),
        StaySafeSkill(start_id + 2, horizon=3),
    ]
