from types import SimpleNamespace

import numpy as np
import torch

from skills import SkillContext, get_default_skills, LatentSkill


def _dummy_context(step: int = 0) -> SkillContext:
    patch = np.zeros((5, 5), dtype=np.int64)
    return SkillContext(
        w=torch.zeros(1, 4),
        h=torch.zeros(1, 1),
        traits=torch.zeros(1, 4),
        env_desc=torch.zeros(1, 3),
        step_in_skill=step,
        extra={"patch": patch},
    )


def test_handcrafted_skill_interface():
    skills = get_default_skills()
    skill = skills[0]
    skill.reset()
    ctx = _dummy_context(step=0)
    action = skill.step(ctx)
    assert isinstance(action, int)
    assert skill.step_in_skill == 1
    assert skill.is_complete() is False


def test_latent_skill_policy():
    ctx = _dummy_context(step=0)
    ctx_dim = ctx.w.numel() + ctx.h.numel() + ctx.env_desc.numel() + ctx.traits.numel() + 1
    cfg = SimpleNamespace(latent_skill_hidden_dim=8, skill_horizon=2, skill_id=0, name="LATENT_TEST")
    skill = LatentSkill(config=cfg, latent_dim=4, num_actions=6, context_dim=ctx_dim)
    skill.reset()
    action = skill.step(ctx)
    assert isinstance(action, int)
    assert skill.step_in_skill == 1
    # second step should advance counter
    action2 = skill.step(ctx)
    assert isinstance(action2, int)
    assert skill.step_in_skill == 2
