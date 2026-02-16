import numpy as np
import torch

from agent import ProtoCreatureAgent
from experiment import _build_env_pool
from trainer import Trainer


def _build_trainer_for_mode(mode: str) -> tuple[Trainer, ProtoCreatureAgent, dict]:
    pool = _build_env_pool(seed=0, schedule_mode="iid", episodes_per_phase=5, max_steps_env=20)
    env_desc_np = [e.get_env_descriptor() for e in pool.envs]
    max_len = max(d.shape[0] for d in env_desc_np)
    env_desc_np = [np.pad(d, (0, max_len - d.shape[0]), constant_values=0.0) for d in env_desc_np]
    env_descriptors = torch.tensor(np.stack(env_desc_np), dtype=torch.float32)
    agent = ProtoCreatureAgent(
        n_cell_types=pool.n_cell_types,
        n_scenarios=pool.n_scenarios,
        env_descriptors=env_descriptors,
        device=torch.device("cpu"),
        n_actions=pool.n_actions,
    )
    trainer = Trainer(
        env=pool,
        agent=agent,
        planning_horizon=5,
        planner_gamma=0.99,
        planner_mode=mode,
        planner_rollouts=3,
        train_env_ids=pool.train_env_ids,
        test_env_ids=pool.test_env_ids,
        env_descriptors=env_descriptors,
    )
    obs = pool.reset(split="train")
    return trainer, agent, obs


def _planner_logits(trainer: Trainer, agent: ProtoCreatureAgent, obs: dict) -> torch.Tensor:
    patch_t = torch.from_numpy(obs["patch"]).unsqueeze(0).long()
    H_t = torch.tensor([[float(obs["energy"])]], dtype=torch.float32)
    scenario_t = torch.tensor([int(obs.get("scenario_id", 0))], dtype=torch.long)
    env_t = torch.tensor([int(obs.get("env_id", 0))], dtype=torch.long)
    env_desc_t = trainer._env_desc_from_ids(env_t)
    text_t = trainer._text_tokens_from_ids(env_t, scenario_t)
    z_obs = agent.perception(patch_t, H_t, scenario_t, env_desc_t, text_tokens=text_t)
    h_w = torch.zeros(1, 1, agent.w_dim, dtype=torch.float32)
    traits = trainer._mixed_traits()
    M = agent.memory.clone()
    return trainer._get_planner_logits(z_obs=z_obs, H_t=H_t, h_w=h_w, traits=traits, M=M)


def test_repeat_planner_logits_shape_and_finite():
    trainer, agent, obs = _build_trainer_for_mode("repeat")
    logits = _planner_logits(trainer, agent, obs)
    assert tuple(logits.shape) == (1, trainer.env.n_actions)
    assert torch.isfinite(logits).all()


def test_rollout_planner_logits_shape_and_finite():
    trainer, agent, obs = _build_trainer_for_mode("rollout")
    logits = _planner_logits(trainer, agent, obs)
    assert tuple(logits.shape) == (1, trainer.env.n_actions)
    assert torch.isfinite(logits).all()
