from __future__ import annotations

import numpy as np
import torch

from agent import ProtoCreatureAgent
from experiment import _build_env_pool
from trainer import Trainer


def _build_trainer() -> Trainer:
    pool = _build_env_pool(seed=0, schedule_mode="iid", episodes_per_phase=4, max_steps_env=12)
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
    return Trainer(
        env=pool,
        agent=agent,
        env_descriptors=env_descriptors,
        enable_risk_shield=True,
        risk_shield_threshold=0.5,
    )


def test_risk_shield_blocks_unsafe_actions():
    trainer = _build_trainer()
    logits = torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float32)
    risk_c = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32)  # action 1 unsafe
    mask = torch.tensor([[True, True, True]])
    out, blocked, _mx = trainer._apply_risk_shield(logits, risk_c, hard_mask=mask)
    assert blocked >= 1
    assert out[0, 1].item() < -1.0e8


def test_risk_shield_no_safe_action_falls_back_to_original_logits():
    trainer = _build_trainer()
    logits = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    risk_c = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])
    out, blocked, _mx = trainer._apply_risk_shield(logits, risk_c, hard_mask=mask)
    assert blocked == 0
    assert torch.allclose(out, logits)

