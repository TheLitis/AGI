import numpy as np
import torch

import trainer as trainer_module
from agent import ProtoCreatureAgent
from env import EnvPool, GridWorldEnv
from memory import Transition
from trainer import Trainer


def test_self_model_survival_target_stays_in_unit_interval(monkeypatch):
    """
    Regression test for CUDA BCE asserts:
    survival targets must be in [0,1] even when EnvPool.max_steps is smaller
    than sampled sequence length (common in mixed pools).
    """

    env_short = GridWorldEnv(max_steps=30, seed=0, env_id=0, env_name="short", multi_task=False)
    env_long = GridWorldEnv(max_steps=50, seed=1, env_id=1, env_name="long", multi_task=False)
    env_pool = EnvPool(envs=[env_short, env_long], schedule_mode="iid", seed=0)

    env_desc_np = [env.get_env_descriptor() for env in env_pool.envs]
    env_descriptors = torch.tensor(np.stack(env_desc_np), dtype=torch.float32)

    agent = ProtoCreatureAgent(
        n_cell_types=env_pool.n_cell_types,
        n_scenarios=env_pool.n_scenarios,
        env_descriptors=env_descriptors,
        device=torch.device("cpu"),
        n_actions=env_pool.n_actions,
    )
    trainer = Trainer(
        env=env_pool,
        agent=agent,
        device=torch.device("cpu"),
        env_descriptors=env_descriptors,
    )

    seq_len = 32
    patch = np.zeros((5, 5), dtype=np.int64)
    for t in range(seq_len):
        done = t == (seq_len - 1)
        trainer.buffer.push(
            Transition(
                obs_patch=patch.copy(),
                energy=1.0,
                action=0,
                reward=0.0,
                done=bool(done),
                next_obs_patch=patch.copy(),
                next_energy=1.0,
                death_flag=1.0 if done else 0.0,
                got_food=0.0,
                took_damage=0.0,
                moved=0.0,
                alive=0.0 if done else 1.0,
                scenario_id=0,
                env_id=1,
                regime_name="unit_test",
            )
        )

    orig_bce = trainer_module.F.binary_cross_entropy

    def checked_bce(inp, target, *args, **kwargs):
        assert float(target.min().item()) >= 0.0
        assert float(target.max().item()) <= 1.0
        return orig_bce(inp, target, *args, **kwargs)

    monkeypatch.setattr(trainer_module.F, "binary_cross_entropy", checked_bce)

    trainer._train_self_model_from_buffer(batch_size=1, seq_len=seq_len, n_batches=1)

