import torch

from agent import ProtoCreatureAgent
from checkpointing import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_restores_agent_state(tmp_path):
    torch.manual_seed(0)
    env_desc = torch.zeros(1, 10, dtype=torch.float32)
    agent = ProtoCreatureAgent(
        n_cell_types=5,
        n_scenarios=1,
        env_descriptors=env_desc,
        device=torch.device("cpu"),
        n_actions=6,
        use_skills=False,
    )

    before = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    ckpt = tmp_path / "agent_ckpt.pt"
    save_checkpoint(ckpt, agent, save_optim=True)

    with torch.no_grad():
        for p in agent.parameters():
            if p.numel() > 0:
                p.add_(0.01 * torch.randn_like(p))

    load_checkpoint(ckpt, agent, load_optim=True, strict=True)
    after = agent.state_dict()

    assert set(after.keys()) == set(before.keys())
    for key, tensor_before in before.items():
        tensor_after = after[key]
        assert torch.allclose(tensor_after, tensor_before)

