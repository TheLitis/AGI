import torch

from models import WorldModel


def test_world_model_event_heads_predict_probabilities_and_reward():
    model = WorldModel(obs_dim=16, h_dim=1, n_actions=4, act_emb_dim=4, w_dim=12)
    w_t = torch.randn(7, 12)
    preds = model.predict_event_components(w_t)
    assert set(preds.keys()) == {"alive", "food", "damage", "move", "reward_env"}
    for k in ("alive", "food", "damage", "move"):
        t = preds[k]
        assert tuple(t.shape) == (7, 1)
        assert torch.isfinite(t).all()
        assert torch.all((t >= 0.0) & (t <= 1.0))
    reward_env = preds["reward_env"]
    assert tuple(reward_env.shape) == (7, 1)
    assert torch.isfinite(reward_env).all()


def test_world_model_loss_supervised_accepts_event_targets():
    torch.manual_seed(0)
    model = WorldModel(obs_dim=10, h_dim=1, n_actions=5, act_emb_dim=4, w_dim=8)
    b, t = 4, 6
    z_obs_seq = torch.randn(b, t, 10)
    H_seq = torch.randn(b, t, 1)
    a_seq = torch.randint(low=0, high=5, size=(b, t))

    event_targets = {
        "alive": torch.rand(b, t - 1),
        "food": torch.rand(b, t - 1),
        "damage": torch.rand(b, t - 1),
        "move": torch.rand(b, t - 1),
        "reward": torch.randn(b, t - 1),
    }

    loss = model.loss_supervised(
        z_obs_seq=z_obs_seq,
        H_seq=H_seq,
        a_seq=a_seq,
        event_targets=event_targets,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
