import pytest
import torch

from trainer import Trainer


def _trainer_stub() -> Trainer:
    t = Trainer.__new__(Trainer)
    t.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t.validate_device_indices = False
    return t


def test_env_desc_lookup_defaults_to_fast_path():
    trainer = _trainer_stub()
    trainer.env_descriptors = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=trainer.device)

    env_ids = torch.tensor([0, 1], dtype=torch.long, device=trainer.device)

    out = trainer._env_desc_from_ids(env_ids)

    assert out.shape == (2, 2)
    assert out.device == trainer.env_descriptors.device


def test_env_desc_lookup_debug_validation_catches_bad_ids():
    trainer = _trainer_stub()
    trainer.validate_device_indices = True
    trainer.env_descriptors = torch.tensor([[1.0, 2.0]], device=trainer.device)

    with pytest.raises(RuntimeError, match="env_id out of range"):
        trainer._env_desc_from_ids(torch.tensor([1], dtype=torch.long, device=trainer.device))


def test_text_token_lookup_debug_validation_catches_bad_scenarios():
    trainer = _trainer_stub()
    trainer.validate_device_indices = True
    trainer.text_token_table = torch.zeros(1, 1, 3, dtype=torch.long, device=trainer.device)

    with pytest.raises(RuntimeError, match="scenario_id out of range"):
        trainer._text_tokens_from_ids(
            torch.tensor([0], dtype=torch.long, device=trainer.device),
            torch.tensor([1], dtype=torch.long, device=trainer.device),
        )
