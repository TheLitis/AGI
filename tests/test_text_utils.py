import numpy as np
import torch

from models import Perception
from text_utils import hash_text_to_ids


def test_hash_text_to_ids_is_deterministic_and_padded():
    ids1 = hash_text_to_ids("Hello world", max_len=5, vocab_size=101)
    ids2 = hash_text_to_ids("Hello world", max_len=5, vocab_size=101)
    assert isinstance(ids1, np.ndarray)
    assert ids1.shape == (5,)
    assert ids1.dtype == np.int64
    assert np.array_equal(ids1, ids2)
    assert ids1[0] != 0
    assert ids1[1] != 0
    assert np.array_equal(ids1[2:], np.zeros((3,), dtype=np.int64))


def test_perception_accepts_text_tokens():
    model = Perception(
        n_cell_types=5,
        patch_size=5,
        h_dim=1,
        hidden_dim=32,
        n_scenarios=3,
        scenario_emb_dim=4,
        env_desc_dim=10,
        env_emb_dim=4,
        text_vocab_size=101,
        text_emb_dim=8,
        text_max_len=5,
    )
    patch = torch.zeros(2, 5, 5, dtype=torch.long)
    H = torch.zeros(2, 1, dtype=torch.float32)
    scenario = torch.zeros(2, dtype=torch.long)
    env_desc = torch.zeros(2, 10, dtype=torch.float32)
    text = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 5, 0, 0]], dtype=torch.long)

    out = model(patch, H, scenario, env_desc, text_tokens=text)
    assert out.shape == (2, 32)

