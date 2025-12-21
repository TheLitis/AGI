import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import traits_to_preference_weights


def test_traits_to_preference_weights_shapes():
    single = torch.zeros(4)
    weights_single = traits_to_preference_weights(single)
    assert weights_single.shape == (4,)

    stacked = torch.zeros(2, 4)
    weights_stacked = traits_to_preference_weights(stacked)
    assert weights_stacked.shape == (2, 4)


def test_traits_to_preference_weights_rowwise():
    stacked = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    weights = traits_to_preference_weights(stacked)
    first_row = traits_to_preference_weights(stacked[0])
    assert torch.allclose(weights[0], first_row)
    assert not torch.allclose(weights[0], weights[1])
