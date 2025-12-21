import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer import Trainer


class _DummySafety:
    safety_threshold = 0.5
    safety_penalty_coef = 2.0


def test_safety_penalty_prefers_safer_action():
    scores_main = torch.tensor([1.0, 0.6])
    scores_safety = torch.tensor([0.1, 0.6])
    penalized = Trainer._apply_safety_penalty(_DummySafety(), scores_main, scores_safety)
    assert torch.argmax(penalized).item() == 1
    assert penalized[1] > penalized[0]
