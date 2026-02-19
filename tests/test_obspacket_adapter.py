from __future__ import annotations

import numpy as np

from interface_adapters import ObsPacket, obs_to_packet, packet_to_obs


def test_obspacket_roundtrip_patch_and_energy():
    obs = {
        "patch": np.arange(25, dtype=np.int64).reshape(5, 5),
        "energy": 0.75,
        "scenario_id": 3,
        "env_id": 7,
        "env_family": "gridworld-basic",
        "env_name": "grid_a",
    }
    packet = obs_to_packet(obs, episode_id="ep1", step_id=2, split="test")
    assert isinstance(packet, ObsPacket)
    assert packet.tokens.shape == (25,)
    assert packet.token_types.shape == (25,)
    assert packet.token_mask.shape == (25,)
    restored = packet_to_obs(packet, patch_shape=(5, 5))
    assert restored["scenario_id"] == 3
    assert restored["env_id"] == 7
    assert restored["env_family"] == "gridworld-basic"
    assert float(restored["energy"]) == 0.75
    assert np.array_equal(restored["patch"], obs["patch"])

