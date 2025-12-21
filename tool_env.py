"""
Simple arithmetic environment ("ToolEnv") with a small, discrete action space.
The agent must adjust an integer memory toward a randomly sampled target using
increment/decrement/reset operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class ToolEnvConfig:
    target_min: int = 0
    target_max: int = 20
    max_steps: int = 32
    step_penalty: float = -1.0
    success_reward: float = 25.0
    memory_min: Optional[int] = None
    memory_max: Optional[int] = None
    patch_size: int = 5  # keep compatible with grid-based perception


class ToolEnv:
    """
    Minimal arithmetic environment with gym-like API.

    State:
        memory: current integer value
        target: target integer to reach
        steps_left: remaining steps before episode termination

    Actions:
        0: NOOP
        1: INC (memory += 1)
        2: DEC (memory -= 1)
        3: RESET (memory = 0)
    """

    ACTIONS = ["NOOP", "INC", "DEC", "RESET", "STAY", "TAKE"]

    def __init__(self, config: ToolEnvConfig, env_id: int = 0, env_name: str = "tool_env", seed: Optional[int] = None):
        self.config = config
        self._env_id = int(env_id)
        self._env_name = env_name
        self.env_family = "tools"
        self.rng = np.random.RandomState(seed)
        self.n_cell_types = len(self.ACTIONS)
        self.view_size = config.patch_size
        self.n_scenarios = 1
        self.current_scenario_id = 0
        self.current_scenario_name = "tools_basic"

        self.memory: int = 0
        self.target: int = 0
        self.steps_left: int = config.max_steps

    # --------- BaseEnv compatibility ---------
    @property
    def env_id(self) -> int:
        return self._env_id

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    @property
    def action_meanings(self):
        return tuple(self.ACTIONS)

    @property
    def max_steps(self) -> int:
        return self.config.max_steps

    def sample_random_action(self) -> int:
        return int(self.rng.randint(0, self.n_actions))

    # --------- Core env logic ---------
    def reset(self, seed: Optional[int] = None, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        if scenario_id is not None:
            self.current_scenario_id = int(scenario_id)
        self.target = int(self.rng.randint(self.config.target_min, self.config.target_max + 1))
        self.memory = 0
        self.steps_left = int(self.config.max_steps)
        return self._get_obs()

    def _encode_state_token(self) -> int:
        if self.memory == self.target:
            return 3  # success / aligned
        if self.memory < self.target:
            return 1  # need to increment
        return 2  # need to decrement

    def _get_obs(self) -> Dict[str, Any]:
        token = self._encode_state_token()
        patch = np.full(
            (self.view_size, self.view_size),
            fill_value=token,
            dtype=np.int64,
        )
        scale = max(abs(self.config.target_min), abs(self.config.target_max), 1)
        energy_norm = float(self.memory) / float(scale)
        return {
            "patch": patch,
            "energy": energy_norm,
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
            "memory": self.memory,
            "target": self.target,
            "steps_left": self.steps_left,
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            action = 0  # fallback to NOOP for out-of-range actions
        reward = float(self.config.step_penalty)
        done = False
        info: Dict[str, Any] = {
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
        }

        if action == 1:  # INC
            self.memory += 1
        elif action == 2:  # DEC
            self.memory -= 1
        elif action == 3:  # RESET
            self.memory = 0
        else:
            # actions 0,4,5 are treated as NOOP/STAY/TAKE in this arithmetic setting
            pass

        if self.config.memory_min is not None:
            self.memory = max(self.memory, int(self.config.memory_min))
        if self.config.memory_max is not None:
            self.memory = min(self.memory, int(self.config.memory_max))

        self.steps_left -= 1

        if self.memory == self.target:
            reward += float(self.config.success_reward)
            done = True
            info["reason"] = "reached_target"
        elif self.steps_left <= 0:
            done = True
            info["reason"] = "max_steps"

        obs = self._get_obs()
        info.update(
            {
                "memory": self.memory,
                "target": self.target,
                "steps_left": self.steps_left,
                "reward_env": reward,
            }
        )
        return obs, reward, done, info

    def get_obs_spec(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "env_name": self.env_name,
            "patch_size": self.view_size,
            "n_cell_types": self.n_cell_types,
            "n_scenarios": self.n_scenarios,
            "obs_fields": ["patch", "energy", "scenario_id", "env_id"],
            "env_descriptor": self.get_env_descriptor(),
        }

    def get_env_descriptor(self) -> np.ndarray:
        """
        Descriptor mirrors core scalar parameters for embedding/conditioning.
        Layout: [target_min, target_max, max_steps, step_penalty, success_reward]
        """
        return np.array(
            [
                float(self.config.target_min),
                float(self.config.target_max),
                float(self.config.max_steps),
                float(self.config.step_penalty),
                float(self.config.success_reward),
            ],
            dtype=np.float32,
        )
