"""
Instruction-conditioned environment to start tackling the "Language" mountain.

This env intentionally keeps the action space fixed at 6 (compatible with EnvPool),
but makes the reward depend on the scenario description ("instruction").

The agent sees the instruction only via Trainer's hashed text conditioning
(scenario name/description), not via extra observation fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env import BaseEnv, build_env_descriptor


@dataclass
class InstructionEnvConfig:
    size: int = 7
    view_size: int = 5
    max_steps: int = 50
    step_penalty: float = -0.01
    progress_reward: float = 0.05
    success_reward: float = 1.0
    wrong_reward: float = -1.0
    spawn_both_goals: bool = True


class InstructionEnv(BaseEnv):
    """
    Two-goal grid with scenario-conditioned instruction:
      - Scenario 0: "Collect goal A"
      - Scenario 1: "Collect goal B"

    Actions (size=6):
      0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY, 5: TAKE
    """

    EMPTY = 0
    WALL = 1
    GOAL_A = 2
    GOAL_B = 3

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "TAKE"]

    def __init__(
        self,
        config: Optional[InstructionEnvConfig] = None,
        env_id: int = 0,
        env_name: str = "instruction_env",
        seed: Optional[int] = None,
    ):
        self.config = config or InstructionEnvConfig()
        self._env_id = int(env_id)
        self._env_name = str(env_name)
        self.env_family = "instruction-basic"
        self.rng = np.random.RandomState(seed)

        self.view_size = int(self.config.view_size)
        if self.view_size % 2 != 1:
            raise ValueError("InstructionEnv view_size must be odd")

        self.n_cell_types = 4
        self.n_scenarios = 2
        self.scenario_configs: List[Dict[str, Any]] = [
            {
                "name": "collect_goal_a",
                "description": "Instruction: go to goal A and TAKE it.",
                "target": "A",
            },
            {
                "name": "collect_goal_b",
                "description": "Instruction: go to goal B and TAKE it.",
                "target": "B",
            },
        ]
        self.current_scenario_id: int = 0
        self.current_scenario_name: str = str(self.scenario_configs[0]["name"])
        self.description: str = "Instruction-conditioned two-goal grid."

        self.grid: Optional[np.ndarray] = None
        self.agent_pos: List[int] = [0, 0]
        self.steps: int = 0
        self._env_descriptor = self._compute_env_descriptor()

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
        return int(self.config.max_steps)

    def sample_random_action(self) -> int:
        return int(self.rng.randint(0, self.n_actions))

    def get_expert_action(self) -> Optional[int]:
        """
        Deterministic oracle policy for BC/online-BC.
        Move toward instructed goal with Manhattan-greedy steps, then TAKE.
        """
        if self.grid is None:
            return None
        target = str(self.scenario_configs[self.current_scenario_id].get("target", "A")).upper()
        size = int(self.grid.shape[0])
        tx, ty = (1, 1) if target == "A" else (size - 2, size - 2)
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])

        if ax == tx and ay == ty:
            return 5  # TAKE

        candidates: List[int] = []
        if ax > tx:
            candidates.append(0)  # UP
        elif ax < tx:
            candidates.append(1)  # DOWN
        if ay > ty:
            candidates.append(2)  # LEFT
        elif ay < ty:
            candidates.append(3)  # RIGHT

        for action in candidates:
            nx, ny = ax, ay
            if action == 0:
                nx -= 1
            elif action == 1:
                nx += 1
            elif action == 2:
                ny -= 1
            elif action == 3:
                ny += 1
            if int(self.grid[nx, ny]) != self.WALL:
                return int(action)
        return 4  # STAY

    # --------- env descriptor / spec ---------
    def _compute_env_descriptor(self) -> np.ndarray:
        size = float(int(self.config.size))
        n_goals = 2.0 if bool(self.config.spawn_both_goals) else 1.0
        goal_density = float(n_goals / max(1.0, size * size))
        wall_density = float(4.0 * size / max(1.0, size * size))
        return build_env_descriptor(
            env_family=self.env_family,
            width=size,
            height=size,
            goal_density=goal_density,
            danger_density=0.0,
            wall_density=wall_density,
            has_door=False,
            has_key=False,
            has_lava=False,
            max_steps=float(self.config.max_steps),
        )

    def get_env_descriptor(self) -> np.ndarray:
        return np.array(self._env_descriptor, copy=True)

    def get_obs_spec(self) -> Dict[str, Any]:
        return {
            "env_id": int(self.env_id),
            "env_name": str(self.env_name),
            "patch_size": int(self.view_size),
            "n_cell_types": int(self.n_cell_types),
            "n_scenarios": int(self.n_scenarios),
            "has_energy": True,
            "obs_fields": ["patch", "energy", "scenario_id", "env_id"],
            "env_descriptor": self.get_env_descriptor(),
        }

    # --------- core logic ---------
    def reset(self, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        sid = int(scenario_id) if scenario_id is not None else int(self.rng.randint(0, self.n_scenarios))
        sid = sid % self.n_scenarios
        self.current_scenario_id = sid
        self.current_scenario_name = str(self.scenario_configs[sid].get("name", f"scenario_{sid}"))

        size = int(self.config.size)
        self.grid = np.full((size, size), fill_value=self.EMPTY, dtype=np.int64)
        # walls
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Deterministic goal positions keep tests stable and learning clean.
        if bool(self.config.spawn_both_goals):
            self.grid[1, 1] = self.GOAL_A
            self.grid[size - 2, size - 2] = self.GOAL_B
        else:
            target = str(self.scenario_configs[sid].get("target", "A")).upper()
            if target == "A":
                self.grid[1, 1] = self.GOAL_A
            else:
                self.grid[size - 2, size - 2] = self.GOAL_B

        self.agent_pos = [size // 2, size // 2]
        self.steps = 0
        self._env_descriptor = self._compute_env_descriptor()
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        if self.grid is None:
            raise RuntimeError("InstructionEnv used before reset()")
        vx = self.view_size // 2
        patch = np.zeros((self.view_size, self.view_size), dtype=np.int64)
        ax, ay = self.agent_pos
        size = int(self.grid.shape[0])
        for i in range(-vx, vx + 1):
            for j in range(-vx, vx + 1):
                x = ax + i
                y = ay + j
                if 0 <= x < size and 0 <= y < size:
                    patch[i + vx, j + vx] = int(self.grid[x, y])
                else:
                    patch[i + vx, j + vx] = self.WALL
        # energy channel: remaining steps normalized (helps world-model).
        rem = float(max(0, int(self.config.max_steps) - int(self.steps)))
        energy = rem / float(max(1, int(self.config.max_steps)))
        return {
            "patch": patch,
            "energy": float(energy),
            "scenario_id": int(self.current_scenario_id),
            "env_id": int(self.env_id),
            "env_name": str(self.env_name),
            "env_family": str(self.env_family),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.grid is None:
            raise RuntimeError("InstructionEnv step() before reset()")
        action = int(action)
        if action < 0 or action >= self.n_actions:
            action = 4  # STAY

        self.steps += 1
        done = False
        moved = False

        ax, ay = self.agent_pos
        nx, ny = ax, ay
        if action == 0:
            nx -= 1
        elif action == 1:
            nx += 1
        elif action == 2:
            ny -= 1
        elif action == 3:
            ny += 1

        if action in {0, 1, 2, 3}:
            if int(self.grid[nx, ny]) != self.WALL:
                self.agent_pos = [int(nx), int(ny)]
                moved = True

        reward_env = float(self.config.step_penalty)
        target = str(self.scenario_configs[self.current_scenario_id].get("target", "A")).upper()
        size = int(self.grid.shape[0])
        tx, ty = (1, 1) if target == "A" else (size - 2, size - 2)
        instruction_success: Optional[bool] = None
        if moved and action in {0, 1, 2, 3}:
            prev_dist = abs(int(ax) - int(tx)) + abs(int(ay) - int(ty))
            cur_ax, cur_ay = int(self.agent_pos[0]), int(self.agent_pos[1])
            cur_dist = abs(cur_ax - int(tx)) + abs(cur_ay - int(ty))
            if cur_dist < prev_dist:
                reward_env += float(self.config.progress_reward)
            elif cur_dist > prev_dist:
                reward_env -= float(self.config.progress_reward)

        reason = ""
        if action in {4, 5}:  # STAY/TAKE can finalize on a goal tile
            ax, ay = self.agent_pos
            cell = int(self.grid[ax, ay])
            if cell in {self.GOAL_A, self.GOAL_B}:
                took = "A" if cell == self.GOAL_A else "B"
                if took == target:
                    reward_env = float(self.config.success_reward)
                    reason = "took_correct_goal"
                    instruction_success = True
                else:
                    reward_env = float(self.config.wrong_reward)
                    reason = "took_wrong_goal"
                    instruction_success = False
                done = True

        if not done and self.steps >= int(self.config.max_steps):
            done = True
            reason = "max_steps"
            ax, ay = self.agent_pos
            dist_to_target = abs(int(ax) - int(tx)) + abs(int(ay) - int(ty))
            instruction_success = bool(dist_to_target <= 1)

        ax, ay = self.agent_pos
        dist_to_target = abs(int(ax) - int(tx)) + abs(int(ay) - int(ty))
        at_target = bool(dist_to_target == 0)

        info: Dict[str, Any] = {
            "env_id": int(self.env_id),
            "env_name": str(self.env_name),
            "env_family": str(self.env_family),
            "scenario_id": int(self.current_scenario_id),
            "scenario_name": str(self.current_scenario_name),
            "instruction_target": str(target),
            "at_target": bool(at_target),
            "distance_to_target": int(dist_to_target),
            "instruction_success": instruction_success,
            "reward_env": float(reward_env),
            "got_food": False,
            "took_damage": False,
            "moved": bool(moved),
            "alive": True,
            "death_flag": 0.0,
            "reason": str(reason),
        }
        return self._get_obs(), 0.0, bool(done), info

