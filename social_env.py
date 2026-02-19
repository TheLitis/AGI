"""
Simple multi-agent environment to start tackling the "Social world / ToM" mountain.

This is not a full multi-agent framework; it's a minimal 2-agent grid where
the other agent follows a fixed policy (greedy to the food). The learning agent
must adapt under cooperative vs competitive reward profiles.

Action space is fixed at 6 to remain compatible with EnvPool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env import BaseEnv, build_env_descriptor
from info_contract import normalize_info_contract


@dataclass
class SocialEnvConfig:
    size: int = 9
    view_size: int = 5
    max_steps: int = 60
    step_penalty: float = -0.01
    progress_reward: float = 0.05
    success_reward: float = 1.0
    fail_reward: float = -1.0
    compete_probability: float = 0.5


class SocialEnv(BaseEnv):
    """
    Two agents in a small grid with one food.

    Scenarios:
      0) cooperate: reward if either agent collects the food.
      1) compete: reward only if *you* collect first; negative if the other collects.

    Actions (size=6):
      0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY, 5: TAKE
    """

    EMPTY = 0
    WALL = 1
    FOOD = 2
    OTHER = 3
    HOME = 4

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "TAKE"]

    def __init__(
        self,
        config: Optional[SocialEnvConfig] = None,
        env_id: int = 0,
        env_name: str = "social_env",
        seed: Optional[int] = None,
    ):
        self.config = config or SocialEnvConfig()
        self._env_id = int(env_id)
        self._env_name = str(env_name)
        self.env_family = "social-basic"
        self.rng = np.random.RandomState(seed)

        self.view_size = int(self.config.view_size)
        if self.view_size % 2 != 1:
            raise ValueError("SocialEnv view_size must be odd")

        self.n_cell_types = 5
        self.n_scenarios = 2
        self.scenario_configs: List[Dict[str, Any]] = [
            {
                "name": "cooperate",
                "description": "Two agents. Reward if either collects the food.",
                "mode": "coop",
            },
            {
                "name": "compete",
                "description": "Two agents. Reward only if you collect first; penalty if other collects.",
                "mode": "compete",
            },
        ]
        self.current_scenario_id: int = 0
        self.current_scenario_name: str = str(self.scenario_configs[0]["name"])
        self.description: str = "Two-agent food collection (coop/compete)."

        self.grid: Optional[np.ndarray] = None
        self.agent_pos: List[int] = [0, 0]
        self.other_pos: List[int] = [0, 0]
        self.food_pos: List[int] = [0, 0]
        self.food_present: bool = True
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
        Deterministic oracle for BC/online-BC:
        move greedily to food, then TAKE.
        """
        if self.grid is None:
            return None
        if not self.food_present:
            return 4  # STAY

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        fx, fy = int(self.food_pos[0]), int(self.food_pos[1])
        if ax == fx and ay == fy:
            return 5  # TAKE

        candidates: List[int] = []
        if ax > fx:
            candidates.append(0)  # UP
        elif ax < fx:
            candidates.append(1)  # DOWN
        if ay > fy:
            candidates.append(2)  # LEFT
        elif ay < fy:
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
        goal_density = float(1.0 / max(1.0, size * size))
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
        if scenario_id is not None:
            sid = int(scenario_id)
        else:
            p_compete = float(max(0.0, min(1.0, float(self.config.compete_probability))))
            sid = 1 if float(self.rng.rand()) < p_compete else 0
        sid = sid % self.n_scenarios
        self.current_scenario_id = sid
        self.current_scenario_name = str(self.scenario_configs[sid].get("name", f"scenario_{sid}"))

        size = int(self.config.size)
        self.grid = np.full((size, size), fill_value=self.EMPTY, dtype=np.int64)
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Deterministic positions keep tests stable.
        self.food_pos = [1, size - 2]
        self.food_present = True
        self.grid[self.food_pos[0], self.food_pos[1]] = self.FOOD

        self.agent_pos = [size // 2, size // 2]
        self.other_pos = [size - 2, 1]
        self.steps = 0
        self._env_descriptor = self._compute_env_descriptor()
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        if self.grid is None:
            raise RuntimeError("SocialEnv used before reset()")
        vx = self.view_size // 2
        patch = np.zeros((self.view_size, self.view_size), dtype=np.int64)
        ax, ay = self.agent_pos
        size = int(self.grid.shape[0])
        ox, oy = self.other_pos
        for i in range(-vx, vx + 1):
            for j in range(-vx, vx + 1):
                x = ax + i
                y = ay + j
                if 0 <= x < size and 0 <= y < size:
                    cell = int(self.grid[x, y])
                    if x == ox and y == oy:
                        cell = self.OTHER
                    patch[i + vx, j + vx] = cell
                else:
                    patch[i + vx, j + vx] = self.WALL
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

    def _move_if_free(self, pos: List[int], nx: int, ny: int) -> bool:
        if self.grid is None:
            return False
        if int(self.grid[nx, ny]) == self.WALL:
            return False
        pos[0] = int(nx)
        pos[1] = int(ny)
        return True

    def _other_policy_step(self) -> None:
        """
        Fixed opponent: greedy move toward food (Manhattan), break ties randomly.
        """
        if not self.food_present:
            return
        ox, oy = self.other_pos
        fx, fy = self.food_pos
        candidates: List[Tuple[int, int]] = []
        best = None
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = ox + dx, oy + dy
            if self.grid is None:
                continue
            if int(self.grid[nx, ny]) == self.WALL:
                continue
            d = abs(nx - fx) + abs(ny - fy)
            if best is None or d < best:
                best = d
                candidates = [(nx, ny)]
            elif d == best:
                candidates.append((nx, ny))
        if not candidates:
            return
        nx, ny = candidates[int(self.rng.randint(0, len(candidates)))]
        self.other_pos = [int(nx), int(ny)]

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.grid is None:
            raise RuntimeError("SocialEnv step() before reset()")
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

        prev_food_dist = abs(int(ax) - int(self.food_pos[0])) + abs(int(ay) - int(self.food_pos[1]))
        if action in {0, 1, 2, 3}:
            moved = self._move_if_free(self.agent_pos, nx, ny)

        # Opponent moves after you (gives the learner a slight advantage).
        self._other_policy_step()

        reward_env = float(self.config.step_penalty)
        if moved and self.food_present and action in {0, 1, 2, 3}:
            cur_ax, cur_ay = int(self.agent_pos[0]), int(self.agent_pos[1])
            cur_food_dist = abs(cur_ax - int(self.food_pos[0])) + abs(cur_ay - int(self.food_pos[1]))
            if cur_food_dist < prev_food_dist:
                reward_env += float(self.config.progress_reward)
            elif cur_food_dist > prev_food_dist:
                reward_env -= float(self.config.progress_reward)

        reason = ""
        got_food = False
        other_got_food = False

        # Food pickup rules:
        #   - you must TAKE on the food tile,
        #   - the other agent auto-collects by stepping onto the food.
        if self.food_present:
            fx, fy = self.food_pos
            ox, oy = self.other_pos
            if ox == fx and oy == fy:
                other_got_food = True
                self.food_present = False
                self.grid[fx, fy] = self.EMPTY
            if action in {4, 5}:
                ax, ay = self.agent_pos
                if ax == fx and ay == fy:
                    got_food = True
                    self.food_present = False
                    self.grid[fx, fy] = self.EMPTY

        mode = str(self.scenario_configs[self.current_scenario_id].get("mode", "coop"))
        if got_food or other_got_food:
            done = True
            if mode == "compete":
                if got_food and not other_got_food:
                    reward_env = float(self.config.success_reward)
                    reason = "you_got_food"
                else:
                    reward_env = float(self.config.fail_reward)
                    reason = "other_got_food"
            else:
                reward_env = float(self.config.success_reward)
                reason = "food_collected"

        if not done and self.steps >= int(self.config.max_steps):
            done = True
            reason = "max_steps"

        info: Dict[str, Any] = {
            "env_id": int(self.env_id),
            "env_name": str(self.env_name),
            "env_family": str(self.env_family),
            "scenario_id": int(self.current_scenario_id),
            "scenario_name": str(self.current_scenario_name),
            "reward_env": float(reward_env),
            "got_food": bool(got_food),
            "other_got_food": bool(other_got_food),
            "took_damage": False,
            "moved": bool(moved),
            "alive": True,
            "death_flag": 0.0,
            "reason": str(reason),
        }
        timeout = bool(done and str(reason) == "max_steps")
        success: Optional[bool] = None
        reason_norm = str(reason).strip().lower()
        if done:
            if reason_norm in {"you_got_food", "food_collected"}:
                success = True
            elif reason_norm == "other_got_food":
                success = False
        info["social_success"] = success
        info = normalize_info_contract(
            info,
            done=bool(done),
            reward_env=float(reward_env),
            terminated_reason=str(reason),
            success=success,
            constraint_violation=False,
            catastrophic=False,
            timeout=timeout,
            events={
                "got_food": float(bool(got_food)),
                "other_got_food": float(bool(other_got_food)),
                "moved": float(bool(moved)),
                "social_success": float(success) if success is not None else 0.0,
            },
        )
        return self._get_obs(), 0.0, bool(done), info

