"""
MiniGrid environment pool that mirrors the toy GridWorld EnvPool API.

This module keeps the same conceptual contract as env.EnvPool: reset/step/sample
work over a small set of envs, observations expose a discrete patch plus a scalar
energy proxy, and info dicts carry food/danger/movement flags used by Trainer.

Observation encoding:
- Uses MiniGrid partial observations (agent-centric) with agent_view_size=5.
- Each cell is mapped to one of a small set of cell types:
    0 empty/floor, 1 wall/unseen, 2 goal/ball, 3 danger (lava),
    4 key, 5 door (n_cell_types = 6).
- The resulting patch is int64 with shape (5, 5) to stay compatible with the
  existing Perception encoder (which embeds patch ids).
- Energy is a simple survival budget proxy: 1 - steps / max_steps.

Task selection and lifelong regime mapping (documented for analyze/README):
- empty_easy / empty_random  -> "food-seeking / survival" regimes (R1/R1_return)
  reward comes from reaching the goal (MiniGrid reward > 0 -> got_food=True).
- doorkey                     -> still food-driven but requires key/door unlock
  (counts as food on success).
- lava_gap                    -> "danger inspection" regime (R3): stepping into
  lava sets took_damage=True and death_flag=1.0.

Reward returned by step is the underlying MiniGrid reward (small positive on
success); the Trainer still computes preference-based rewards from info flags,
so sign consistency with R1/R2/R3 is preserved.
"""

from typing import Any, Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT

from env import BaseEnv, EnvPool, build_env_descriptor, canonical_env_family


# Cell type ids (kept small to stay compatible with the existing embedding sizes).
CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2
CELL_DANGER = 3
CELL_KEY = 4
CELL_DOOR = 5
N_CELL_TYPES = 6


# Canonical MiniGrid scenarios (string aliases -> task specs + feature hints).
SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "minigrid-empty": {
        "task_id": "MiniGrid-Empty-8x8-v0",
        "name": "minigrid-empty",
        "category": "food",
        "split": "train",
        "features": {"has_lava": False, "has_door": False, "has_key": False, "goal_density": 0.05, "danger_density": 0.0, "wall_density": 0.1},
    },
    "minigrid-doorkey": {
        "task_id": "MiniGrid-DoorKey-6x6-v0",
        "name": "minigrid-doorkey",
        "category": "food",
        "split": "train",
        "features": {"has_lava": False, "has_door": True, "has_key": True, "goal_density": 0.05, "danger_density": 0.0, "wall_density": 0.15},
    },
    "minigrid-lavacrossing": {
        "task_id": "MiniGrid-LavaCrossingS9N1-v0",
        "name": "minigrid-lavacrossing",
        "category": "danger",
        "split": "test",
        "features": {"has_lava": True, "has_door": False, "has_key": False, "goal_density": 0.02, "danger_density": 0.25, "wall_density": 0.05},
    },
    "minigrid-multiroom": {
        "task_id": "MiniGrid-MultiRoom-N6-v0",
        "name": "minigrid-multiroom",
        "category": "survival",
        "split": "test",
        "features": {"has_lava": False, "has_door": True, "has_key": False, "goal_density": 0.03, "danger_density": 0.0, "wall_density": 0.2},
    },
    # Backward-compatible aliases
    "lava_gap": {
        "task_id": "MiniGrid-LavaGapS5-v0",
        "name": "lava_gap",
        "category": "danger",
        "split": "train",
        "features": {"has_lava": True, "has_door": False, "has_key": False, "goal_density": 0.02, "danger_density": 0.25, "wall_density": 0.05},
    },
    "empty_easy": {
        "task_id": "MiniGrid-Empty-6x6-v0",
        "name": "empty_easy",
        "category": "food",
        "split": "train",
        "features": {"has_lava": False, "has_door": False, "has_key": False, "goal_density": 0.05, "danger_density": 0.0, "wall_density": 0.05},
    },
}


def _default_task_specs() -> List[Dict[str, Any]]:
    """
    Small menu of MiniGrid tasks. Names are used as scenario labels for lifelong
    regimes (R1/R2/R3/R1_return).

    Each spec:
      - task_id: MiniGrid env ID
      - name: short label used as scenario/env_name
      - category: "food", "survival", or "danger"
      - split: "train" or "test" (for explicit train/test task splits)
    """
    return [
        dict(SCENARIO_REGISTRY["minigrid-empty"]),
        dict(SCENARIO_REGISTRY["minigrid-doorkey"]),
        dict(SCENARIO_REGISTRY["minigrid-lavacrossing"]),
        dict(SCENARIO_REGISTRY["minigrid-multiroom"]),
    ]


def _scenario_specs_from_names(names: List[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for raw in names:
        split_hint = None
        name = raw
        if raw.startswith("test:"):
            split_hint = "test"
            name = raw[len("test:") :]
        elif raw.startswith("train:"):
            split_hint = "train"
            name = raw[len("train:") :]
        name_lc = name.lower()
        base = SCENARIO_REGISTRY.get(name_lc)
        if base is None:
            # fall back to existing ids if user passed MiniGrid-* id directly
            base = {
                "task_id": name,
                "name": name_lc,
                "category": "food",
                "split": "train",
                "features": {},
            }
        spec = dict(base)
        if split_hint:
            spec["split"] = split_hint
        specs.append(spec)
    return specs


class MiniGridTaskEnv(BaseEnv):
    """
    Wraps a single MiniGrid environment to look like GridWorldEnv.
    - action space: 6 actions (L/R/Forward/Pickup/Toggle/Stay(no-op))
    - observation: (5x5) patch of discrete ids + scalar energy
    """

    ACTIONS = ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "PICKUP", "TOGGLE", "STAY"]

    def __init__(
        self,
        task_id: str,
        scenario_names: List[str],
        scenario_index: int,
        view_size: int = 5,
        seed: int = 0,
        env_id: int = 0,
        env_name: Optional[str] = None,
        category: str = "food",
        env_family: Optional[str] = None,
        env_features: Optional[Dict[str, Any]] = None,
    ):
        self.view_size = view_size
        self.env = gym.make(task_id, agent_view_size=view_size)
        self.unwrapped = self.env.unwrapped

        self.max_steps = int(getattr(self.unwrapped, "max_steps", 100))
        self.rng = np.random.RandomState(seed)
        self.n_cell_types = N_CELL_TYPES
        self.n_scenarios = len(scenario_names)
        self.scenario_configs = [{"name": n} for n in scenario_names]
        self.current_scenario_id = int(scenario_index) % max(1, self.n_scenarios)
        self.current_scenario_name = scenario_names[self.current_scenario_id]

        self._env_id = int(env_id)
        self._env_name = env_name or task_id
        self.env_family = env_family or canonical_env_family(self._env_name)
        self.env_features = env_features or {}

        # Keep a per-task hint for danger vs food/survival
        cat = category or "food"
        if cat not in ("food", "survival", "danger"):
            base = (env_name or task_id or "").lower()
            if "lava" in base:
                cat = "danger"
            elif "empty" in base or "doorkey" in base:
                cat = "food"
            else:
                cat = "survival"
        self.category = cat
        self._env_descriptor = self._compute_env_descriptor()

        # map actions -> MiniGrid discrete actions (done is a benign no-op)
        self._action_map = {
            0: 0,  # left
            1: 1,  # right
            2: 2,  # forward
            3: 3,  # pickup
            4: 5,  # toggle/open
            5: 6,  # done (acts like stay/no-op)
        }

    # --------- metadata helpers (BaseEnv) ---------
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
    def action_meanings(self) -> Iterable[str]:
        return tuple(self.ACTIONS)

    def sample_random_action(self) -> int:
        return int(self.rng.randint(0, self.n_actions))

    # --------- descriptors ---------
    def get_env_descriptor(self) -> np.ndarray:
        """
        Approximate descriptor aligned with GridWorldEnv:
          width, height, goal density, danger density, wall fraction, max_steps.
        Normalization mirrors env.GridWorldEnv.get_env_descriptor.
        """
        return self._env_descriptor.copy()

    def _compute_env_descriptor(self) -> np.ndarray:
        width = float(getattr(self.unwrapped, "width", 6))
        height = float(getattr(self.unwrapped, "height", 6))
        goal_density = float(self.env_features.get("goal_density", 0.05))
        danger_density = float(self.env_features.get("danger_density", 0.25 if "lava" in self._env_name else 0.0))
        wall_density = float(self.env_features.get("wall_density", 0.15))
        has_door = bool(self.env_features.get("has_door", False))
        has_key = bool(self.env_features.get("has_key", False))
        has_lava = bool(self.env_features.get("has_lava", "lava" in self._env_name))
        return build_env_descriptor(
            env_family=self.env_family,
            width=width,
            height=height,
            goal_density=goal_density,
            danger_density=danger_density,
            wall_density=wall_density,
            has_door=has_door,
            has_key=has_key,
            has_lava=has_lava,
            max_steps=float(self.max_steps),
        )

    def get_obs_spec(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
            "patch_size": self.view_size,
            "n_cell_types": self.n_cell_types,
            "n_scenarios": self.n_scenarios,
            "has_energy": True,
            "obs_fields": ["patch", "energy", "scenario_id", "env_id"],
            "env_descriptor": self.get_env_descriptor(),
        }

    # --------- core helpers ---------
    def reset(self, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        if scenario_id is not None:
            self.current_scenario_id = int(scenario_id) % max(1, self.n_scenarios)
            self.current_scenario_name = self.scenario_configs[self.current_scenario_id]["name"]

        self.steps = 0
        obs, _ = self.env.reset(seed=int(self.rng.randint(0, 1_000_000)))
        return self._format_obs(obs)

    def _encode_patch(self, obs: Dict[str, Any]) -> np.ndarray:
        image = obs.get("image")
        if image is None:
            raise ValueError("MiniGrid observation missing 'image' key")
        patch = np.zeros((self.view_size, self.view_size), dtype=np.int64)
        for i in range(self.view_size):
            for j in range(self.view_size):
                obj_idx = int(image[i, j, 0])
                obj_name = IDX_TO_OBJECT.get(obj_idx, "empty")
                if obj_name == "lava":
                    cell = CELL_DANGER
                elif obj_name in {"wall", "unseen"}:
                    cell = CELL_WALL
                elif obj_name in {"goal", "ball"}:
                    cell = CELL_GOAL
                elif obj_name == "key":
                    cell = CELL_KEY
                elif obj_name == "door":
                    cell = CELL_DOOR
                else:
                    cell = CELL_EMPTY
                patch[i, j] = cell
        return patch

    def _format_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        energy = max(0.0, 1.0 - float(self.steps) / float(max(1, self.max_steps)))
        patch = self._encode_patch(obs)
        return {
            "patch": patch,
            "energy": energy,
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
        }

    def step(self, action: int):
        if not (0 <= action < self.n_actions):
            raise AssertionError(f"Action {action} out of bounds for MiniGridTaskEnv")

        prev_pos = tuple(getattr(self.unwrapped, "agent_pos", (-1, -1)))
        mg_action = self._action_map.get(int(action), 6)
        obs, reward, terminated, truncated, info_raw = self.env.step(mg_action)
        self.steps += 1

        done = bool(terminated or truncated)
        patch = self._encode_patch(obs)
        moved = False
        try:
            moved = tuple(getattr(self.unwrapped, "agent_pos", prev_pos)) != prev_pos
        except Exception:
            moved = action != 5  # best-effort fallback

        got_food = bool(reward > 0.0)
        took_damage = bool(patch[self.view_size // 2, self.view_size // 2] == CELL_DANGER)
        if not took_damage and terminated and reward <= 0.0 and "lava" in self.env_name:
            took_damage = True
        alive = not took_damage
        death_flag = 1.0 if took_damage else 0.0

        info = {
            "got_food": got_food,
            "took_damage": took_damage,
            "moved": moved or action != 5,
            "alive": alive,
            "death_flag": death_flag,
            "reason": "",
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
            "category": self.category,
        }
        if done and took_damage:
            info["reason"] = "terminated_danger"
        elif done and got_food:
            info["reason"] = "goal_reached"
        elif done and self.steps >= self.max_steps:
            info["reason"] = "max_steps"

        obs_fmt = self._format_obs(obs)
        return obs_fmt, float(reward), done, info


class MiniGridEnvPool(EnvPool):
    """
    EnvPool-backed MiniGrid family with a handful of canonical tasks.
    Public interface mirrors env.EnvPool for Trainer/experiment wiring.
    """

    def __init__(
        self,
        seed: int = 0,
        schedule_mode: str = "iid",
        task_specs: Optional[List[Dict[str, Any]]] = None,
        scenario_names: Optional[List[str]] = None,
    ):
        if task_specs is not None:
            specs = task_specs
        elif scenario_names is not None:
            specs = _scenario_specs_from_names(scenario_names)
        else:
            specs = _default_task_specs()
        valid_specs: List[Dict[str, Any]] = []
        for spec in specs:
            task_id = spec.get("task_id")
            try:
                gym.spec(task_id)
            except Exception as exc:
                print(f"[WARN] Skipping MiniGrid task '{task_id}': {exc}")
                continue
            valid_specs.append(spec)
        if not valid_specs:
            raise RuntimeError("MiniGridEnvPool could not initialize any environments (no valid task specs).")

        scenario_names = [s.get("name", f"task_{i}") for i, s in enumerate(valid_specs)]
        envs: List[MiniGridTaskEnv] = []
        final_specs: List[Dict[str, Any]] = []
        for idx, spec in enumerate(valid_specs):
            try:
                envs.append(
                    MiniGridTaskEnv(
                        task_id=spec["task_id"],
                        scenario_names=scenario_names,
                        scenario_index=idx,
                        view_size=5,
                        seed=seed + idx,
                        env_id=idx,
                        env_name=spec.get("name", f"task_{idx}"),
                        category=spec.get("category", "food"),
                        env_family=spec.get("name", spec.get("env_family", "")),
                        env_features=spec.get("features", {}),
                    )
                )
                final_specs.append(spec)
            except Exception as exc:
                raise RuntimeError(f"Failed to init MiniGrid env '{spec.get('task_id')}': {exc}") from exc
        if not envs:
            raise RuntimeError("MiniGridEnvPool could not initialize any environments.")

        # Explicit train/test split using spec["split"], with fallbacks.
        train_env_ids = [i for i, spec in enumerate(final_specs) if spec.get("split", "train") == "train"]
        test_env_ids = [i for i, spec in enumerate(final_specs) if spec.get("split", "train") == "test"]

        if not train_env_ids:
            train_env_ids = list(range(max(1, len(envs) - 1)))

        if not test_env_ids:
            test_env_ids = [len(envs) - 1]

        super().__init__(
            envs=envs,
            schedule_mode=schedule_mode,
            seed=seed,
            train_env_ids=train_env_ids,
            test_env_ids=test_env_ids,
            scenario_ids_by_phase={
                "A": train_env_ids,
                "B": train_env_ids + test_env_ids,
                "C": test_env_ids,
            },
        )

        self.task_specs = final_specs
        self.env_categories = [spec.get("category", "unknown") for spec in final_specs]
        self.env_names = [spec.get("name", f"task_{i}") for i, spec in enumerate(final_specs)]

        # Override scenario count so embeddings can cover all task labels.
        self.n_scenarios = len(self.env_names)

    @property
    def task_metadata(self) -> List[Dict[str, Any]]:
        """Return a small, JSON-serializable summary of the MiniGrid tasks."""
        data: List[Dict[str, Any]] = []
        for env_id, spec in enumerate(self.task_specs):
            data.append(
                {
                    "env_id": int(env_id),
                    "name": str(spec.get("name", f"task_{env_id}")),
                    "task_id": str(spec.get("task_id", "")),
                    "category": str(spec.get("category", "unknown")),
                    "split": str(spec.get("split", "train")),
                }
            )
        return data

    def reset(self, scenario_id: Optional[int] = None, split: Optional[str] = None):
        return super().reset(scenario_id=scenario_id, split=split)

    def step(self, action: int):
        return super().step(action)
