"""
Environments for the proto-creature project.

- `GridWorldEnv`: small 2D gridworld with food/danger/home cells, optional
  multi-task curriculum scheduling.
- `EnvPool`: wrapper to sample across multiple environments with lifecycle
  phases A/B/C (train/test splits).

Numerical behavior is unchanged; this module adds documentation and light
PEP8 cleanup only.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Iterable
from info_contract import normalize_info_contract

# Lightweight registry to keep env-family ids stable across backends.
ENV_FAMILY_IDS = {
    "gridworld-basic": 0,
    "minigrid-empty": 1,
    "minigrid-doorkey": 2,
    "minigrid-lavacrossing": 3,
    "minigrid-multiroom": 4,
    "tools-basic": 5,
    "computer-basic": 6,
    "repo-basic": 7,
    "instruction-basic": 8,
    "social-basic": 9,
}


def canonical_env_family(name: str) -> str:
    base = (name or "").lower()
    for key in ENV_FAMILY_IDS.keys():
        if key in base:
            return key
    if "repo" in base:
        return "repo-basic"
    if "tool" in base:
        return "tools-basic"
    if "instruction" in base or "language" in base:
        return "instruction-basic"
    if "social" in base or "multiagent" in base:
        return "social-basic"
    if "computer" in base or "code" in base or "project" in base:
        return "computer-basic"
    if "lava" in base:
        return "minigrid-lavacrossing"
    if "door" in base or "key" in base:
        return "minigrid-doorkey"
    if "empty" in base:
        return "minigrid-empty"
    if "multiroom" in base or "room" in base:
        return "minigrid-multiroom"
    if "grid" in base:
        return "gridworld-basic"
    return "gridworld-basic"


def env_family_id(name: str) -> int:
    fam = canonical_env_family(name)
    return int(ENV_FAMILY_IDS.get(fam, max(ENV_FAMILY_IDS.values()) + 1))


def build_env_descriptor(
    env_family: str,
    width: float,
    height: float,
    goal_density: float,
    danger_density: float,
    wall_density: float,
    has_door: bool = False,
    has_key: bool = False,
    has_lava: bool = False,
    max_steps: float = 100.0,
) -> np.ndarray:
    """
    Unified env descriptor used by Perception/SelfModel across backends.
    All values are normalized to [0,1] where possible.
    Layout:
      [0] env_family_id / 10.0
      [1] width / 20
      [2] height / 20
      [3] goal density proxy
      [4] danger density proxy
      [5] wall density proxy
      [6] has_door (0/1)
      [7] has_key (0/1)
      [8] has_lava (0/1)
      [9] max_steps / 200
    """
    desc = np.array(
        [
            env_family_id(env_family) / 10.0,
            float(width) / 20.0,
            float(height) / 20.0,
            float(goal_density),
            float(danger_density),
            float(wall_density),
            1.0 if has_door else 0.0,
            1.0 if has_key else 0.0,
            1.0 if has_lava else 0.0,
            float(max_steps) / 200.0,
        ],
        dtype=np.float32,
    )
    return desc

# =========================
#  Environment interfaces
# =========================


class BaseEnv:
    """
    Minimal interface for environments that the agent/trainers can swap between.
    Keeping it lightweight: reset/step plus a few metadata helpers.
    """

    def reset(self, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @property
    def n_actions(self) -> int:
        raise NotImplementedError

    @property
    def action_meanings(self) -> Iterable[str]:
        raise NotImplementedError

    @property
    def env_id(self) -> int:
        raise NotImplementedError

    @property
    def env_name(self) -> str:
        raise NotImplementedError

    def sample_random_action(self) -> int:
        """Uniform random action; override if env has specific sampling needs."""
        raise NotImplementedError

    def get_obs_spec(self) -> Dict[str, Any]:
        """
        Lightweight description of observation fields used for wiring encoders.
        Should at least describe patch_size/energy presence/ids counts where relevant.
        """
        raise NotImplementedError

# =========================
#  Environment: GridWorld
# =========================


class GridWorldEnv(BaseEnv):
    """
    Lightweight 2D gridworld used across all four training stages.
    
    Cells:
      0 = EMPTY
      1 = WALL
      2 = FOOD
      3 = DANGER
      4 = HOME (currently unused)
    
    The environment returns zero reward; trait-based rewards are computed in
    the Trainer. Info dict includes events (food, damage, moved, alive),
    scenario metadata, and death flags.
    """
    EMPTY = 0
    WALL = 1
    FOOD = 2
    DANGER = 3
    HOME = 4

    def __init__(
        self,
        size: int = 10,
        view_size: int = 5,
        max_energy: int = 20,
        food_energy: int = 10,
        move_cost: int = 1,
        danger_damage: int = 8,
        food_prob: float = 0.05,
        danger_prob: float = 0.05,
        max_steps: int = 200,
        seed: Optional[int] = None,
        # v0.9: мультизадачный режим
        multi_task: bool = False,
        scenario_configs: Optional[List[Dict[str, Any]]] = None,
        # v0.10: режим выбора сценариев (non-stationarity / curriculum)  # NEW
        #   "iid"         — как раньше: сценарий выбирается случайно каждый эпизод
        #   "round_robin" — проходим сценарии по кругу: 0,1,2,3,0,1,2,3,...
        #   "curriculum"  — эпизоды идут фазами: сперва только 0, потом 1, потом 2 и т.д.
        schedule_mode: str = "iid",             # NEW
        episodes_per_phase: int = 50,           # NEW: длина фазы для curriculum
        env_id: int = 0,
        env_name: str = "gridworld_v0",
    ):
        assert view_size % 2 == 1, "view_size must be odd"
        self.size = size
        self.view_size = view_size
        self.max_energy = max_energy
        self.food_energy = food_energy
        self.move_cost = move_cost
        self.danger_damage = danger_damage
        self.food_prob = food_prob
        self.danger_prob = danger_prob
        self.max_steps = max_steps

        # Identity (for multi-env pooling/embeddings)
        self._env_id = int(env_id)
        self._env_name = env_name
        self.env_family = "gridworld-basic"

        self.rng = np.random.RandomState(seed)
        self.grid = None
        self.agent_pos = None
        self.energy = None
        self.steps = None

        # Actions
        self.ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "TAKE"]
        self.n_cell_types = 5  # EMPTY/WALL/FOOD/DANGER/HOME

        # env descriptor (raw, normalized via get_env_descriptor)
        self._env_descriptor = self._compute_env_descriptor()

        # v0.9: мультизадачность
        self.multi_task = multi_task
        self.scenario_configs = scenario_configs
        self.current_scenario_id = 0
        self.current_scenario_name = "single"

        if self.multi_task and self.scenario_configs is None:
            self._init_default_scenarios()

        if self.multi_task and self.scenario_configs is not None:
            self.n_scenarios = len(self.scenario_configs)
        else:
            self.n_scenarios = 1

        # v0.10: non-stationarity / curriculum                         # NEW
        self.schedule_mode = schedule_mode
        self.episodes_per_phase = episodes_per_phase
        self.episode_idx = 0  # счётчик эпизодов (reset'ов)            # NEW
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

    def get_obs_spec(self) -> Dict[str, Any]:
        """
        Small contract for wiring encoders (Perception/Text later):
          - patch_size / n_cell_types tell embedding sizes,
          - n_scenarios for scenario embedding,
          - has_energy flag for scalar channel,
          - env_id/env descriptor for env conditioning at agent side.
        """
        return {
            "env_id": self.env_id,
            "env_name": self.env_name,
            "patch_size": self.view_size,
            "n_cell_types": self.n_cell_types,
            "n_scenarios": self.n_scenarios,
            "has_energy": True,
            "obs_fields": ["patch", "energy", "scenario_id", "env_id"],
            "env_descriptor": self.get_env_descriptor(),
        }

    def _compute_env_descriptor(self) -> np.ndarray:
        wall_prob = 0.0  # walls only on borders in current GridWorld
        return build_env_descriptor(
            env_family=self.env_family,
            width=float(self.size),
            height=float(self.size),
            goal_density=float(self.food_prob),
            danger_density=float(self.danger_prob),
            wall_density=wall_prob,
            has_door=False,
            has_key=False,
            has_lava=False,
            max_steps=float(self.max_steps),
        )

    def get_env_descriptor(self) -> np.ndarray:
        """
        Return normalized environment descriptor vector.
        """
        return self._env_descriptor.copy()




    def _init_default_scenarios(self):
        """
        v0.9: базовый набор сценариев. Агент не видит scenario_id в obs,
        только в info (для логирования). Разные сценарии = разные
        динамики/распределения мира, но общая биология и traits.
        """
        base_size = self.size
        base_food_prob = self.food_prob
        base_danger_prob = self.danger_prob
        base_move_cost = self.move_cost
        base_danger_damage = self.danger_damage

        self.scenario_configs = [
            {
                "name": "balanced",
                "size": base_size,
                "food_prob": base_food_prob,
                "danger_prob": base_danger_prob,
                "move_cost": base_move_cost,
                "danger_damage": base_danger_damage,
            },
            {
                "name": "empty",
                "size": base_size,
                "food_prob": base_food_prob * 0.2,
                "danger_prob": 0.0,
                "move_cost": base_move_cost,
                "danger_damage": base_danger_damage,
            },
            {
                # "ферма": много еды, мало опасности
                "name": "food_rich",
                "size": base_size,
                "food_prob": min(base_food_prob * 3.0, 0.3),
                "danger_prob": base_danger_prob * 0.5,
                "move_cost": base_move_cost,
                "danger_damage": base_danger_damage,
            },
            {
                # "опасный район": много опасности, мало еды, удар сильнее
                "name": "dangerous",
                "size": base_size,
                "food_prob": base_food_prob * 0.5,
                "danger_prob": min(base_danger_prob * 3.0, 0.3),
                "move_cost": base_move_cost,
                "danger_damage": int(base_danger_damage * 1.5),
            },
            {
                # "пустыня": большой мир, мало всего
                "name": "sparse_large",
                "size": base_size + 4,
                "food_prob": base_food_prob * 0.5,
                "danger_prob": base_danger_prob * 0.5,
                "move_cost": base_move_cost,
                "danger_damage": base_danger_damage,
            },
            {
                # псевдо doorkey: умеренная опасность и еда, длиннее путь
                "name": "door_key",
                "size": base_size + 2,
                "food_prob": base_food_prob * 0.8,
                "danger_prob": base_danger_prob * 0.8,
                "move_cost": base_move_cost,
                "danger_damage": base_danger_damage,
            },
            {
                # псевдо lavacrossing: много опасности, мало еды, высокий штраф
                "name": "lavacrossing",
                "size": base_size + 2,
                "food_prob": base_food_prob * 0.2,
                "danger_prob": min(base_danger_prob * 4.0, 0.4),
                "move_cost": base_move_cost,
                "danger_damage": int(base_danger_damage * 2.0),
            },
        ]

    # --------- scenario scheduling (NEW) ---------

    def _apply_scenario(self, sid: int):
        """
        Применяет параметры сценария к миру (size, food_prob, danger_prob, ...).
        """
        if not self.multi_task or self.scenario_configs is None:
            self.current_scenario_id = 0
            self.current_scenario_name = "single"
            return

        sid = int(sid) % len(self.scenario_configs)
        self.current_scenario_id = sid
        conf = self.scenario_configs[sid]
        self.current_scenario_name = conf.get("name", f"scenario_{sid}")

        # обновляем динамические параметры мира под сценарий и дескриптор
        self.size = int(conf.get("size", self.size))
        self.food_prob = float(conf.get("food_prob", self.food_prob))
        self.danger_prob = float(conf.get("danger_prob", self.danger_prob))
        self.move_cost = int(conf.get("move_cost", self.move_cost))
        self.danger_damage = int(conf.get("danger_damage", self.danger_damage))
        self._env_descriptor = self._compute_env_descriptor()

    def _choose_scenario_id(self, scenario_id: Optional[int] = None):
        """
        Выбор сценария с учётом режима schedule_mode и номера эпизода.
        Если scenario_id явно передан в reset(...), он имеет приоритет.
        """
        # single-task режим
        if not self.multi_task or self.scenario_configs is None:
            self.current_scenario_id = 0
            self.current_scenario_name = "single"
            return

        if scenario_id is not None:
            sid = int(scenario_id) % len(self.scenario_configs)
        else:
            mode = (self.schedule_mode or "iid").lower()
            n = len(self.scenario_configs)

            if mode == "iid":
                # как раньше: независимый случайный сценарий каждый эпизод
                sid = self.rng.randint(0, n)
            elif mode == "round_robin":
                # эпизоды идут по кругу: 0,1,2,3,0,1,2,3,...
                sid = self.episode_idx % n
            elif mode == "curriculum":
                # фазы по episodes_per_phase:
                #   0 .. phase_len-1       -> сценарий 0
                #   phase_len .. 2*phase_len-1 -> сценарий 1
                #   ...
                phase_len = max(1, int(self.episodes_per_phase))
                phase = self.episode_idx // phase_len
                sid = min(phase, n - 1)
            else:
                # дефолт на всякий случай — iid
                sid = self.rng.randint(0, n)

        self._apply_scenario(sid)

    # --------- core helpers ---------

    def reset(self, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        # v0.10: считаем эпизоды (каждый reset = новый эпизод)         # NEW
        self.episode_idx += 1

        # v0.9+v0.10: выбираем сценарий (с учётом schedule_mode)        # NEW
        self._choose_scenario_id(scenario_id)

        # генерим мир под выбранный сценарий
        self.grid = np.zeros((self.size, self.size), dtype=np.int64)
        for i in range(self.size):
            for j in range(self.size):
                if self.rng.rand() < self.food_prob:
                    self.grid[i, j] = self.FOOD
                elif self.rng.rand() < self.danger_prob:
                    self.grid[i, j] = self.DANGER
                else:
                    self.grid[i, j] = self.EMPTY

        # границы — стены
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # старт в центре (в новом размере)
        self.agent_pos = [self.size // 2, self.size // 2]
        self.energy = self.max_energy
        self.steps = 0

        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        vx = self.view_size // 2
        patch = np.zeros((self.view_size, self.view_size), dtype=np.int64)
        ax, ay = self.agent_pos
        for i in range(-vx, vx + 1):
            for j in range(-vx, vx + 1):
                x = ax + i
                y = ay + j
                if 0 <= x < self.size and 0 <= y < self.size:
                    patch[i + vx, j + vx] = self.grid[x, y]
                else:
                    patch[i + vx, j + vx] = self.WALL
        energy_norm = float(self.energy) / float(self.max_energy)
        return {
            "patch": patch,
            "energy": energy_norm,
            "scenario_id": self.current_scenario_id,
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Возвращает: obs, reward (всегда 0.0), done, info.
        info содержит физические события; Trainer уже превратит их в scalar reward.
        """
        assert 0 <= action < self.n_actions
        self.steps += 1
        done = False

        ax, ay = self.agent_pos

        info = {
            "got_food": False,
            "took_damage": False,
            "moved": False,
            "alive": True,
            "death_flag": 0.0,
            "reason": "",
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
            # v0.9: логируем, в каком сценарии были
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
        }

        # стоимость шага по энергии
        self.energy -= self.move_cost

        # движение
        if self.ACTIONS[action] == "UP":
            nx, ny = ax - 1, ay
        elif self.ACTIONS[action] == "DOWN":
            nx, ny = ax + 1, ay
        elif self.ACTIONS[action] == "LEFT":
            nx, ny = ax, ay - 1
        elif self.ACTIONS[action] == "RIGHT":
            nx, ny = ax, ay + 1
        else:
            nx, ny = ax, ay  # STAY/TAKE

        if (
            0 <= nx < self.size
            and 0 <= ny < self.size
            and self.grid[nx, ny] != self.WALL
        ):
            if nx != ax or ny != ay:
                info["moved"] = True
            self.agent_pos = [nx, ny]
            ax, ay = nx, ny  # обновляем

        # клетка под ногами
        cell = self.grid[ax, ay]

        # опасность
        if cell == self.DANGER:
            self.energy -= self.danger_damage
            info["took_damage"] = True

        # действие TAKE — попытка съесть еду
        if self.ACTIONS[action] == "TAKE":
            if cell == self.FOOD:
                self.energy = min(self.max_energy, self.energy + self.food_energy)
                self.grid[ax, ay] = self.EMPTY
                info["got_food"] = True

        # смерть по энергии
        if self.energy <= 0:
            done = True
            info["alive"] = False
            info["death_flag"] = 1.0
            info["reason"] = "energy_depleted"

        # ограничение по шагам
        if self.steps >= self.max_steps:
            done = True
            if info["reason"] == "":
                info["reason"] = "max_steps"

        obs = self._get_obs()
        reward = 0.0  # вся настоящая награда считается в Trainer по traits
        reason = str(info.get("reason", "") or "")
        # Reaching the configured episode horizon is a normal termination for
        # gridworld curricula, not an infrastructure timeout.
        timeout = False
        catastrophic = bool(float(info.get("death_flag", 0.0) or 0.0) > 0.0)
        constraint_violation = bool(info.get("took_damage", False) or catastrophic)
        success: Optional[bool] = None
        if done:
            if reason == "energy_depleted":
                success = False
            elif reason == "max_steps":
                success = True
        info = normalize_info_contract(
            info,
            done=bool(done),
            reward_env=float(reward),
            terminated_reason=reason,
            success=success,
            constraint_violation=constraint_violation,
            catastrophic=catastrophic,
            timeout=timeout,
            events={
                "got_food": float(bool(info.get("got_food", False))),
                "took_damage": float(bool(info.get("took_damage", False))),
                "moved": float(bool(info.get("moved", False))),
                "alive": float(bool(info.get("alive", True))),
                "death_flag": float(info.get("death_flag", 0.0) or 0.0),
            },
        )
        return obs, reward, done, info


# =========================
#  Multi-env wrapper
# =========================


class EnvPool(BaseEnv):
    """
    Wraps multiple BaseEnv-compatible environments and selects one per episode.
    Supports iid sampling or round-robin over envs.
    """

    def __init__(
        self,
        envs: List[BaseEnv],
        schedule_mode: str = "iid",
        seed: Optional[int] = None,
        train_env_ids: Optional[List[int]] = None,
        test_env_ids: Optional[List[int]] = None,
        scenario_ids_by_phase: Optional[Dict[str, List[int]]] = None,
        scenario_schedule_mode: Optional[str] = None,
    ):
        assert len(envs) > 0, "EnvPool requires at least one env"
        self.envs = envs
        self.n_envs = len(envs)
        self.train_env_ids = list(train_env_ids or [])
        self.test_env_ids = list(test_env_ids or [])
        self.schedule_mode = (schedule_mode or "iid").lower()
        self.scenario_schedule_mode = (scenario_schedule_mode or schedule_mode or "iid").lower()
        self.scenario_ids_by_phase = scenario_ids_by_phase or {}
        self.episode_idx = 0
        self.active_env_idx: Optional[int] = None
        self.phase: Optional[str] = None
        self.active_env_ids: Optional[List[int]] = None

        # RNG for env selection / fallback random actions
        self.rng = np.random.RandomState(seed)

        # validate shared action-space
        n_actions = envs[0].n_actions
        for e in envs[1:]:
            if e.n_actions != n_actions:
                raise ValueError("All envs in EnvPool must share the same n_actions")
        self._n_actions = n_actions
        self.ACTIONS = list(getattr(envs[0], "ACTIONS", []))

        # shared observation dimensions (take max to be safe)
        self.n_cell_types = max(getattr(e, "n_cell_types", 0) or 0 for e in envs) or 5
        self.n_scenarios = max(getattr(e, "n_scenarios", 0) or 1 for e in envs) or 1

    @property
    def env_id(self) -> int:
        if self.active_env_idx is None:
            return -1
        return getattr(self.envs[self.active_env_idx], "env_id", self.active_env_idx)

    @property
    def env_name(self) -> str:
        if self.active_env_idx is None:
            return "unknown"
        return getattr(self.envs[self.active_env_idx], "env_name", f"env_{self.active_env_idx}")

    @property
    def current_scenario_id(self) -> int:
        if self.active_env_idx is None:
            return 0
        return getattr(self.envs[self.active_env_idx], "current_scenario_id", 0)

    @property
    def current_scenario_name(self) -> str:
        if self.active_env_idx is None:
            return "single"
        return getattr(self.envs[self.active_env_idx], "current_scenario_name", "single")

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def action_meanings(self) -> Iterable[str]:
        return tuple(self.ACTIONS) if self.ACTIONS else tuple(self.envs[0].action_meanings)

    @property
    def max_steps(self) -> int:
        """
        Use minimum max_steps across envs to keep survival fraction well-defined.
        Assumes all envs share similar episode length scale.
        """
        return min(getattr(e, "max_steps", 0) or 0 for e in self.envs) or self.envs[0].max_steps

    def _get_active_env_ids(self) -> List[int]:
        """
        Return env ids filtered by current phase if set; otherwise all envs.
        """
        if self.active_env_ids is not None:
            return self.active_env_ids
        return list(range(self.n_envs))

    def _get_active_scenario_ids(self) -> List[int]:
        """
        Return scenario ids filtered by current phase if configured; otherwise all scenarios.
        """
        if self.phase and self.phase in self.scenario_ids_by_phase:
            return list(self.scenario_ids_by_phase[self.phase])
        if "all" in self.scenario_ids_by_phase:
            return list(self.scenario_ids_by_phase["all"])
        return list(range(self.n_scenarios or 1))

    def set_phase(self, phase: str):
        """
        Phase "A": only train envs.
        Phase "B": train + test envs.
        Phase "C": only test envs.
        Any other value resets to default (all envs).
        """
        self.phase = phase
        if phase == "A":
            ids = self.train_env_ids or list(range(self.n_envs))
        elif phase == "B":
            ids = (self.train_env_ids or []) + (self.test_env_ids or [])
            if not ids:
                ids = list(range(self.n_envs))
        elif phase == "C":
            ids = self.test_env_ids or self.train_env_ids or list(range(self.n_envs))
        else:
            self.active_env_ids = None
            return

        seen = set()
        filtered: List[int] = []
        for eid in ids:
            if eid in seen:
                continue
            if 0 <= eid < self.n_envs:
                seen.add(eid)
                filtered.append(eid)
        self.active_env_ids = filtered

    def _sample_env_id(self, mode: str = "all") -> int:
        """
        mode: "all" | "train" | "test"
        """
        base_ids = self._get_active_env_ids()
        if mode == "train" and self.train_env_ids:
            ids = [i for i in base_ids if i in self.train_env_ids]
            if not ids:
                ids = self.train_env_ids
        elif mode == "test" and self.test_env_ids:
            ids = [i for i in base_ids if i in self.test_env_ids]
            if not ids:
                ids = self.test_env_ids
        else:
            ids = base_ids

        if self.schedule_mode == "round_robin":
            return ids[self.episode_idx % len(ids)]
        # default iid
        return int(ids[self.rng.randint(0, len(ids))])

    def _sample_scenario_id(self, mode: str = "all") -> Optional[int]:
        """
        mode: "all" | "train" | "test"
        Follows same scheduling logic as env sampling but over scenario ids (for envs that use them).
        """
        scenario_ids = self._get_active_scenario_ids()
        if not scenario_ids:
            return None
        if self.scenario_schedule_mode == "round_robin":
            return scenario_ids[self.episode_idx % len(scenario_ids)]
        # default iid
        return int(scenario_ids[self.rng.randint(0, len(scenario_ids))])

    def reset(self, scenario_id: Optional[int] = None, split: Optional[str] = None) -> Dict[str, Any]:
        self.episode_idx += 1
        mode = split or "all"
        idx = self._sample_env_id(mode=mode)
        self.active_env_idx = idx
        env = self.envs[idx]
        chosen_scenario = scenario_id
        if chosen_scenario is None:
            if self.scenario_ids_by_phase and (self.n_envs == 1 or getattr(env, "multi_task", False)):
                chosen_scenario = self._sample_scenario_id(mode=mode)
        obs = env.reset(scenario_id=chosen_scenario)
        obs = dict(obs)
        obs["env_id"] = getattr(env, "env_id", idx)
        obs["env_name"] = getattr(env, "env_name", f"env_{idx}")
        if "env_family" not in obs and hasattr(env, "env_family"):
            obs["env_family"] = getattr(env, "env_family")
        # ensure scenario fields are present for downstream logging
        if "scenario_id" not in obs and chosen_scenario is not None:
            obs["scenario_id"] = int(chosen_scenario)
        if "scenario_name" not in obs and hasattr(env, "current_scenario_name"):
            obs["scenario_name"] = getattr(env, "current_scenario_name", "scenario")
        return obs

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.active_env_idx is None:
            raise RuntimeError("EnvPool.step called before reset")
        env = self.envs[self.active_env_idx]
        obs, reward, done, info = env.step(action)
        obs = dict(obs)
        info = dict(info)
        env_id = getattr(env, "env_id", self.active_env_idx)
        env_name = getattr(env, "env_name", f"env_{self.active_env_idx}")
        obs["env_id"] = env_id
        obs["env_name"] = env_name
        info["env_id"] = env_id
        info["env_name"] = env_name
        if "env_family" not in obs and hasattr(env, "env_family"):
            obs["env_family"] = getattr(env, "env_family")
        if "env_family" not in info and "env_family" in obs:
            info["env_family"] = obs["env_family"]
        if "scenario_id" not in obs and hasattr(env, "current_scenario_id"):
            obs["scenario_id"] = getattr(env, "current_scenario_id", 0)
        if "scenario_name" not in obs and hasattr(env, "current_scenario_name"):
            obs["scenario_name"] = getattr(env, "current_scenario_name", "single")
        if "scenario_id" not in info and "scenario_id" in obs:
            info["scenario_id"] = obs["scenario_id"]
        if "scenario_name" not in info and "scenario_name" in obs:
            info["scenario_name"] = obs["scenario_name"]
        info = normalize_info_contract(
            info,
            done=bool(done),
            reward_env=float(info.get("reward_env", reward)),
            terminated_reason=str(info.get("terminated_reason", info.get("reason", "")) or ""),
            success=(info.get("success") if "success" in info else None),
            constraint_violation=(bool(info.get("constraint_violation")) if "constraint_violation" in info else None),
            catastrophic=(bool(info.get("catastrophic")) if "catastrophic" in info else None),
            timeout=(bool(info.get("timeout")) if "timeout" in info else None),
            events=(info.get("events") if isinstance(info.get("events"), dict) else None),
        )
        return obs, reward, done, info

    def sample_random_action(self) -> int:
        return int(self.rng.randint(0, self._n_actions))

    def get_obs_spec(self) -> Dict[str, Any]:
        return {
            "n_envs": self.n_envs,
            "n_actions": self._n_actions,
            "n_cell_types": self.n_cell_types,
            "n_scenarios": self.n_scenarios,
            "obs_fields": ["patch", "energy", "scenario_id", "env_id"],
        }

    def get_action_mask(self) -> Optional[np.ndarray]:
        if self.active_env_idx is None:
            return None
        env = self.envs[self.active_env_idx]
        if hasattr(env, "get_action_mask"):
            try:
                return env.get_action_mask()
            except Exception:
                return None
        return None

    def set_action_mask_enabled(self, enabled: bool) -> None:
        for env in self.envs:
            if hasattr(env, "set_action_mask_enabled"):
                try:
                    env.set_action_mask_enabled(enabled)
                except Exception:
                    continue

    def get_env_descriptor(self, env_id: int) -> np.ndarray:
        return self.envs[env_id].get_env_descriptor()

    def get_all_descriptors(self) -> List[np.ndarray]:
        """
        Return env descriptors for all pooled envs.
        """
        return [self.get_env_descriptor(i) for i in range(self.n_envs)]
