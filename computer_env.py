"""
ComputerEnv v0: simulated computer/project environment with discrete actions.

The environment abstracts away real file operations while exposing progress
metrics (tests passed, files touched) so that agents can practice task
sequencing and test-running behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np

from env import BaseEnv, build_env_descriptor


@dataclass
class ComputerState:
    files_opened: int
    files_modified: int
    tests_passed: int
    tests_total: int
    last_test_passed: bool
    steps_taken: int


@dataclass
class ComputerTask:
    task_id: int
    name: str
    difficulty: float  # [0,1]
    max_steps: int
    initial_state: ComputerState
    target_state: ComputerState
    description: str = ""


@dataclass
class TaskSet:
    tasks: List[ComputerTask]


@dataclass
class ComputerEnvConfig:
    step_penalty: float = -0.01
    test_reward: float = 1.0
    full_test_fail_penalty: float = -0.1
    bonus_complete: float = 5.0
    rng_seed: Optional[int] = None
    patch_size: int = 5
    progress_token_max: int = 8
    safe_pass_prob: float = 0.5
    full_pass_prob: float = 0.7
    pass_noise: float = 0.1


class ComputerEnv(BaseEnv):
    """
    Discrete, aggregated computer-world environment.

    Actions:
      0: NO_OP
      1: OPEN_FILE      -> increases files_opened
      2: MODIFY_FILE    -> increases files_modified
      3: RUN_TESTS_SAFE -> lower reward, lower failure penalty
      4: RUN_TESTS_FULL -> higher reward potential, higher failure penalty
      5: SWITCH_TASK    -> cycle to next task in the TaskSet
    """

    ACTIONS = [
        "NO_OP",
        "OPEN_FILE",
        "MODIFY_FILE",
        "RUN_TESTS_SAFE",
        "RUN_TESTS_FULL",
        "SWITCH_TASK",
    ]

    def __init__(
        self,
        task_set: TaskSet,
        config: Optional[ComputerEnvConfig] = None,
        env_id: int = 0,
        env_name: str = "computer_env",
        seed: Optional[int] = None,
    ):
        self.task_set = task_set
        self.config = config or ComputerEnvConfig()
        self._env_id = int(env_id)
        self._env_name = env_name
        self.rng = np.random.RandomState(seed if seed is not None else self.config.rng_seed)
        self.view_size = self.config.patch_size
        self.n_cell_types = max(6, self.config.progress_token_max + 1)
        self._n_actions = len(self.ACTIONS)
        self.current_task_idx: int = 0
        self.state: Optional[ComputerState] = None
        self.current_task: Optional[ComputerTask] = None
        self.env_family = "computer-basic"
        self.n_scenarios = max(1, len(self.task_set.tasks))
        self.current_scenario_id = 0
        self.current_scenario_name = self.task_set.tasks[0].name if self.task_set.tasks else "task0"
        self.description = "Мир: проект с тестами, цель – добиться, чтобы все тесты проходили."
        self._env_descriptor = self._compute_env_descriptor()
        self.scenario_configs = [
            {"name": t.name, "description": t.description, "task_id": t.task_id} for t in self.task_set.tasks
        ]

    # ----- BaseEnv compatibility -----
    @property
    def env_id(self) -> int:
        return self._env_id

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def action_meanings(self):
        return tuple(self.ACTIONS)

    @property
    def max_steps(self) -> int:
        if self.current_task is not None:
            return self.current_task.max_steps
        if self.task_set.tasks:
            return self.task_set.tasks[0].max_steps
        return 50

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def sample_random_action(self) -> int:
        return int(self.rng.randint(0, self.n_actions))

    # ----- Core helpers -----
    def _compute_env_descriptor(self) -> np.ndarray:
        max_steps = self.max_steps
        tests_total = self.current_task.target_state.tests_total if self.current_task else 1
        difficulty = self.current_task.difficulty if self.current_task else 0.0
        return build_env_descriptor(
            env_family=self.env_family,
            width=float(tests_total),
            height=float(max_steps),
            goal_density=float(tests_total) / max(1.0, float(max_steps)),
            danger_density=float(difficulty),
            wall_density=0.0,
            has_door=False,
            has_key=False,
            has_lava=False,
            max_steps=float(max_steps),
        )

    def _select_task(self, task_id: Optional[int] = None) -> ComputerTask:
        if not self.task_set.tasks:
            raise ValueError("ComputerEnv requires at least one task")
        if task_id is None:
            task_idx = int(self.rng.randint(0, len(self.task_set.tasks)))
        else:
            task_idx = int(task_id) % len(self.task_set.tasks)
        self.current_task_idx = task_idx
        task = self.task_set.tasks[task_idx]
        self.current_task = task
        self.current_scenario_id = task.task_id
        self.current_scenario_name = task.name
        return task

    def reset(self, task_id: Optional[int] = None, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        if scenario_id is not None and task_id is None:
            task_id = scenario_id
        task = self._select_task(task_id)
        self.state = copy.deepcopy(task.initial_state)
        self._env_descriptor = self._compute_env_descriptor()
        return self._get_obs()

    def _encode_patch_token(self) -> int:
        assert self.state is not None and self.current_task is not None
        progress = self.state.tests_passed / max(1, self.current_task.target_state.tests_total)
        token = int(progress * self.config.progress_token_max)
        token = max(0, min(self.config.progress_token_max, token))
        return token

    def _get_obs(self) -> Dict[str, Any]:
        assert self.state is not None and self.current_task is not None
        remaining = max(0, self.current_task.max_steps - self.state.steps_taken)
        progress = self.state.tests_passed / max(1, self.current_task.target_state.tests_total)
        patch = np.full(
            (self.view_size, self.view_size),
            fill_value=self._encode_patch_token(),
            dtype=np.int64,
        )
        obs: Dict[str, Any] = {
            "patch": patch,
            "energy": float(progress),
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_family": self.env_family,
            "files_opened": int(self.state.files_opened),
            "files_modified": int(self.state.files_modified),
            "tests_passed": int(self.state.tests_passed),
            "tests_total": int(self.state.tests_total),
            "progress": float(progress),
            "steps_taken": int(self.state.steps_taken),
            "remaining_steps": int(remaining),
            "last_test_passed": int(self.state.last_test_passed),
            "description": self.current_task.description or self.description,
        }
        return obs

    def _apply_test_run(self, safe: bool) -> Tuple[int, bool]:
        """
        Simulate a test run. Returns (delta_passed, last_passed).
        """
        assert self.state is not None and self.current_task is not None
        remaining = max(0, self.state.tests_total - self.state.tests_passed)
        if remaining == 0:
            return 0, True
        base = self.config.safe_pass_prob if safe else self.config.full_pass_prob
        prob = np.clip(base - self.current_task.difficulty * self.config.pass_noise, 0.0, 1.0)
        passed = int(self.rng.binomial(remaining, prob))
        passed = max(0, min(remaining, passed))
        last_passed = passed > 0
        self.state.tests_passed += passed
        return passed, last_passed

    def _switch_task(self):
        if not self.task_set.tasks:
            return
        next_idx = (self.current_task_idx + 1) % len(self.task_set.tasks)
        self.reset(task_id=next_idx)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self.state is not None and self.current_task is not None
        action = int(action)
        reward = float(self.config.step_penalty)
        info: Dict[str, Any] = {
            "scenario_id": self.current_scenario_id,
            "scenario_name": self.current_scenario_name,
            "env_id": self.env_id,
            "env_name": self.env_name,
            "alive": True,
            "death_flag": 0.0,
            "description": self.current_task.description or self.description,
        }

        self.state.steps_taken += 1
        if action == 1:  # OPEN_FILE
            self.state.files_opened += 1
        elif action == 2:  # MODIFY_FILE
            self.state.files_modified += 1
        elif action == 3:  # RUN_TESTS_SAFE
            delta, last_passed = self._apply_test_run(safe=True)
            reward += self.config.test_reward * delta
            info["tests_gained"] = delta
            info["last_test_passed"] = bool(last_passed)
            self.state.last_test_passed = bool(last_passed)
        elif action == 4:  # RUN_TESTS_FULL
            delta, last_passed = self._apply_test_run(safe=False)
            reward += self.config.test_reward * delta
            if delta == 0:
                reward += self.config.full_test_fail_penalty
            info["tests_gained"] = delta
            info["last_test_passed"] = bool(last_passed)
            self.state.last_test_passed = bool(last_passed)
        elif action == 5:  # SWITCH_TASK
            self._switch_task()
            obs = self._get_obs()
            return obs, reward, False, info
        # NO_OP just incurs step penalty

        done = False
        if self.state.tests_passed >= self.state.tests_total:
            reward += self.config.bonus_complete
            done = True
        if self.state.steps_taken >= self.current_task.max_steps:
            done = True

        obs = self._get_obs()
        info["reward_env"] = reward
        info["progress"] = obs["progress"]
        info["tests_passed"] = obs["tests_passed"]
        info["tests_total"] = obs["tests_total"]
        info["steps_taken"] = obs["steps_taken"]
        info["remaining_steps"] = obs["remaining_steps"]
        return obs, reward, done, info

    def get_obs_spec(self) -> Dict[str, Any]:
        return {
            "n_cell_types": self.n_cell_types,
            "patch_size": self.view_size,
            "n_scenarios": self.n_scenarios,
            "fields": [
                "patch",
                "energy",
                "scenario_id",
                "env_id",
                "tests_passed",
                "tests_total",
                "steps_taken",
                "remaining_steps",
                "last_test_passed",
            ],
        }

    def get_env_descriptor(self) -> np.ndarray:
        return np.array(self._env_descriptor, copy=True)


def _make_task(
    task_id: int,
    name: str,
    difficulty: float,
    tests_total: int,
    max_steps: int,
    description: str,
) -> ComputerTask:
    init_state = ComputerState(
        files_opened=0,
        files_modified=0,
        tests_passed=0,
        tests_total=tests_total,
        last_test_passed=False,
        steps_taken=0,
    )
    target_state = ComputerState(
        files_opened=0,
        files_modified=0,
        tests_passed=tests_total,
        tests_total=tests_total,
        last_test_passed=True,
        steps_taken=max_steps,
    )
    return ComputerTask(
        task_id=task_id,
        name=name,
        difficulty=difficulty,
        max_steps=max_steps,
        initial_state=init_state,
        target_state=target_state,
        description=description,
    )


def build_computer_taskset(scenario_names: Optional[List[str]] = None, difficulty_shift: float = 0.0) -> TaskSet:
    presets = {
        "simple_project": {
            "difficulty": 0.2,
            "tests_total": 2,
            "max_steps": 30,
            "description": "Простой проект: несколько коротких тестов, минимальные правки.",
        },
        "refactor_project": {
            "difficulty": 0.4,
            "tests_total": 4,
            "max_steps": 40,
            "description": "Рефакторинг: нужно открыть и менять файлы, довести тесты до зелёного.",
        },
        "flaky_tests_project": {
            "difficulty": 0.6,
            "tests_total": 5,
            "max_steps": 50,
            "description": "Флейки: тесты могут проходить не всегда, ценится стабильность и экономия шагов.",
        },
    }
    names = scenario_names or list(presets.keys())
    tasks: List[ComputerTask] = []
    for idx, name in enumerate(names):
        preset = presets.get(name, None)
        if preset is None:
            # fallback generic task
            preset = {
                "difficulty": 0.3 + 0.05 * idx,
                "tests_total": 3 + idx,
                "max_steps": 35 + 5 * idx,
                "description": f"Общий компьютерный режим '{name}'.",
            }
        task = _make_task(
            task_id=idx,
            name=name,
            difficulty=min(1.0, max(0.0, preset["difficulty"] + difficulty_shift)),
            tests_total=int(preset["tests_total"]),
            max_steps=int(preset["max_steps"]),
            description=str(preset.get("description", "")),
        )
        tasks.append(task)
    return TaskSet(tasks=tasks)
