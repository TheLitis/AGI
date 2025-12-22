"""
RepoToolEnv: a tiny on-disk "real tool loop" environment.

This is a bridge between the simulated ComputerEnv/ToolEnv and real workflows:
the agent acts in a sandbox directory, can apply a small set of predefined
patches, and runs `pytest` to verify progress.

Design constraints:
- Keep the observation contract compatible with existing Perception:
  a discrete `patch` (int64, 5x5) + scalar `energy`.
- Keep the action space size = 6 to stay compatible with EnvPool mixing.
- Avoid heavy dependencies; use subprocess + stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zlib

import numpy as np

from env import BaseEnv, build_env_descriptor


@dataclass
class RepoEdit:
    """
    A small text edit inside a file (string replace, not regex).
    Intended to represent "micro-patches" without rewriting whole files.
    """

    path: str
    find: str
    replace: str
    count: int = 1


@dataclass
class RepoPatch:
    """
    A patch candidate represented as file overwrites.

    The environment is intentionally discrete/finite: the agent selects among a
    small menu of patches instead of generating arbitrary code.
    """

    name: str
    files: Optional[Dict[str, str]] = None  # relative path -> file content
    edits: Optional[List[RepoEdit]] = None
    description: str = ""


@dataclass
class RepoTask:
    task_id: int
    name: str
    description: str
    initial_files: Dict[str, str]
    patches: List[RepoPatch]


@dataclass
class RepoToolEnvConfig:
    sandbox_root: str = "repo_sandboxes"
    max_steps: int = 24
    timeout_sec: float = 10.0
    patch_size: int = 5
    progress_token_max: int = 8
    hash_buckets: int = 64
    progress_reward_scale: float = 0.5
    shuffle_patch_bindings: bool = True
    failure_sig_tokens: int = 5
    patch_option_tokens: int = 5
    step_penalty: float = -0.01
    apply_patch_penalty: float = -0.02
    run_tests_penalty: float = -0.10
    revert_penalty: float = -0.02
    cycle_patches_penalty: float = -0.01
    success_reward: float = 5.0
    cleanup_on_reset: bool = True
    cleanup_on_done: bool = True
    keep_failed_sandboxes: bool = False
    pytest_args: Tuple[str, ...] = ("-q",)
    # Procedural task generation (for scenario names like "proc_arith", "proc", etc.)
    procedural_candidates: int = 8
    procedural_test_cases: int = 6
    procedural_max_int: int = 9
    # Tool-loop shaping (for procedural scenario tags like *_loop / *_toolloop / *_open).
    # The first failing RUN_TESTS is necessary to populate the failure focus + candidate menu,
    # so we reduce its penalty and optionally give a small bonus when candidates are created.
    toolloop_bootstrap_run_tests_penalty: float = -0.02
    toolloop_candidate_reward: float = 0.05
    toolloop_run_tests_penalty: float = -0.04
    toolloop_apply_without_candidates_penalty: float = -0.06
    toolloop_repeat_apply_penalty: float = -0.06


def _safe_relpath(path: str) -> str:
    rel = str(path).replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError(f"Path traversal is not allowed: {path!r}")
    return rel


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_pytest_counts(output: str) -> Tuple[int, int, int]:
    """
    Best-effort parse: return (passed, failed, errors).
    """
    text = output or ""
    passed = 0
    failed = 0
    errors = 0

    m = re.search(r"(?P<n>\d+)\s+passed", text)
    if m:
        passed = int(m.group("n"))
    m = re.search(r"(?P<n>\d+)\s+failed", text)
    if m:
        failed = int(m.group("n"))
    m = re.search(r"(?P<n>\d+)\s+errors?", text)
    if m:
        errors = int(m.group("n"))
    return passed, failed, errors


def _pytest_failure_signature(output: str) -> str:
    """
    Best-effort compact signature to hash into discrete token buckets.
    """
    text = (output or "").strip()
    if not text:
        return ""
    if "[timeout]" in text:
        return "timeout"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("FAILED ") or s.startswith("ERROR "):
            return s
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s
    return ""


_TOKEN_RE = re.compile(r"[a-z_]+|\d+|\*\*|//|==|!=|<=|>=|\+|-|\*|/|\^", flags=re.IGNORECASE)


def _tokenize_for_hash(text: str) -> List[str]:
    """
    Lightweight tokenizer that preserves a few operator tokens (e.g., '+', '//', '**')
    so the agent can condition on patch/error signatures without full NLP deps.
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(str(text))]


def _is_procedural_task_name(name: str) -> bool:
    n = (name or "").strip().lower()
    return n.startswith("proc") or n.startswith("procedural")


def _procedural_spec(name: str) -> Tuple[str, List[str]]:
    """
    Parse a procedural task name into (category, tags).

    Examples:
      - "proc" -> ("mixed", [])
      - "proc_arith" / "proc:arith" -> ("arith", [])
      - "proc_arith_ood" -> ("arith", ["ood"])
      - "procedural_refactor_hard" -> ("refactor", ["hard"])
    """
    raw = (name or "").strip().lower()
    for prefix in ("proc:", "procedural:", "proc_", "procedural_"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    raw = raw.strip("_: ")
    if not raw:
        return "mixed", []
    parts = [p for p in re.split(r"[_:]+", raw) if p]
    if not parts:
        return "mixed", []
    return parts[0] or "mixed", parts[1:]


def _procedural_category(name: str) -> str:
    """
    Parse a procedural task name into a coarse category.
    Examples:
      - "proc" -> "mixed"
      - "proc_arith" / "proc:arith" -> "arith"
      - "procedural_string" -> "string"
    """
    category, _tags = _procedural_spec(name)
    return category or "mixed"


def build_repo_taskset(scenario_names: Optional[List[str]] = None) -> List[RepoTask]:
    """
    Small curated taskset. Each task is a minimal repo with a failing pytest.
    """
    presets: Dict[str, Dict[str, Any]] = {
        "calc_add": {
            "description": "Fix `add(a,b)` so that tests pass.",
            "initial_files": {
                "calc.py": "def add(a, b):\n    return a - b\n",
                "test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_plus",
                    description="return a + b",
                    edits=[
                        RepoEdit(path="calc.py", find="return a - b", replace="return a + b", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_abs_diff",
                    description="return abs(a - b)",
                    edits=[
                        RepoEdit(path="calc.py", find="return a - b", replace="return abs(a - b)", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_plus_one",
                    description="return a + b + 1",
                    edits=[
                        RepoEdit(path="calc.py", find="return a - b", replace="return a + b + 1", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_str_concat",
                    description="return str(a) + str(b)",
                    edits=[
                        RepoEdit(path="calc.py", find="return a - b", replace="return str(a) + str(b)", count=1),
                    ],
                ),
            ],
        },
        "calc_div": {
            "description": "Fix `div(a,b)` to use true division.",
            "initial_files": {
                "calc.py": "def div(a, b):\n    return a // b\n",
                "test_calc.py": "from calc import div\n\n\ndef test_div():\n    assert div(3, 2) == 1.5\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_true_div",
                    description="return a / b",
                    files={"calc.py": "def div(a, b):\n    return a / b\n"},
                ),
                RepoPatch(
                    name="candidate_div_plus1",
                    description="return a / (b + 1)",
                    files={"calc.py": "def div(a, b):\n    return a / (b + 1)\n"},
                ),
            ],
        },
        "calc_pow": {
            "description": "Fix `power(a,b)` to compute exponentiation.",
            "initial_files": {
                "calc.py": "def power(a, b):\n    return a ^ b\n",
                "test_calc.py": "from calc import power\n\n\ndef test_power_small():\n    assert power(2, 3) == 8\n\n\ndef test_power_one():\n    assert power(5, 1) == 5\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_mul",
                    description="return a * b",
                    edits=[
                        RepoEdit(path="calc.py", find="return a ^ b", replace="return a * b", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_exp",
                    description="return a ** b",
                    edits=[
                        RepoEdit(path="calc.py", find="return a ^ b", replace="return a ** b", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_pow_builtin",
                    description="return pow(a, b)",
                    edits=[
                        RepoEdit(path="calc.py", find="return a ^ b", replace="return pow(a, b)", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_xor_nochange",
                    description="return a ^ b (no-op)",
                    edits=[
                        RepoEdit(path="calc.py", find="return a ^ b", replace="return a ^ b", count=1),
                    ],
                ),
            ],
        },
        "string_reverse": {
            "description": "Fix `reverse(s)` to reverse a string.",
            "initial_files": {
                "text.py": "def reverse(s):\n    return \"\".join(sorted(s))\n",
                "test_text.py": "from text import reverse\n\n\ndef test_reverse_abc():\n    assert reverse('abc') == 'cba'\n\n\ndef test_reverse_empty():\n    assert reverse('') == ''\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_slice",
                    description="return s[::-1]",
                    files={"text.py": "def reverse(s):\n    return s[::-1]\n"},
                ),
                RepoPatch(
                    name="candidate_identity",
                    description="return s",
                    files={"text.py": "def reverse(s):\n    return s\n"},
                ),
            ],
        },
        "list_sum": {
            "description": "Fix `sum_list(xs)` to return the sum of the list.",
            "initial_files": {
                "list_utils.py": "def sum_list(xs):\n    return max(xs)\n",
                "test_list_utils.py": "from list_utils import sum_list\n\n\ndef test_sum_list_small():\n    assert sum_list([1, 2, 3]) == 6\n\n\ndef test_sum_list_single():\n    assert sum_list([5]) == 5\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_len",
                    description="return len(xs)",
                    files={"list_utils.py": "def sum_list(xs):\n    return len(xs)\n"},
                ),
                RepoPatch(
                    name="candidate_sum",
                    description="return sum(xs)",
                    files={"list_utils.py": "def sum_list(xs):\n    return sum(xs)\n"},
                ),
            ],
        },
        "calc_bundle": {
            "description": "Fix both `add` and `div` (two tests, two files).",
            "initial_files": {
                "add.py": "def add(a, b):\n    return a - b\n",
                "div.py": "def div(a, b):\n    return a // b\n",
                "test_add.py": "from add import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                "test_div.py": "from div import div\n\n\ndef test_div():\n    assert div(3, 2) == 1.5\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_add_plus",
                    description="add.py: return a + b",
                    files={"add.py": "def add(a, b):\n    return a + b\n"},
                ),
                RepoPatch(
                    name="candidate_div_true",
                    description="div.py: return a / b",
                    files={"div.py": "def div(a, b):\n    return a / b\n"},
                ),
            ],
        },
        "calc_mod": {
            "description": "Fix `mod(a,b)` to return the remainder.",
            "initial_files": {
                "calc.py": "def mod(a, b):\n    return a // b\n",
                "test_calc.py": "from calc import mod\n\n\ndef test_mod_small():\n    assert mod(7, 3) == 1\n\n\ndef test_mod_equal():\n    assert mod(5, 5) == 0\n",
            },
            "patches": [
                RepoPatch(name="candidate_mod", description="return a % b", files={"calc.py": "def mod(a, b):\n    return a % b\n"}),
                RepoPatch(
                    name="candidate_sub",
                    description="return a - b",
                    files={"calc.py": "def mod(a, b):\n    return a - b\n"},
                ),
            ],
        },
        "calc_clamp": {
            "description": "Fix `clamp(x, lo, hi)` to clamp within bounds.",
            "initial_files": {
                "util.py": "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    return hi\n",
                "test_util.py": "from util import clamp\n\n\ndef test_clamp_inside():\n    assert clamp(5, 0, 10) == 5\n\n\ndef test_clamp_low():\n    assert clamp(-1, 0, 10) == 0\n\n\ndef test_clamp_high():\n    assert clamp(11, 0, 10) == 10\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_minmax",
                    description="return max(lo, min(hi, x))",
                    files={"util.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n"},
                ),
                RepoPatch(
                    name="candidate_maxmin_swapped",
                    description="return min(lo, max(hi, x))",
                    files={"util.py": "def clamp(x, lo, hi):\n    return min(lo, max(hi, x))\n"},
                ),
            ],
        },
        "string_remove_digits": {
            "description": "Fix `remove_digits(s)` to drop digits from a string.",
            "initial_files": {
                "text.py": "def remove_digits(s):\n    return s\n",
                "test_text.py": "from text import remove_digits\n\n\ndef test_remove_digits_mixed():\n    assert remove_digits('a1b2c3') == 'abc'\n\n\ndef test_remove_digits_none():\n    assert remove_digits('xyz') == 'xyz'\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_filter_isdigit",
                    description="filter: not c.isdigit()",
                    files={"text.py": "def remove_digits(s):\n    return ''.join([c for c in s if not c.isdigit()])\n"},
                ),
                RepoPatch(
                    name="candidate_keep_digits",
                    description="keep only digits",
                    files={"text.py": "def remove_digits(s):\n    return ''.join([c for c in s if c.isdigit()])\n"},
                ),
            ],
        },
        "list_mean": {
            "description": "Fix `mean(xs)` to return the arithmetic mean as float.",
            "initial_files": {
                "stats.py": "def mean(xs):\n    return sum(xs)\n",
                "test_stats.py": "from stats import mean\n\n\ndef test_mean_small():\n    assert mean([1, 2, 3]) == 2.0\n\n\ndef test_mean_fractional():\n    assert mean([1, 2]) == 1.5\n\n\ndef test_mean_single():\n    assert mean([5]) == 5.0\n",
            },
            "patches": [
                RepoPatch(
                    name="candidate_mean_div_len",
                    description="return sum(xs) / len(xs)",
                    edits=[
                        RepoEdit(path="stats.py", find="return sum(xs)", replace="return sum(xs) / len(xs)", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_mean_intdiv",
                    description="return sum(xs) // len(xs)",
                    edits=[
                        RepoEdit(path="stats.py", find="return sum(xs)", replace="return sum(xs) // len(xs)", count=1),
                    ],
                ),
                RepoPatch(
                    name="candidate_mean_plus_len",
                    description="return (sum(xs) + len(xs)) / len(xs)",
                    edits=[
                        RepoEdit(
                            path="stats.py",
                            find="return sum(xs)",
                            replace="return (sum(xs) + len(xs)) / len(xs)",
                            count=1,
                        ),
                    ],
                ),
                RepoPatch(
                    name="candidate_mean_float_cast",
                    description="return float(sum(xs)) / len(xs)",
                    edits=[
                        RepoEdit(
                            path="stats.py",
                            find="return sum(xs)",
                            replace="return float(sum(xs)) / len(xs)",
                            count=1,
                        ),
                    ],
                ),
            ],
        },
    }

    names = scenario_names or list(presets.keys())
    tasks: List[RepoTask] = []
    for idx, name in enumerate(names):
        if _is_procedural_task_name(str(name)):
            tasks.append(
                RepoTask(
                    task_id=idx,
                    name=str(name),
                    description="Procedural repo task (generated on reset).",
                    initial_files={},
                    patches=[],
                )
            )
            continue
        preset = presets.get(name)
        if preset is None:
            preset = {
                "description": f"Generic repo task: {name}",
                "initial_files": {
                    "main.py": "def answer():\n    return 41\n",
                    "test_main.py": "from main import answer\n\n\ndef test_answer():\n    assert answer() == 42\n",
                },
                "patches": [
                    RepoPatch(name="fix_answer_42", files={"main.py": "def answer():\n    return 42\n"}),
                    RepoPatch(name="wrong_answer_0", files={"main.py": "def answer():\n    return 0\n"}),
                ],
            }
        tasks.append(
            RepoTask(
                task_id=idx,
                name=str(name),
                description=str(preset.get("description", "")),
                initial_files=dict(preset.get("initial_files") or {}),
                patches=list(preset.get("patches") or []),
            )
        )
    return tasks


class RepoToolEnv(BaseEnv):
    """
    Real tool-loop environment backed by an on-disk sandbox.

    Actions (fixed size=6):
      0: NO_OP (inspect: cycles observation view)
      1: APPLY_PATCH_0
      2: APPLY_PATCH_1
      3: RUN_TESTS (pytest)
      4: REVERT (restore initial files)
      5: CYCLE_PATCHES (show a new patch pair)
    """

    ACTIONS = ["NO_OP", "APPLY_PATCH_0", "APPLY_PATCH_1", "RUN_TESTS", "REVERT", "CYCLE_PATCHES"]

    def __init__(
        self,
        task_set: List[RepoTask],
        config: Optional[RepoToolEnvConfig] = None,
        env_id: int = 0,
        env_name: str = "repo_tool_env",
        seed: Optional[int] = None,
    ):
        self.task_set = list(task_set)
        if not self.task_set:
            raise ValueError("RepoToolEnv requires at least one RepoTask")
        self.config = config or RepoToolEnvConfig()
        self._env_id = int(env_id)
        self._env_name = str(env_name)
        self.env_family = "repo-basic"
        self.rng = np.random.RandomState(seed)
        self.view_size = int(self.config.patch_size)
        self._n_actions = len(self.ACTIONS)
        # Reserve a small block of tokens for non-hashed control signals (flags/view/action).
        # Keep hash tokens disjoint from those reserved ids to reduce collisions.
        self.n_cell_types = int(self.config.progress_token_max) + 32 + int(max(0, self.config.hash_buckets))

        self.n_scenarios = max(1, len(self.task_set))
        self.scenario_configs = [
            {"name": t.name, "description": t.description, "task_id": t.task_id} for t in self.task_set
        ]
        self.current_scenario_id = 0
        self.current_scenario_name = self.task_set[0].name
        self.description = "Sandbox repo tasks: apply patch candidates and run pytest."

        self.workdir: Optional[Path] = None
        self.current_task_idx: int = 0
        self.current_task: Optional[RepoTask] = None
        self.steps_taken: int = 0
        self.patches_applied: List[bool] = [False, False]
        # Track which patch index was last applied in each action slot, so we can
        # discourage redundant "apply-spam" in tool-loop scenarios.
        self.last_applied_patch_idx: List[Optional[int]] = [None, None]
        # Map action slots (APPLY_PATCH_0 / APPLY_PATCH_1) -> patch index in current_task.patches
        self.action_patch_indices: List[int] = [0, 1]
        # Patch navigation state (when a task has >2 patch candidates).
        # `patch_order` is a (possibly shuffled) permutation of patch indices.
        # `patch_cursor` selects the current pair: (cursor, cursor+1).
        self.patch_order: List[int] = []
        self.patch_cursor: int = 0
        # Observation "inspection" mode:
        #   0 = patch option view, 1 = file list view, 2 = pytest output view, 3 = focus snippet.
        self.view_mode: int = 0
        self.last_test_passed: Optional[bool] = None
        self.last_tests_passed: int = 0
        self.last_tests_total: int = 0
        self.last_pytest_output: str = ""
        self.workspace_dirty: bool = True
        # Failure focus state (used for inspection/tool-loop support).
        self.focus_func: Optional[str] = None
        self.focus_file: Optional[str] = None
        self.focus_text: str = ""
        # Dynamic/tool-loop patch generation can key off the last failure signature.
        self._last_failure_sig_for_candidates: str = ""
        self._env_descriptor = self._compute_env_descriptor()

    # ----- BaseEnv compatibility -----
    @property
    def env_id(self) -> int:
        return self._env_id

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def action_meanings(self):
        return tuple(self.ACTIONS)

    @property
    def max_steps(self) -> int:
        return int(self.config.max_steps)

    def sample_random_action(self) -> int:
        return int(self.rng.randint(0, self.n_actions))

    # ----- Sandbox helpers -----
    def _sandbox_root(self) -> Path:
        root = Path(self.config.sandbox_root)
        if not root.is_absolute():
            root = Path(os.getcwd()) / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _cleanup_workdir(self, keep: bool = False) -> None:
        if self.workdir is None:
            return
        path = self.workdir
        self.workdir = None
        if keep:
            return
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    def _materialize_task(self, task: RepoTask) -> None:
        if self.config.cleanup_on_reset:
            self._cleanup_workdir(keep=False)
        root = self._sandbox_root()
        self.workdir = Path(tempfile.mkdtemp(prefix=f"repo_{self._env_id}_{task.name}_", dir=str(root)))
        for rel, content in task.initial_files.items():
            rp = _safe_relpath(rel)
            _write_text(self.workdir / rp, content)
        self.workspace_dirty = True

    def _apply_patch(self, patch: RepoPatch) -> None:
        if self.workdir is None:
            raise RuntimeError("RepoToolEnv patch applied before reset()")
        for rel, content in (patch.files or {}).items():
            rp = _safe_relpath(rel)
            _write_text(self.workdir / rp, content)
        for edit in patch.edits or []:
            rp = _safe_relpath(edit.path)
            target = self.workdir / rp
            if not target.exists() or not target.is_file():
                continue
            find = str(edit.find or "")
            if not find:
                continue
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            count = int(getattr(edit, "count", 1) or 0)
            if count <= 0:
                updated = text.replace(find, str(edit.replace or ""))
            else:
                updated = text.replace(find, str(edit.replace or ""), count)
            if updated != text:
                _write_text(target, updated)
        self.workspace_dirty = True

    def _revert(self) -> None:
        if self.current_task is None:
            return
        self._materialize_task(self.current_task)
        self.patches_applied = [False, False]
        self.last_applied_patch_idx = [None, None]
        self.view_mode = 0
        self.last_test_passed = None
        self.last_tests_passed = 0
        self.last_tests_total = 0
        self.last_pytest_output = ""
        self.focus_func = None
        self.focus_file = None
        self.focus_text = ""
        self._last_failure_sig_for_candidates = ""

    def _choose_action_patch_indices(self, task: Optional[RepoTask]) -> List[int]:
        """
        Initialize (or reset) the patch-candidate selection state for a task.

        For tasks with >2 candidates, action 5 can cycle through disjoint pairs of
        candidates so the agent is never stuck with an unhelpful initial pair.
        """
        self.patch_order = []
        self.patch_cursor = 0

        if task is None or not task.patches:
            return [0, 0]
        n = int(len(task.patches))
        if n == 1:
            return [0, 0]

        order = list(range(n))
        shuffle = bool(self.config.shuffle_patch_bindings) and n > 2
        if shuffle and self._is_toolloop_task(task):
            shuffle = False
        if shuffle:
            order = [int(x) for x in self.rng.permutation(n)]
        self.patch_order = order
        self.patch_cursor = 0
        return self._current_patch_pair(task)

    def _current_patch_pair(self, task: Optional[RepoTask] = None) -> List[int]:
        task_obj = task or self.current_task
        if task_obj is None or not task_obj.patches:
            return [0, 0]
        n = int(len(task_obj.patches))
        if n <= 1:
            return [0, 0]
        if not self.patch_order or len(self.patch_order) != n:
            self.patch_order = list(range(n))
        i0 = int(self.patch_order[self.patch_cursor % n])
        i1 = int(self.patch_order[(self.patch_cursor + 1) % n])
        return [i0, i1]

    def _cycle_patches(self) -> None:
        task_obj = self.current_task
        if task_obj is None or not task_obj.patches:
            return
        n = int(len(task_obj.patches))
        if n <= 2:
            return
        if not self.patch_order or len(self.patch_order) != n:
            self.patch_order = list(range(n))
        # move to the next pair (0,1)->(2,3)->(4,5)->...
        self.patch_cursor = int((self.patch_cursor + 2) % n)
        self.action_patch_indices = self._current_patch_pair(task_obj)

    def _cycle_view_mode(self) -> None:
        # 0 = patch options, 1 = file list, 2 = pytest output, 3 = focused snippet.
        self.view_mode = int((int(getattr(self, "view_mode", 0)) + 1) % 4)

    def _extract_focus_func(self, pytest_output: str) -> Optional[str]:
        text = (pytest_output or "").strip()
        if not text:
            return None
        for line in text.splitlines():
            m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if m:
                return str(m.group(1))
        m = re.search(r"FAILED\s+\S+::([A-Za-z_][A-Za-z0-9_]*)", text)
        if m:
            name = str(m.group(1))
            if name.startswith("test_") and len(name) > 5:
                return name[5:]
            return name
        return None

    def _find_def_file(self, func_name: str) -> Optional[Path]:
        if self.workdir is None:
            return None
        name = str(func_name or "").strip()
        if not name:
            return None

        pat = re.compile(rf"^\s*def\s+{re.escape(name)}\s*\(", flags=re.MULTILINE)
        candidates: List[Tuple[int, Path]] = []
        for p in self.workdir.rglob("*.py"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.workdir)).replace("\\", "/")
            penalty = 0
            if "/tests/" in f"/{rel}/" or rel.startswith("tests/"):
                penalty += 10
            if p.name.startswith("test_"):
                penalty += 10
            if p.name == "conftest.py":
                penalty += 5
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if pat.search(content):
                candidates.append((penalty, p))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], str(x[1])))
        return candidates[0][1]

    def _update_failure_focus(self) -> None:
        """
        Populate focus_* based on the last pytest output.

        Goal: give the agent a low-bandwidth "cursor" pointing to where the bug likely is.
        """
        if bool(self.last_test_passed):
            self.focus_func = None
            self.focus_file = None
            self.focus_text = ""
            return
        out = str(self.last_pytest_output or "")
        func = self._extract_focus_func(out)
        if not func:
            self.focus_func = None
            self.focus_file = None
            self.focus_text = ""
            return

        p = self._find_def_file(func)
        if p is None or self.workdir is None:
            self.focus_func = None
            self.focus_file = None
            self.focus_text = ""
            return

        rel = str(p.relative_to(self.workdir)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            self.focus_func = None
            self.focus_file = None
            self.focus_text = ""
            return

        lines = text.splitlines()
        def_idx: Optional[int] = None
        ret_idx: Optional[int] = None
        def_pat = re.compile(rf"^\s*def\s+{re.escape(func)}\s*\(")
        for i, line in enumerate(lines):
            if def_pat.search(line):
                def_idx = i
                break
        if def_idx is not None:
            for j in range(def_idx + 1, len(lines)):
                if "return " in lines[j]:
                    ret_idx = j
                    break
        parts: List[str] = [rel]
        if def_idx is not None:
            parts.append(lines[def_idx].strip())
        if ret_idx is not None:
            parts.append(lines[ret_idx].strip())
        self.focus_func = str(func)
        self.focus_file = rel
        self.focus_text = " | ".join([p for p in parts if p])

    def _is_toolloop_task(self, task: Optional[RepoTask] = None) -> bool:
        task_obj = task or self.current_task
        if task_obj is None:
            return False
        if not _is_procedural_task_name(getattr(task_obj, "name", "")):
            return False
        _cat, tags = _procedural_spec(getattr(task_obj, "name", ""))
        tagset = {str(t).lower() for t in tags}
        return bool(tagset.intersection({"loop", "toolloop", "open"}))

    def _refresh_toolloop_candidates(self) -> None:
        """
        Generate a fresh candidate patch menu based on the current failure focus.

        This is still discrete (action-space stays size=6), but it turns the
        repo task into a multi-step "run tests -> inspect -> edit -> rerun"
        loop instead of a one-shot multiple-choice patch selection.
        """
        if self.current_task is None or self.workdir is None:
            return
        if not self.focus_func or not self.focus_file:
            return

        rel = _safe_relpath(self.focus_file)
        path = self.workdir / rel
        if not path.exists() or not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        lines = text.splitlines()
        if not lines:
            return

        func = str(self.focus_func)
        def_line = None
        args_raw = ""
        for line in lines:
            m = re.match(rf"^\s*def\s+{re.escape(func)}\s*\((?P<args>[^)]*)\)\s*:", line)
            if m:
                def_line = line
                args_raw = str(m.group("args") or "")
                break
        if def_line is None:
            return

        args: List[str] = []
        for a in args_raw.split(","):
            s = str(a).strip()
            if not s:
                continue
            s = s.split("=", 1)[0].strip()
            if s:
                args.append(s)

        ret_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("return "):
                ret_idx = i
                break
        if ret_idx is None:
            return

        def _ws_prefix(s: str) -> str:
            return s[: len(s) - len(s.lstrip())]

        a0 = args[0] if len(args) >= 1 else "a"
        a1 = args[1] if len(args) >= 2 else "b"

        pool: List[str]
        correct: Optional[str] = None
        fname = func.lower()
        if fname in {"add", "plus"}:
            correct = f"{a0} + {a1}"
            pool = [
                f"{a0} - {a1}",
                f"{a1} - {a0}",
                f"abs({a0} - {a1})",
                f"{a0} * {a1}",
                f"{a0} + {a1} + 1",
                f"str({a0}) + str({a1})",
            ]
        elif fname in {"div", "divide"}:
            correct = f"{a0} / {a1}"
            pool = [
                f"{a0} // {a1}",
                f"{a1} / {a0}",
                f"{a0} / ({a1} + 1)",
                f"{a0} * {a1}",
            ]
        elif fname in {"power", "pow"}:
            correct = f"{a0} ** {a1}"
            pool = [
                f"{a0} ^ {a1}",
                f"{a0} * {a1}",
                f"pow({a0}, {a1})",
                f"{a0} ** ({a1} + 1)",
            ]
        elif fname in {"reverse"} and len(args) >= 1:
            s = args[0]
            correct = f"{s}[::-1]"
            pool = [
                s,
                f"\"\".join(sorted({s}))",
                f"\"\".join(reversed({s}))",
                f"{s}[::1]",
                f"{s}[::-1][1:]",
            ]
        elif fname in {"mean"} and len(args) >= 1:
            xs = args[0]
            correct = f"sum({xs}) / len({xs})"
            pool = [
                f"sum({xs})",
                f"len({xs})",
                f"max({xs})",
                f"sum({xs}) / max(1, len({xs}))",
            ]
        elif fname in {"clamp"} and len(args) >= 3:
            x, lo, hi = args[0], args[1], args[2]
            correct = f"max({lo}, min({hi}, {x}))"
            pool = [
                f"min({lo}, max({hi}, {x}))",
                f"min({hi}, max({lo}, {x}))",
                f"max({hi}, min({lo}, {x}))",
            ]
        else:
            return

        pool_full = list(pool)
        if correct is not None:
            pool_full = pool_full + [correct]

        cfg = self.config
        required = int(max(4, getattr(cfg, "procedural_candidates", 8) or 8))
        if self._is_toolloop_task(self.current_task):
            required = int(min(required, 4))
            required = int(max(2, required))
        if required % 2 == 1:
            required += 1

        cand_exprs = self._proc_pick_candidates(
            self.rng,
            required=required,
            must_include=[str(correct)] if correct is not None else [],
            pool=pool_full,
        )
        if correct is not None:
            corr = str(correct)
            if corr in cand_exprs:
                cand_exprs = [corr] + [e for e in cand_exprs if e != corr]

        indent = _ws_prefix(lines[ret_idx])
        patches: List[RepoPatch] = []
        ends_with_nl = bool(text.endswith("\n"))
        for i, expr in enumerate(cand_exprs):
            updated_lines = list(lines)
            updated_lines[ret_idx] = f"{indent}return {expr}"
            updated = "\n".join(updated_lines) + ("\n" if ends_with_nl else "")
            patches.append(
                RepoPatch(
                    name=f"auto_{fname}_{i}",
                    description=f"{rel}: return {expr}",
                    files={rel: updated},
                )
            )

        if not patches:
            return

        self.current_task.patches = patches
        self.patches_applied = [False, False]
        self.last_applied_patch_idx = [None, None]
        self.action_patch_indices = self._choose_action_patch_indices(self.current_task)

    def _switch_task(self, task_idx: int) -> None:
        task_idx = int(task_idx) % len(self.task_set)
        self.current_task_idx = task_idx
        self.current_task = self.task_set[task_idx]
        self.current_scenario_id = int(task_idx)
        self.current_scenario_name = str(self.current_task.name)
        self._maybe_generate_procedural_task(self.current_task)
        self.patches_applied = [False, False]
        self.last_applied_patch_idx = [None, None]
        self.action_patch_indices = self._choose_action_patch_indices(self.current_task)
        self.view_mode = 0
        self.last_test_passed = None
        self.last_tests_passed = 0
        self.last_tests_total = 0
        self.last_pytest_output = ""
        self.focus_func = None
        self.focus_file = None
        self.focus_text = ""
        self._last_failure_sig_for_candidates = ""
        self._materialize_task(self.current_task)
        self._env_descriptor = self._compute_env_descriptor()

    def _maybe_generate_procedural_task(self, task: Optional[RepoTask]) -> None:
        """
        If the selected task is procedural (e.g. name starts with "proc"),
        generate a fresh instance: initial files + candidate edits + tests.
        """
        if task is None:
            return
        if not _is_procedural_task_name(getattr(task, "name", "")):
            return
        cfg = self.config
        category, tags = _procedural_spec(getattr(task, "name", ""))
        tagset = {str(t).lower() for t in tags}
        variant = "ood" if "ood" in tagset else "default"
        toolloop = bool(tagset.intersection({"loop", "toolloop", "open"}))
        n_candidates = int(max(4, getattr(cfg, "procedural_candidates", 8) or 8))
        if n_candidates % 2 == 1:
            n_candidates += 1
        n_tests = int(max(3, getattr(cfg, "procedural_test_cases", 6) or 6))
        max_int = int(max(2, getattr(cfg, "procedural_max_int", 9) or 9))

        seed = int(self.rng.randint(0, 2**31 - 1))
        rng = np.random.RandomState(seed)

        # Select a template family.
        if category == "arith":
            template = str(rng.choice(["add", "div", "power"]))
        elif category == "list":
            template = "mean"
        elif category == "string":
            template = "reverse"
        elif category in ("bundle", "multifile", "multi"):
            template = "bundle"
        elif category in ("refactor", "rename"):
            template = "refactor"
        elif category in ("regression", "regress"):
            template = "clamp"
        elif category == "mixed":
            # keep "mixed" solvable by a single patch by default
            template = str(rng.choice(["add", "div", "power", "mean", "reverse", "clamp"]))
        else:
            # fallback: keep it solvable and fast
            template = str(rng.choice(["add", "div", "power"]))

        if template == "add":
            initial_files, patches = self._proc_task_add(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural arithmetic task: implement add(a,b) correctly."
        elif template == "div":
            initial_files, patches = self._proc_task_div(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural arithmetic task: implement div(a,b) correctly."
        elif template == "power":
            initial_files, patches = self._proc_task_power(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural arithmetic task: implement power(a,b) correctly."
        elif template == "mean":
            initial_files, patches = self._proc_task_mean(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural list task: implement mean(xs) correctly."
        elif template == "reverse":
            initial_files, patches = self._proc_task_reverse(rng, n_tests=n_tests, n_candidates=n_candidates, variant=variant)
            desc = "Procedural string task: implement reverse(s) correctly."
        elif template == "bundle":
            initial_files, patches = self._proc_task_bundle(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural multi-file task: fix add(a,b) and div(a,b)."
        elif template == "refactor":
            initial_files, patches = self._proc_task_refactor(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural refactor task: fix core plus(a,b) and wrapper add(a,b)."
        elif template == "clamp":
            initial_files, patches = self._proc_task_clamp(rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant)
            desc = "Procedural regression task: implement clamp(x, lo, hi) correctly."
        else:
            initial_files, patches = self._proc_task_add(
                rng, n_tests=n_tests, max_int=max_int, n_candidates=n_candidates, variant=variant
            )
            desc = "Procedural arithmetic task: implement add(a,b) correctly."

        if variant == "ood":
            desc = f"{desc} (OOD variant)"
        if toolloop:
            desc = f"{desc} (tool-loop candidates)"
            patches = []

        task.description = desc
        task.initial_files = initial_files
        task.patches = patches

    def _proc_make_expr_task(
        self,
        *,
        module: str,
        module_file: str,
        test_file: str,
        func_name: str,
        args_sig: str,
        bug_expr: str,
        candidate_exprs: List[str],
        call_args: List[str],
        expected_values: List[str],
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        code = f"def {func_name}({args_sig}):\n    return {bug_expr}\n"
        test_lines: List[str] = [f"from {module} import {func_name}", "", "", f"def test_{func_name}_cases():"]
        for ca, exp in zip(call_args, expected_values):
            test_lines.append(f"    assert {func_name}({ca}) == {exp}")
        test_lines.append("")
        tests = "\n".join(test_lines)

        find = f"return {bug_expr}"
        patches: List[RepoPatch] = []
        for i, expr in enumerate(candidate_exprs):
            patches.append(
                RepoPatch(
                    name=f"cand_{i}",
                    description=f"{func_name}: return {expr}",
                    edits=[RepoEdit(path=module_file, find=find, replace=f"return {expr}", count=1)],
                )
            )

        initial_files = {module_file: code, test_file: tests}
        return initial_files, patches

    def _proc_task_reverse(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        bug_pool = [
            "s",
            "\"\".join(sorted(s))",
            "s[::1]",
            "\"\".join(reversed(s))[1:]",
            "s[::-1][1:]",
        ]
        if str(variant) == "ood":
            bug_pool = bug_pool + ["s[1:] + s[:1]", "s[::-1] + s", "s[:-1][::-1]", "''.join(reversed(s))[:-1]"]
        correct = "s[::-1]"
        bug_expr = str(rng.choice(bug_pool))
        if bug_expr == correct:
            bug_expr = "s"

        def _rand_str() -> str:
            alphabet = "abcdxyz"
            n = int(rng.randint(0, 8))
            return "".join(str(rng.choice(list(alphabet))) for _ in range(n))

        call_args: List[str] = []
        expected: List[str] = []
        # Ensure deterministic "anti-trivial" cases exist.
        fixed = ["abc", "", "abca"]
        for s in fixed[: max(0, min(len(fixed), int(n_tests))) ]:
            call_args.append(repr(s))
            expected.append(repr(s[::-1]))
        for _ in range(max(0, int(n_tests) - len(call_args))):
            s = _rand_str()
            call_args.append(repr(s))
            expected.append(repr(s[::-1]))

        pool = bug_pool + [correct, "\"\".join(reversed(s))", "''.join(reversed(s))", "str().join(reversed(s))"]
        cand_list = self._proc_pick_candidates(
            rng,
            required=int(n_candidates),
            must_include=[correct],
            pool=pool + [bug_expr],
        )

        return self._proc_make_expr_task(
            module="text",
            module_file="text.py",
            test_file="test_text.py",
            func_name="reverse",
            args_sig="s",
            bug_expr=bug_expr,
            candidate_exprs=cand_list,
            call_args=call_args,
            expected_values=expected,
        )

    def _proc_pick_candidates(
        self,
        rng: np.random.RandomState,
        *,
        required: int,
        must_include: List[str],
        pool: List[str],
    ) -> List[str]:
        """
        Pick a de-duplicated, shuffled list of candidate expressions.
        Guarantees that each `must_include` expr appears at least once.
        """
        req = int(max(1, required))
        uniq: List[str] = []
        seen = set()
        for expr in list(must_include) + list(pool):
            s = str(expr)
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        if not uniq:
            uniq = [str(x) for x in must_include] or ["0"]

        perm = [uniq[i] for i in rng.permutation(len(uniq))]
        cand = perm[: min(req, len(perm))]
        # Ensure must_include elements are present (replace random slots if needed).
        for expr in must_include:
            s = str(expr)
            if s in cand:
                continue
            if not cand:
                cand.append(s)
                continue
            j = int(rng.randint(0, len(cand)))
            cand[j] = s
        return cand

    def _proc_task_bundle(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        n_total = int(max(4, n_candidates))
        if n_total % 2 == 1:
            n_total += 1
        n_each = int(max(2, n_total // 2))

        # add.py
        if str(variant) == "ood":
            bug_pool_add = ["a + b - 1", "a - b - 1", "a * b", "(a * 2) + b", "a + (b * 2)", "b - a"]
        else:
            bug_pool_add = ["a - b", "a + b + 1", "abs(a - b)", "a * b", "b - a"]
        correct_add = "a + b"
        bug_expr_add = str(rng.choice(bug_pool_add))
        if bug_expr_add == correct_add:
            bug_expr_add = "a - b"

        call_args_add: List[str] = []
        expected_add: List[str] = []
        for _ in range(int(n_tests)):
            a = int(rng.randint(-max_int, max_int + 1))
            b = int(rng.randint(-max_int, max_int + 1))
            call_args_add.append(f"{a}, {b}")
            expected_add.append(str(a + b))

        pool_add = bug_pool_add + [correct_add, "a", "b", "0", "a + (b * 0)"]
        cand_add = self._proc_pick_candidates(
            rng,
            required=n_each,
            must_include=[correct_add],
            pool=pool_add + [bug_expr_add],
        )
        files_add, patches_add = self._proc_make_expr_task(
            module="add",
            module_file="add.py",
            test_file="test_add.py",
            func_name="add",
            args_sig="a, b",
            bug_expr=bug_expr_add,
            candidate_exprs=cand_add,
            call_args=call_args_add,
            expected_values=expected_add,
        )

        # div.py
        if str(variant) == "ood":
            bug_pool_div = ["a // b", "a / (b + 2)", "(a - b) / b", "(a + b) / b", "b / (a if a != 0 else 1)"]
        else:
            bug_pool_div = ["a // b", "b / a", "a / (b + 1)", "a * b"]
        correct_div = "a / b"
        bug_expr_div = str(rng.choice(bug_pool_div))
        if bug_expr_div == correct_div:
            bug_expr_div = "a // b"

        call_args_div: List[str] = []
        expected_div: List[str] = []
        # Force at least one non-integer division so '//' never passes by accident.
        call_args_div.append("3, 2")
        expected_div.append(repr(3 / 2))
        for _ in range(max(0, int(n_tests) - 1)):
            b = int(rng.randint(1, max(2, max_int) + 1))
            a = int(rng.randint(-max_int, max_int + 1))
            if b == 0:
                b = 1
            call_args_div.append(f"{a}, {b}")
            expected_div.append(repr(a / b))

        pool_div = bug_pool_div + [correct_div, "float(a) / float(b)", "a / float(b)"]
        cand_div = self._proc_pick_candidates(
            rng,
            required=n_each,
            must_include=[correct_div],
            pool=pool_div + [bug_expr_div],
        )
        files_div, patches_div = self._proc_make_expr_task(
            module="div",
            module_file="div.py",
            test_file="test_div.py",
            func_name="div",
            args_sig="a, b",
            bug_expr=bug_expr_div,
            candidate_exprs=cand_div,
            call_args=call_args_div,
            expected_values=expected_div,
        )

        initial_files = dict(files_add)
        initial_files.update(files_div)

        # Interleave add/div candidates so the two action slots can naturally cover both files.
        patches: List[RepoPatch] = []
        for i in range(max(len(patches_add), len(patches_div))):
            if i < len(patches_add):
                patches.append(patches_add[i])
            if i < len(patches_div):
                patches.append(patches_div[i])
        return initial_files, patches

    def _proc_task_refactor(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        n_total = int(max(4, n_candidates))
        if n_total % 2 == 1:
            n_total += 1
        n_each = int(max(2, n_total // 2))

        if str(variant) == "ood":
            bug_pool_plus = ["a + b - 1", "a - b - 1", "a * b", "(a * 2) + b", "b - a"]
        else:
            bug_pool_plus = ["a - b", "a + b + 1", "abs(a - b)", "a * b", "b - a"]
        correct_plus = "a + b"
        bug_expr_plus = str(rng.choice(bug_pool_plus))
        if bug_expr_plus == correct_plus:
            bug_expr_plus = "a - b"

        bug_pool_wrapper = ["plus(a, b) + 1", "plus(a, b) - 1", "(plus(a, b) + 2) - 1", "plus(a, b) * 1 + 1"]
        if str(variant) == "ood":
            bug_pool_wrapper = bug_pool_wrapper + ["plus(a, b) + (1 if a >= 0 else 0)"]
        correct_add = "plus(a, b)"
        bug_expr_add = str(rng.choice(bug_pool_wrapper))
        if bug_expr_add == correct_add:
            bug_expr_add = "plus(a, b) + 1"

        call_args: List[str] = []
        expected: List[str] = []
        fixed_pairs = [(2, 3), (-1, 5), (0, 0)]
        for a, b in fixed_pairs[: max(0, min(len(fixed_pairs), int(n_tests))) ]:
            call_args.append(f"{a}, {b}")
            expected.append(str(a + b))
        for _ in range(max(0, int(n_tests) - len(call_args))):
            a = int(rng.randint(-max_int, max_int + 1))
            b = int(rng.randint(-max_int, max_int + 1))
            call_args.append(f"{a}, {b}")
            expected.append(str(a + b))

        code_plus = f"def plus(a, b):\n    return {bug_expr_plus}\n"
        code_add = f"from math_ops import plus\n\n\ndef add(a, b):\n    return {bug_expr_add}\n"

        test_lines_plus: List[str] = ["from math_ops import plus", "", "", "def test_plus_cases():"]
        for ca, exp in zip(call_args, expected):
            test_lines_plus.append(f"    assert plus({ca}) == {exp}")
        test_lines_plus.append("")

        test_lines_add: List[str] = ["from api import add", "", "", "def test_add_cases():"]
        for ca, exp in zip(call_args, expected):
            test_lines_add.append(f"    assert add({ca}) == {exp}")
        test_lines_add.append("")

        initial_files = {
            "math_ops.py": code_plus,
            "api.py": code_add,
            "test_math_ops.py": "\n".join(test_lines_plus),
            "test_api.py": "\n".join(test_lines_add),
        }

        find_plus = f"return {bug_expr_plus}"
        pool_plus = bug_pool_plus + [correct_plus, "a", "b", "0", "a + (b * 0)"]
        cand_plus = self._proc_pick_candidates(
            rng,
            required=n_each,
            must_include=[correct_plus],
            pool=pool_plus + [bug_expr_plus],
        )
        patches_plus = [
            RepoPatch(
                name=f"plus_{i}",
                description=f"plus: return {expr}",
                edits=[RepoEdit(path="math_ops.py", find=find_plus, replace=f"return {expr}", count=1)],
            )
            for i, expr in enumerate(cand_plus)
        ]

        find_add = f"return {bug_expr_add}"
        pool_add = bug_pool_wrapper + [correct_add, "a + b", "(a + b)", "plus(a, b) * 1"]
        cand_add = self._proc_pick_candidates(
            rng,
            required=n_each,
            must_include=[correct_add],
            pool=pool_add + [bug_expr_add],
        )
        patches_add = [
            RepoPatch(
                name=f"add_{i}",
                description=f"add: return {expr}",
                edits=[RepoEdit(path="api.py", find=find_add, replace=f"return {expr}", count=1)],
            )
            for i, expr in enumerate(cand_add)
        ]

        # Interleave core/wrapper candidates to keep them adjacent in the default ordering.
        patches: List[RepoPatch] = []
        for i in range(max(len(patches_plus), len(patches_add))):
            if i < len(patches_plus):
                patches.append(patches_plus[i])
            if i < len(patches_add):
                patches.append(patches_add[i])
        return initial_files, patches

    def _proc_task_clamp(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        bug_pool = ["hi", "lo", "x", "min(hi, x)", "max(lo, x)", "min(hi, max(lo, x + 1))"]
        if str(variant) == "ood":
            bug_pool = bug_pool + ["min(hi, max(lo, x - 1))", "max(hi, min(lo, x))", "max(lo, min(hi, x + 1))"]
        correct = "max(lo, min(hi, x))"
        bug_expr = str(rng.choice(bug_pool))
        if bug_expr == correct:
            bug_expr = "x"

        call_args: List[str] = []
        expected: List[str] = []
        for i in range(int(n_tests)):
            lo = int(rng.randint(-max_int, max_int))
            hi = int(rng.randint(lo + 1, lo + 1 + max(2, max_int)))
            region = int(i % 3)
            if region == 0:
                x = lo - int(rng.randint(1, 4))
            elif region == 1:
                x = int(rng.randint(lo, hi + 1))
            else:
                x = hi + int(rng.randint(1, 4))
            call_args.append(f"{x}, {lo}, {hi}")
            expected.append(str(max(lo, min(hi, x))))

        pool = bug_pool + [correct, "min(hi, max(lo, x))", "max(lo, min(hi, x))"]
        cand_list = self._proc_pick_candidates(
            rng,
            required=int(n_candidates),
            must_include=[correct],
            pool=pool + [bug_expr],
        )

        return self._proc_make_expr_task(
            module="util",
            module_file="util.py",
            test_file="test_util.py",
            func_name="clamp",
            args_sig="x, lo, hi",
            bug_expr=bug_expr,
            candidate_exprs=cand_list,
            call_args=call_args,
            expected_values=expected,
        )

    def _proc_task_add(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        if str(variant) == "ood":
            bug_pool = ["a + b - 1", "a - b - 1", "a * b", "(a * 2) + b", "a + (b * 2)", "b - a"]
        else:
            bug_pool = ["a - b", "a + b + 1", "abs(a - b)", "a * b", "b - a"]
        correct = "a + b"
        bug_expr = str(rng.choice(bug_pool))
        # ensure bug differs from correct
        if bug_expr == correct:
            bug_expr = "a - b"

        # Generate deterministic test cases.
        call_args: List[str] = []
        expected: List[str] = []
        for _ in range(int(n_tests)):
            a = int(rng.randint(-max_int, max_int + 1))
            b = int(rng.randint(-max_int, max_int + 1))
            call_args.append(f"{a}, {b}")
            expected.append(str(a + b))

        pool = bug_pool + [correct, "a", "b", "0", "a + (b * 0)"]
        cand_list = self._proc_pick_candidates(
            rng,
            required=int(n_candidates),
            must_include=[correct],
            pool=pool + [bug_expr],
        )

        return self._proc_make_expr_task(
            module="calc",
            module_file="calc.py",
            test_file="test_calc.py",
            func_name="add",
            args_sig="a, b",
            bug_expr=bug_expr,
            candidate_exprs=cand_list,
            call_args=call_args,
            expected_values=expected,
        )

    def _proc_task_div(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        if str(variant) == "ood":
            bug_pool = ["a // b", "a / (b + 2)", "(a - b) / b", "(a + b) / b", "a / abs(b)"]
        else:
            bug_pool = ["a // b", "a / (b + 1)", "a * b", "a - b"]
        correct = "a / b"
        bug_expr = str(rng.choice(bug_pool))
        if bug_expr == correct:
            bug_expr = "a // b"

        call_args: List[str] = []
        expected: List[str] = []
        for _ in range(int(n_tests)):
            a = int(rng.randint(-max_int, max_int + 1))
            b = int(rng.randint(1, max_int + 1))
            if int(rng.randint(0, 4)) == 0:
                b = -b
            call_args.append(f"{a}, {b}")
            expected.append(repr(a / b))

        pool = bug_pool + [correct, "float(a) / b", "a / float(b)", "a / (b or 1)"]
        cand_list = self._proc_pick_candidates(
            rng,
            required=int(n_candidates),
            must_include=[correct],
            pool=pool + [bug_expr],
        )

        return self._proc_make_expr_task(
            module="calc",
            module_file="calc.py",
            test_file="test_calc.py",
            func_name="div",
            args_sig="a, b",
            bug_expr=bug_expr,
            candidate_exprs=cand_list,
            call_args=call_args,
            expected_values=expected,
        )

    def _proc_task_power(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        if str(variant) == "ood":
            bug_pool = ["a ** (b + 2)", "a ** (b - 1)", "pow(a, b + 1)", "a ^ b", "a * b"]
        else:
            bug_pool = ["a ^ b", "a * b", "a + b", "a ** (b + 1)"]
        correct = "a ** b"
        bug_expr = str(rng.choice(bug_pool))
        if bug_expr == correct:
            bug_expr = "a ^ b"

        call_args: List[str] = []
        expected: List[str] = []
        for _ in range(int(n_tests)):
            a = int(rng.randint(0, max(2, max_int // 2) + 1))
            b = int(rng.randint(0, 6))
            call_args.append(f"{a}, {b}")
            expected.append(str(a**b))

        pool = bug_pool + [correct, "pow(a, b)", "int(pow(a, b))"]
        cand_list = self._proc_pick_candidates(
            rng,
            required=int(n_candidates),
            must_include=[correct],
            pool=pool + [bug_expr],
        )

        return self._proc_make_expr_task(
            module="calc",
            module_file="calc.py",
            test_file="test_calc.py",
            func_name="power",
            args_sig="a, b",
            bug_expr=bug_expr,
            candidate_exprs=cand_list,
            call_args=call_args,
            expected_values=expected,
        )

    def _proc_task_mean(
        self,
        rng: np.random.RandomState,
        *,
        n_tests: int,
        max_int: int,
        n_candidates: int,
        variant: str = "default",
    ) -> Tuple[Dict[str, str], List[RepoPatch]]:
        if str(variant) == "ood":
            bug_pool = [
                "sum(xs)",
                "sum(xs) / (len(xs) + 1)",
                "float(sum(xs) // len(xs))",
                "(sum(xs) + 1) / len(xs)",
                "max(xs)",
            ]
        else:
            bug_pool = ["sum(xs)", "sum(xs) // len(xs)", "(sum(xs) + len(xs)) / len(xs)", "max(xs)"]
        correct = "sum(xs) / len(xs)"
        bug_expr = str(rng.choice(bug_pool))
        if bug_expr == correct:
            bug_expr = "sum(xs)"

        call_args: List[str] = []
        expected: List[str] = []
        for _ in range(int(n_tests)):
            n = int(rng.randint(1, 6))
            xs = [int(rng.randint(-max_int, max_int + 1)) for _ in range(n)]
            call_args.append(repr(xs))
            expected.append(repr(float(sum(xs)) / float(len(xs))))

        pool = bug_pool + [correct, "float(sum(xs)) / len(xs)"]
        cand_list = self._proc_pick_candidates(
            rng,
            required=int(n_candidates),
            must_include=[correct],
            pool=pool + [bug_expr],
        )

        return self._proc_make_expr_task(
            module="stats",
            module_file="stats.py",
            test_file="test_stats.py",
            func_name="mean",
            args_sig="xs",
            bug_expr=bug_expr,
            candidate_exprs=cand_list,
            call_args=call_args,
            expected_values=expected,
        )

    def _run_pytest(self) -> Tuple[bool, int, int, str]:
        if self.workdir is None:
            raise RuntimeError("RepoToolEnv pytest run before reset()")
        if self.workspace_dirty:
            self._clear_bytecode_cache()
        cmd = [sys.executable, "-B", "-m", "pytest", *list(self.config.pytest_args)]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(self.config.timeout_sec),
            )
        except subprocess.TimeoutExpired as exc:
            out = ""
            if getattr(exc, "stdout", None):
                out += str(exc.stdout)
            if getattr(exc, "stderr", None):
                out += "\n" + str(exc.stderr)
            out = (out + "\n[timeout]").strip()
            return False, 0, 0, out

        out = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        passed, failed, errors = _parse_pytest_counts(out)
        total = passed + failed + errors
        ok = completed.returncode == 0
        return ok, passed, total, out.strip()

    def _clear_bytecode_cache(self) -> None:
        if self.workdir is None:
            return
        try:
            for p in self.workdir.rglob("__pycache__"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            for p in self.workdir.rglob("*.pyc"):
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    # ----- Observations -----
    def _compute_env_descriptor(self) -> np.ndarray:
        tests_total = float(max(1, int(self.last_tests_total or 1)))
        max_steps = float(max(1, int(self.max_steps)))
        return build_env_descriptor(
            env_family=self.env_family,
            width=tests_total,
            height=max_steps,
            goal_density=tests_total / max_steps,
            danger_density=0.0,
            wall_density=0.0,
            has_door=False,
            has_key=False,
            has_lava=False,
            max_steps=max_steps,
        )

    def _progress(self) -> float:
        if self.last_tests_total <= 0:
            return 0.0
        return float(self.last_tests_passed) / float(max(1, self.last_tests_total))

    def _encode_patch(self, last_action: int) -> np.ndarray:
        pmax = int(self.config.progress_token_max)
        token_progress = int(round(self._progress() * float(pmax)))
        token_progress = max(0, min(pmax, token_progress))
        patch = np.full((self.view_size, self.view_size), fill_value=token_progress, dtype=np.int64)

        hash_buckets = int(max(0, self.config.hash_buckets))
        hash_base = int(pmax + 32)

        def _hash_token(text: str) -> int:
            if hash_buckets <= 0 or not text:
                return 0
            h = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
            return int(hash_base + (h % hash_buckets))

        # flags
        if self.patches_applied[0]:
            patch[0, 0] = pmax + 1
        if self.patches_applied[1]:
            patch[0, 1] = pmax + 2

        # last test status
        if self.last_test_passed is True:
            patch[0, 2] = pmax + 3
        elif self.last_test_passed is False:
            patch[0, 2] = pmax + 4
        else:
            patch[0, 2] = pmax + 5

        # last action id (bounded)
        patch[0, 3] = pmax + 6 + int(max(0, min(self.n_actions - 1, last_action)))
        # view/page token so the agent can explicitly "inspect" different slices of info.
        if self.view_size >= 5:
            page = 0
            if self.current_task is not None and self.current_task.patches:
                page = int((self.patch_cursor // 2) if self.patch_cursor >= 0 else 0)
            view = int(getattr(self, "view_mode", 0) or 0)
            patch[0, 4] = pmax + 12 + int((view % 4) * 4 + (page % 4))

        # hashed metadata + failure signature (kept discrete to stay compatible with existing Perception)
        if hash_buckets > 0 and self.current_task is not None and self.view_size >= 2:
            task_text = f"{self.current_task.name} {self.current_task.description}".strip()
            patch[1, 0] = _hash_token(task_text)

            p0: Optional[RepoPatch] = None
            p1: Optional[RepoPatch] = None
            if self.current_task.patches:
                idx0 = int(self.action_patch_indices[0]) if self.action_patch_indices else 0
                if 0 <= idx0 < len(self.current_task.patches):
                    p0 = self.current_task.patches[idx0]
                    patch[1, 1] = _hash_token(f"{p0.name} {p0.description}".strip())
                idx1 = int(self.action_patch_indices[1]) if len(self.action_patch_indices) > 1 else 1
                if 0 <= idx1 < len(self.current_task.patches):
                    p1 = self.current_task.patches[idx1]
                    patch[1, 2] = _hash_token(f"{p1.name} {p1.description}".strip())

            patch[1, 3] = _hash_token(_pytest_failure_signature(self.last_pytest_output))
            if self.last_test_passed is True:
                patch[1, 4] = _hash_token("pass")
            elif self.last_test_passed is False:
                patch[1, 4] = _hash_token("fail")

            # Extra tokens from the failure signature to reduce hash collisions.
            n_sig = int(max(0, self.config.failure_sig_tokens))
            if n_sig > 0 and self.view_size >= 3:
                sig = _pytest_failure_signature(self.last_pytest_output)
                words = _tokenize_for_hash(sig)
                for i in range(min(n_sig, self.view_size)):
                    patch[2, i] = _hash_token(words[i].lower()) if i < len(words) else 0

            # Tokenized patch option text (for action 1 vs action 2).
            n_opt = int(max(0, self.config.patch_option_tokens))
            if n_opt > 0 and self.view_size >= 5:
                view = int(getattr(self, "view_mode", 0) or 0)
                max_cells = int(min(self.view_size, n_opt))
                if view == 1:
                    file_list = sorted((self.current_task.initial_files or {}).keys())
                    tokens = _tokenize_for_hash(" ".join(file_list))
                    for i in range(max_cells):
                        patch[3, i] = _hash_token(tokens[i]) if i < len(tokens) else 0
                        patch[4, i] = _hash_token(tokens[i + max_cells]) if (i + max_cells) < len(tokens) else 0
                elif view == 2:
                    tokens = _tokenize_for_hash(self.last_pytest_output)
                    for i in range(max_cells):
                        patch[3, i] = _hash_token(tokens[i]) if i < len(tokens) else 0
                        patch[4, i] = _hash_token(tokens[i + max_cells]) if (i + max_cells) < len(tokens) else 0
                elif view == 3:
                    tokens = _tokenize_for_hash(self.focus_text)
                    for i in range(max_cells):
                        patch[3, i] = _hash_token(tokens[i]) if i < len(tokens) else 0
                        patch[4, i] = _hash_token(tokens[i + max_cells]) if (i + max_cells) < len(tokens) else 0
                else:
                    words0 = _tokenize_for_hash(f"{p0.name} {p0.description}".strip()) if p0 is not None else []
                    words1 = _tokenize_for_hash(f"{p1.name} {p1.description}".strip()) if p1 is not None else []
                    for i in range(max_cells):
                        patch[3, i] = _hash_token(words0[i]) if i < len(words0) else 0
                        patch[4, i] = _hash_token(words1[i]) if i < len(words1) else 0
        return patch

    def _get_obs(self, last_action: int = 0) -> Dict[str, Any]:
        patch = self._encode_patch(last_action)
        energy = float(self._progress())
        return {
            "patch": patch,
            "energy": energy,
            "scenario_id": int(self.current_scenario_id),
            "scenario_name": str(self.current_scenario_name),
            "env_id": int(self.env_id),
            "env_name": str(self.env_name),
            "env_family": str(self.env_family),
            "tests_passed": int(self.last_tests_passed),
            "tests_total": int(self.last_tests_total),
            "steps_taken": int(self.steps_taken),
            "remaining_steps": int(max(0, self.max_steps - self.steps_taken)),
        }

    def get_obs_spec(self) -> Dict[str, Any]:
        return {
            "n_cell_types": int(self.n_cell_types),
            "patch_size": int(self.view_size),
            "n_scenarios": int(self.n_scenarios),
            "fields": [
                "patch",
                "energy",
                "scenario_id",
                "env_id",
                "tests_passed",
                "tests_total",
                "steps_taken",
                "remaining_steps",
            ],
        }

    def get_env_descriptor(self) -> np.ndarray:
        return np.array(self._env_descriptor, copy=True)

    # ----- Core API -----
    def reset(self, seed: Optional[int] = None, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)
        if scenario_id is None:
            self.current_task_idx = int(self.rng.randint(0, len(self.task_set)))
        else:
            self.current_task_idx = int(scenario_id) % len(self.task_set)
        self.current_task = self.task_set[self.current_task_idx]
        self.current_scenario_id = int(self.current_task_idx)
        self.current_scenario_name = str(self.current_task.name)

        self.steps_taken = 0
        self._maybe_generate_procedural_task(self.current_task)
        self.patches_applied = [False, False]
        self.last_applied_patch_idx = [None, None]
        self.action_patch_indices = self._choose_action_patch_indices(self.current_task)
        self.view_mode = 0
        self.last_test_passed = None
        self.last_tests_passed = 0
        self.last_tests_total = 0
        self.last_pytest_output = ""
        self.focus_func = None
        self.focus_file = None
        self.focus_text = ""
        self._last_failure_sig_for_candidates = ""
        self._materialize_task(self.current_task)
        self._env_descriptor = self._compute_env_descriptor()
        return self._get_obs(last_action=0)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= self.n_actions:
            action = 0

        cfg = self.config
        toolloop = self._is_toolloop_task(self.current_task)

        self.steps_taken += 1
        reward = float(cfg.step_penalty)
        done = False
        death_flag = 0.0
        alive = 1.0

        if action == 0:  # NO_OP (used as "inspect": cycle observation view)
            self._cycle_view_mode()
        elif action == 1:  # APPLY_PATCH_0
            reward += float(cfg.apply_patch_penalty)
            if toolloop:
                if self.current_task is None or not (self.current_task.patches or []):
                    reward += float(cfg.toolloop_apply_without_candidates_penalty)
                else:
                    idx0 = int(self.action_patch_indices[0]) if self.action_patch_indices else 0
                    if self.last_applied_patch_idx[0] is not None and idx0 == int(self.last_applied_patch_idx[0]):
                        reward += float(cfg.toolloop_repeat_apply_penalty)
            if self.current_task and self.current_task.patches:
                idx = int(self.action_patch_indices[0]) if self.action_patch_indices else 0
                if 0 <= idx < len(self.current_task.patches):
                    self._apply_patch(self.current_task.patches[idx])
                    self.patches_applied[0] = True
                    self.last_applied_patch_idx[0] = int(idx)
        elif action == 2:  # APPLY_PATCH_1
            reward += float(cfg.apply_patch_penalty)
            if toolloop:
                if self.current_task is None or not (self.current_task.patches or []):
                    reward += float(cfg.toolloop_apply_without_candidates_penalty)
                else:
                    idx1 = int(self.action_patch_indices[1]) if len(self.action_patch_indices) > 1 else 1
                    if self.last_applied_patch_idx[1] is not None and idx1 == int(self.last_applied_patch_idx[1]):
                        reward += float(cfg.toolloop_repeat_apply_penalty)
            if self.current_task and self.current_task.patches:
                idx = int(self.action_patch_indices[1]) if len(self.action_patch_indices) > 1 else 1
                if 0 <= idx < len(self.current_task.patches):
                    self._apply_patch(self.current_task.patches[idx])
                    self.patches_applied[1] = True
                    self.last_applied_patch_idx[1] = int(idx)
        elif action == 3:  # RUN_TESTS
            bootstrap = bool(
                toolloop
                and self.current_task is not None
                and not (self.current_task.patches or [])
                and self.last_test_passed is None
            )
            if bootstrap:
                reward += float(cfg.toolloop_bootstrap_run_tests_penalty)
            elif toolloop:
                reward += float(cfg.toolloop_run_tests_penalty)
            else:
                reward += float(cfg.run_tests_penalty)
            if self.workspace_dirty or self.last_test_passed is None:
                prev_progress = float(self._progress())
                prev_candidates = int(len(self.current_task.patches or [])) if (toolloop and self.current_task is not None) else 0
                ok, passed, total, out = self._run_pytest()
                self.last_test_passed = bool(ok)
                self.last_tests_passed = int(passed)
                self.last_tests_total = int(total)
                self.last_pytest_output = out
                self.workspace_dirty = False
                self._env_descriptor = self._compute_env_descriptor()
                self._update_failure_focus()
                if (not bool(ok)) and self._is_toolloop_task(self.current_task):
                    sig = _pytest_failure_signature(out)
                    if sig != str(self._last_failure_sig_for_candidates or "") or not (self.current_task.patches or []):
                        self._refresh_toolloop_candidates()
                        self._last_failure_sig_for_candidates = sig
                        if self.current_task is not None:
                            new_candidates = int(len(self.current_task.patches or []))
                            if new_candidates > prev_candidates:
                                reward += float(cfg.toolloop_candidate_reward)
                new_progress = float(self._progress())
                reward += float(cfg.progress_reward_scale) * (new_progress - prev_progress)
            if self.last_test_passed:
                reward += float(cfg.success_reward)
                done = True
        elif action == 4:  # REVERT
            reward += float(cfg.revert_penalty)
            self._revert()
        elif action == 5:  # CYCLE_PATCHES
            reward += float(cfg.cycle_patches_penalty)
            self._cycle_patches()

        if self.steps_taken >= int(self.max_steps):
            done = True
            if not bool(self.last_test_passed):
                death_flag = 1.0
                alive = 0.0

        info: Dict[str, Any] = {
            "env_id": int(self.env_id),
            "env_name": str(self.env_name),
            "env_family": str(self.env_family),
            "scenario_id": int(self.current_scenario_id),
            "scenario_name": str(self.current_scenario_name),
            "reward_env": float(reward),
            "tests_passed": int(self.last_tests_passed),
            "tests_total": int(self.last_tests_total),
            "last_test_passed": bool(self.last_test_passed) if self.last_test_passed is not None else None,
            "death_flag": float(death_flag),
            "alive": float(alive),
            "steps_taken": int(self.steps_taken),
            "remaining_steps": int(max(0, self.max_steps - self.steps_taken)),
        }

        if done and self.config.cleanup_on_done:
            keep = bool(self.config.keep_failed_sandboxes and not bool(self.last_test_passed))
            self._cleanup_workdir(keep=keep)

        return self._get_obs(last_action=action), float(reward), bool(done), info
