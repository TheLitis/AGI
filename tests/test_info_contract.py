from __future__ import annotations

from typing import Any, Dict

from computer_env import ComputerEnv, ComputerEnvConfig, build_computer_taskset
from env import GridWorldEnv
from instruction_env import InstructionEnv, InstructionEnvConfig
from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset
from social_env import SocialEnv, SocialEnvConfig
from tool_env import ToolEnv, ToolEnvConfig


REQUIRED_INFO_KEYS = (
    "terminated_reason",
    "success",
    "constraint_violation",
    "catastrophic",
    "timeout",
    "reward_env",
    "events",
)


def _assert_info_contract(info: Dict[str, Any]) -> None:
    for key in REQUIRED_INFO_KEYS:
        assert key in info
    assert isinstance(info["terminated_reason"], str)
    assert info["success"] is None or isinstance(info["success"], bool)
    assert isinstance(info["constraint_violation"], bool)
    assert isinstance(info["catastrophic"], bool)
    assert isinstance(info["timeout"], bool)
    assert isinstance(info["reward_env"], float)
    assert isinstance(info["events"], dict)


def test_gridworld_info_contract_fields_present():
    env = GridWorldEnv(max_steps=4, max_energy=4, seed=0, multi_task=False)
    env.reset()
    _obs, _reward, _done, info = env.step(4)  # STAY
    _assert_info_contract(info)


def test_instruction_info_contract_fields_present():
    env = InstructionEnv(config=InstructionEnvConfig(max_steps=6), seed=1)
    env.reset()
    _obs, _reward, _done, info = env.step(4)  # STAY
    _assert_info_contract(info)


def test_social_info_contract_fields_present():
    env = SocialEnv(config=SocialEnvConfig(max_steps=6), seed=2)
    env.reset()
    _obs, _reward, _done, info = env.step(4)  # STAY
    _assert_info_contract(info)


def test_computer_info_contract_fields_present():
    taskset = build_computer_taskset(["simple_project"])
    env = ComputerEnv(task_set=taskset, config=ComputerEnvConfig(rng_seed=3), seed=3)
    env.reset()
    _obs, _reward, _done, info = env.step(0)  # NO_OP
    _assert_info_contract(info)


def test_tool_info_contract_fields_present():
    env = ToolEnv(config=ToolEnvConfig(max_steps=4), seed=4)
    env.reset()
    _obs, _reward, _done, info = env.step(0)  # NOOP
    _assert_info_contract(info)


def test_repo_info_contract_fields_present(tmp_path):
    taskset = build_repo_taskset(["calc_add"])
    cfg = RepoToolEnvConfig(sandbox_root=str(tmp_path), max_steps=4, timeout_sec=5.0)
    env = RepoToolEnv(task_set=taskset, config=cfg, seed=5)
    env.reset()
    _obs, _reward, _done, info = env.step(0)  # NO_OP/inspect
    _assert_info_contract(info)

