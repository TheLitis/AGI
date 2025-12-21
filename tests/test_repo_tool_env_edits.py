from repo_tool_env import RepoEdit, RepoPatch, RepoTask, RepoToolEnv, RepoToolEnvConfig


def test_repo_tool_env_supports_micro_edits(tmp_path):
    task = RepoTask(
        task_id=0,
        name="calc_add_edit",
        description="Fix add via a micro-edit patch.",
        initial_files={
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        patches=[
            RepoPatch(
                name="edit_minus_to_plus",
                description="Replace '-' with '+' in return expression.",
                files={},
                edits=[RepoEdit(path="calc.py", find="return a - b", replace="return a + b", count=1)],
            )
        ],
    )

    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=10,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
    )
    env = RepoToolEnv(task_set=[task], config=cfg, env_id=0, env_name="repo_edit", seed=0)

    env.reset(scenario_id=0)
    _, _, done0, info0 = env.step(3)  # RUN_TESTS (fail)
    assert done0 is False
    assert info0["last_test_passed"] is False

    env.step(1)  # APPLY_PATCH_0 (micro-edit)
    _, reward1, done1, info1 = env.step(3)  # RUN_TESTS (pass)
    assert done1 is True
    assert info1["last_test_passed"] is True
    assert reward1 > 0.0

