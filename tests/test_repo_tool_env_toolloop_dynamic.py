from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def _find_patch_idx_for_expr(env: RepoToolEnv, expr: str) -> int:
    assert env.current_task is not None
    for idx, patch in enumerate(env.current_task.patches):
        if expr and expr in (patch.description or ""):
            return idx
        for content in (patch.files or {}).values():
            if f"return {expr}" in (content or ""):
                return idx
    raise AssertionError(f"Could not find a candidate patch containing expr={expr!r}")


def _apply_patch_idx(env: RepoToolEnv, patch_idx: int, max_cycles: int = 20) -> None:
    for _ in range(max_cycles):
        if env.action_patch_indices[0] == patch_idx:
            env.step(1)
            return
        if env.action_patch_indices[1] == patch_idx:
            env.step(2)
            return
        env.step(5)  # CYCLE_PATCHES
    raise AssertionError("Could not bind the desired patch index into APPLY_PATCH_0/1 slots")


def test_repo_tool_env_toolloop_bundle_dynamic_is_solvable(tmp_path):
    tasks = build_repo_taskset(["proc_bundle_loop"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=160,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
        procedural_candidates=8,
        procedural_test_cases=6,
        procedural_max_int=9,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_toolloop", seed=0)

    env.reset(scenario_id=0)
    assert env.current_task is not None
    assert env.current_task.initial_files
    assert env.last_test_passed is False
    assert env.current_task.patches, "tool-loop scenarios should bootstrap candidates at reset"

    # The correct fix should be present somewhere in the candidate set (not necessarily in the first pair).
    focus0 = getattr(env, "focus_func", None)
    desired0 = "a + b" if focus0 == "add" else "a / b"
    _find_patch_idx_for_expr(env, desired0)

    # Solve: fix whichever function is currently failing, then rerun tests until all pass.
    for _ in range(6):
        if env.last_test_passed is True:
            break
        focus = getattr(env, "focus_func", None)
        assert focus in ("add", "div")
        desired_expr = "a + b" if focus == "add" else "a / b"
        idx = _find_patch_idx_for_expr(env, desired_expr)
        _apply_patch_idx(env, idx)
        _, _, done, info = env.step(3)  # RUN_TESTS
        if done and info.get("last_test_passed") is True:
            break

    assert env.last_test_passed is True
