from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_procedural_refactor_task_is_solvable(tmp_path):
    tasks = build_repo_taskset(["proc_refactor"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=240,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
        procedural_candidates=8,
        procedural_test_cases=6,
        procedural_max_int=9,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_proc_refactor", seed=0)

    env.reset(scenario_id=0)
    assert env.current_task is not None
    assert env.current_task.initial_files
    assert env.current_task.patches

    # Initial run should fail (two buggy implementations).
    _, _, done0, info0 = env.step(3)  # RUN_TESTS
    assert done0 is False
    assert info0["last_test_passed"] is False
    assert info0["tests_total"] >= 2

    patches = list(env.current_task.patches)

    def _targets(p):
        targets = set()
        for rel in (p.files or {}).keys():
            targets.add(str(rel))
        for e in p.edits or []:
            targets.add(str(e.path))
        return targets

    core_idxs = [i for i, p in enumerate(patches) if "math_ops.py" in _targets(p)]
    wrapper_idxs = [i for i, p in enumerate(patches) if "api.py" in _targets(p)]
    assert core_idxs, "expected math_ops.py candidates"
    assert wrapper_idxs, "expected api.py candidates"

    def _apply_patch_index(target_idx: int) -> None:
        n_pairs = max(1, (len(patches) + 1) // 2)
        for _ in range(n_pairs + 2):
            idx0, idx1 = env.action_patch_indices
            if idx0 == target_idx:
                env.step(1)  # APPLY_PATCH_0
                return
            if idx1 == target_idx:
                env.step(2)  # APPLY_PATCH_1
                return
            env.step(5)  # CYCLE_PATCHES
        raise AssertionError(f"could not navigate to patch index {target_idx}")

    # Find a core fix that improves progress (plus passes, add still fails).
    core_fix = None
    for idx in core_idxs:
        env.step(4)  # REVERT
        _apply_patch_index(idx)
        _, _, done, info = env.step(3)  # RUN_TESTS
        if info.get("last_test_passed") is True:
            # If this ever happens, the wrapper happened to already be correct.
            assert done is True
            return
        if info.get("tests_total") >= 2 and info.get("tests_passed") == 1:
            core_fix = idx
            break
    assert core_fix is not None, "did not find a core patch that makes partial progress"

    # With core fixed, find the wrapper fix.
    for idx in wrapper_idxs:
        env.step(4)  # REVERT
        _apply_patch_index(core_fix)
        _apply_patch_index(idx)
        _, _, done, info = env.step(3)  # RUN_TESTS
        if info.get("last_test_passed") is True:
            assert done is True
            assert info["tests_total"] >= 2
            assert info["tests_passed"] == info["tests_total"]
            return

    raise AssertionError("procedural refactor task did not become solvable within the candidate search budget")

