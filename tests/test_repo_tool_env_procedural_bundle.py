from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset


def test_repo_tool_env_procedural_bundle_task_is_solvable(tmp_path):
    tasks = build_repo_taskset(["proc_bundle"])
    cfg = RepoToolEnvConfig(
        sandbox_root=str(tmp_path / "repo_sandboxes"),
        max_steps=300,
        timeout_sec=10.0,
        shuffle_patch_bindings=False,
        cleanup_on_done=True,
        cleanup_on_reset=True,
        keep_failed_sandboxes=False,
        procedural_candidates=8,
        procedural_test_cases=6,
        procedural_max_int=9,
    )
    env = RepoToolEnv(task_set=tasks, config=cfg, env_id=0, env_name="repo_proc_bundle", seed=0)

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

    add_idxs = [i for i, p in enumerate(patches) if "add.py" in _targets(p)]
    div_idxs = [i for i, p in enumerate(patches) if "div.py" in _targets(p)]
    assert add_idxs, "expected add.py candidates"
    assert div_idxs, "expected div.py candidates"

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

    def _find_single_fix(candidate_indices):
        for idx in candidate_indices:
            env.step(4)  # REVERT
            _apply_patch_index(idx)
            _, _, _, info = env.step(3)  # RUN_TESTS
            if info.get("tests_total") >= 2 and info.get("tests_passed") == 1:
                return idx
        raise AssertionError("did not find a single-file fix patch")

    add_fix = _find_single_fix(add_idxs)
    div_fix = _find_single_fix(div_idxs)

    # Apply both fixes and confirm success.
    env.step(4)  # REVERT
    _apply_patch_index(add_fix)
    _apply_patch_index(div_fix)
    _, _, done, info = env.step(3)  # RUN_TESTS
    assert done is True
    assert info["last_test_passed"] is True
    assert info["tests_total"] >= 2
    assert info["tests_passed"] == info["tests_total"]

