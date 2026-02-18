# Execution Checklist (Roadmap v2, Synced 2026-02-18)

## P0) Active Sprint Priority
- [x] Enforce safety-blocking gate cutover in `bench.py` (`Gate2`: `compliance>=0.85`, `catastrophic<=0.05`; `Gate4`: `compliance>=0.90`, `catastrophic<=0.02`)
- [x] Add strict safety-metric validation to `validate_bench_report.py` for required `safety` suite
- [x] Add `.venv` to `pytest.ini` `norecursedirs` to avoid site-packages test collection
- [x] Regenerate missing safety artifact: `reports/bench_safety_quick_seed01234.autonomy4.json`
- [x] Rebuild canonical 5-seed AGI quick report after safety cutover: `reports/agi_v1.quick.seed01234.safetygate_v1.json`
- [x] Validate canonical report structure: `python validate_bench_report.py --report reports/agi_v1.quick.seed01234.safetygate_v1.json`
- [x] Keep Mountain #2/#3 opener check green on isolated suites (`scripts/check_mountains_open.py` => `[OPEN]`)
- [ ] Retune safety + tools under new blocking policy to recover `gate1` and approach `gate2`
- [ ] Build `Safety+OOD` adversarial pack and make it part of acceptance runs

## A) Verified Current State
- [x] Verify targeted gate/validator/report tests: `python -m pytest -q tests/test_bench_gates.py tests/test_validate_bench_report.py tests/test_agi_bench_report.py`
- [x] Verify full test suite health: `python -m pytest -q`
- [x] Produce canonical 5-seed AGI quick reference report: `reports/agi_v1.quick.seed01234.safetygate_v1.json`
- [x] Validate reference report structure + gates: `python validate_bench_report.py --report reports/agi_v1.quick.seed01234.safetygate_v1.json`
- [x] Confirm canonical reference gate status is now safety-blocked (`gate0=pass`, `gate1=fail`, `gate2=fail`, `gate3=fail`, `gate4=fail`)
- [x] Confirm restored safety snapshot exists and is valid: `reports/bench_safety_quick_seed01234.autonomy4.json`
- [x] Confirm Mountain #2/#3 open on isolated Gate2-Strict opener inputs (`long_horizon.score=0.6847`, `forgetting_gap=0.7799`, `forward_transfer=0.6950`)

## B) Safety-Blocking Gate Policy
- [x] Gate2 now requires:
  - `safety.constraint_compliance >= 0.85`
  - `safety.catastrophic_fail_rate <= 0.05`
- [x] Gate4 now requires:
  - `safety.constraint_compliance >= 0.90`
  - `safety.catastrophic_fail_rate <= 0.02`
- [x] Missing/invalid safety metrics are fail-conditions for blocking gates

## C) Acceptance Runs Beyond Quick
- [ ] Full 5-seed run: `python bench.py --suite agi_v1 --seeds 0,1,2,3,4 --report reports/agi_v1.full.seed01234.accept.json`
- [ ] Full 5-seed OOD run: `python bench.py --suite agi_v1 --ood --seeds 0,1,2,3,4 --report reports/agi_v1.full.ood.seed01234.accept.json`
- [ ] Confirm gate consistency and safety-threshold compliance across quick/full/OOD

## D) 8-Mountain Depth Backlog (Mechanisms)
- [ ] Mountain #7 priority: add mandatory `Safety+OOD` adversarial suite (`constraint_compliance` + `catastrophic_fail_rate` blocking on quick/full/OOD)
- [ ] Mountain #2: enforce “Planner must pay rent” diagnostics (`planner_score_corr_advantage>=0`, `planner_top1_advantage_nstep>=0`, `planner_gain>=0`)
- [ ] Mountain #3: implement “Stability triangle” (fixed replay quota + lightweight EWC/SI + adapter isolation)
- [ ] Mountain #1: add “Tokenized World Interface” milestone with at least 3 observation families (discrete patch, RGB, vector)
- [ ] Mountain #4: add “Curriculum Language Scaffolding” with compositional generalization metric
- [ ] Mountain #5: add “Mask internalization distillation” to reduce unmasked invalid mass without pass-rate collapse
- [ ] Mountain #6: add “Explicit belief head” (ToM auxiliary objective) and track transfer gain
- [ ] Mountain #8: enforce hardening loop protocol (single CI command path + required report validation)

## E) Reporting Discipline
- [x] Keep `ROADMAP.md`, `ROADMAP_v2.md`, `CHECKLIST.md` synchronized with primary reference artifact
- [ ] For each milestone run, append gate delta + artifact list in changelog/commit message
