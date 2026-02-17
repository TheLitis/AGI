# Execution Checklist (Roadmap v2, Synced 2026-02-17)

## P0) Active Sprint Priority
- [x] Add dedicated `long_horizon` suite and wire it into AGI bench selection order
- [x] Add per-case `max_steps_env` plumbing (`bench.py` -> `run_experiment`)
- [x] Replace safety placeholders with runtime eval metrics (`constraint_compliance`, `catastrophic_fail_rate`, `death_rate`, `reason_counts`)
- [x] Reach seed0 smoke targets for priority suites (`reports/bench_priority_quick_seed0.autonomy2.json`)
- [x] Run 2-seed quick sanity for priority suites (`reports/bench_priority_quick_seed01.autonomy1.json`)
- [x] Tune long-horizon quality on multi-seed quick runs (target: stable `long_horizon.score >= 0.65`) (`reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json`, `score = 0.6847`)
- [x] Run 5-seed lifelong stability snapshot (`reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json`, `forgetting_gap = 0.7799`, `forward_transfer = 0.6950`)
- [x] Run 5-seed safety stability snapshot (`reports/bench_safety_quick_seed01234.autonomy4.json`)
- [x] Improve lifelong metrics after long-horizon tuning (`forward_transfer = 0.6950`, `forgetting_gap = 0.7799`)
- [x] Verify Mountain #2/#3 opener check (`scripts/check_mountains_open.py` returns `[OPEN]`)
- [x] Rebaseline Gate0-3 on canonical 5-seed AGI quick report (`reports/agi_v1.quick.seed01234.rebaseline_phase2.retune1.json`)
- [x] Add Windows run-id filename safety guardrail + tests (`tests/test_experiment_logger.py`)
- [x] Reproduce independent rerun #2 (`reports/agi_v1.quick.seed01234.rebaseline_phase2.rerun2.json`)
- [ ] Reduce safety catastrophic failures and lift compliance on bench seeds

## A) Verified Current State
- [x] Verify gate/schema unit tests: `python -m pytest tests/test_bench_gates.py tests/test_agi_bench_report.py tests/test_bench_scoring.py -q`
- [x] Verify full test suite health: `python -m pytest -q`
- [x] Produce canonical 5-seed AGI quick reference report: `reports/agi_v1.quick.seed01234.rebaseline_phase2.retune1.json`
- [x] Validate reference report structure + gates: `python validate_bench_report.py --report reports/agi_v1.quick.seed01234.rebaseline_phase2.retune1.json --expect-gate gate0=pass gate1=pass gate2=pass gate3=pass`
- [x] Confirm canonical reference has `gate0..gate4 = pass`
- [x] Confirm canonical `core/language` are `status=ok` and have no runtime `OSError: [Errno 22] Invalid argument`
- [x] Confirm capability vector thresholds are met (`generalization/sample_efficiency/robustness/tool_workflow`)
- [x] Produce latest long-horizon 5-seed report: `reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json` (`score = 0.6847`)
- [x] Produce latest lifelong 5-seed report: `reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json` (`forgetting_gap = 0.7799`, `forward_transfer = 0.6950`)
- [x] Confirm Mountain #2/#3 open on internal Gate2-Strict criteria (`long_horizon.score >= 0.65`, `lifelong.forgetting_gap >= -1.0`, `lifelong.forward_transfer >= 0.5`)
- [x] Regenerate validated multi-seed AGI quick reference (`seeds 0..4`) after core/language runtime fix
- [x] Reproduce independent rerun #2 with same config: `reports/agi_v1.quick.seed01234.rebaseline_phase2.rerun2.json` (`gate0..gate4=pass`)

## B) Gate4 Close Tasks (Operational)
- [x] Reproduce independent rerun #2: `python bench.py --suite agi_v1 --quick --seeds 0,1,2,3,4 --report reports/agi_v1.quick.seed01234.rebaseline_phase2.rerun2.json`
- [x] Validate rerun #2 structure and gates: `python validate_bench_report.py --report reports/agi_v1.quick.seed01234.rebaseline_phase2.rerun2.json --expect-gate gate0=pass gate1=pass gate2=pass gate3=pass`
- [x] Raise `overall.confidence` to `>= 0.80` on repeated runs (`0.8836` on canonical report, `0.8836` on rerun #2)
- [x] Keep Gate3 CI thresholds below limits after confidence-tuning changes

## C) Acceptance Runs Beyond Quick
- [ ] Full 5-seed run: `python bench.py --suite agi_v1 --seeds 0,1,2,3,4 --report reports/agi_v1.full.seed01234.accept.json`
- [ ] Full 5-seed OOD run: `python bench.py --suite agi_v1 --ood --seeds 0,1,2,3,4 --report reports/agi_v1.full.ood.seed01234.accept.json`
- [ ] Confirm Gate status consistency between quick/full/OOD runs

## D) 8-Mountain Depth Backlog
- [x] Add long-horizon benchmark suite (100+ step planning and hierarchical subgoals)
- [x] Add runtime safety metrics to evaluation + bench safety suite consumption
- [ ] Add adversarial safety/OOD pack (constraint compliance + catastrophic fail metrics)
- [ ] Add stronger social/ToM transfer metrics
- [ ] Expand multimodal coverage (beyond current text/grid/tool stack)

## E) Reporting Discipline
- [x] Keep `ROADMAP.md`, `ROADMAP_v2.md`, `CHECKLIST.md` in sync with the latest accepted report
- [ ] For every new milestone run, append short gate delta + artifact list in commit message or changelog

