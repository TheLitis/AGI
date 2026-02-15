# Execution Checklist (Roadmap v2, Synced 2026-02-15)

## P0) Active Sprint Priority
- [x] Add dedicated `long_horizon` suite and wire it into AGI bench selection order
- [x] Add per-case `max_steps_env` plumbing (`bench.py` -> `run_experiment`)
- [x] Replace safety placeholders with runtime eval metrics (`constraint_compliance`, `catastrophic_fail_rate`, `death_rate`, `reason_counts`)
- [x] Reach seed0 smoke targets for priority suites (`reports/bench_priority_quick_seed0.autonomy2.json`)
- [x] Run 2-seed quick sanity for priority suites (`reports/bench_priority_quick_seed01.autonomy1.json`)
- [x] Tune long-horizon quality on multi-seed quick runs (target: stable `long_horizon.score >= 0.65`) (`reports/bench_long_horizon_quick_seed01234.autonomy4.json`)
- [x] Run 5-seed lifelong stability snapshot (`reports/bench_lifelong_quick_seed01234.autonomy4.json`)
- [x] Run 5-seed safety stability snapshot (`reports/bench_safety_quick_seed01234.autonomy4.json`)
- [ ] Improve lifelong metrics after long-horizon tuning (`forward_transfer` up, `forgetting_gap` near zero)
- [ ] Reduce safety catastrophic failures and lift compliance on bench seeds

## A) Verified Current State
- [x] Verify gate/schema unit tests: `python -m pytest tests/test_bench_gates.py tests/test_agi_bench_report.py tests/test_bench_scoring.py -q`
- [x] Verify full test suite health: `python -m pytest -q`
- [x] Produce 5-seed AGI quick reference report: `reports/agi_v1.quick.seed01234.stab2.json`
- [x] Confirm `gate2 == pass` on reference report
- [x] Confirm `gate3 == pass` on reference report
- [x] Confirm `gate4 == fail` and blocker is `overall.confidence < 0.80`
- [x] Confirm capability vector thresholds are met (`generalization/sample_efficiency/robustness/tool_workflow`)

## B) Gate4 Close Tasks (Operational)
- [ ] Reproduce independent rerun #2: `python bench.py --suite agi_v1 --quick --seeds 0,1,2,3,4 --report reports/agi_v1.quick.seed01234.rerun2.json`
- [ ] Validate rerun #2 structure and gates: `python validate_bench_report.py --report reports/agi_v1.quick.seed01234.rerun2.json`
- [ ] Raise `overall.confidence` to `>= 0.80` on repeated runs
- [ ] Keep Gate3 CI thresholds below limits after confidence-tuning changes

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
