# Execution Checklist (Roadmap v2, Synced 2026-02-15)

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
- [ ] Add long-horizon benchmark suite (100+ step planning and hierarchical subgoals)
- [ ] Add adversarial safety/OOD pack (constraint compliance + catastrophic fail metrics)
- [ ] Add stronger social/ToM transfer metrics
- [ ] Expand multimodal coverage (beyond current text/grid/tool stack)

## E) Reporting Discipline
- [x] Keep `ROADMAP.md`, `ROADMAP_v2.md`, `CHECKLIST.md` in sync with the latest accepted report
- [ ] For every new milestone run, append short gate delta + artifact list in commit message or changelog
