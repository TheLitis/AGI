# Execution Checklist (Roadmap v2)

## 0) Environment
- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Verify bench tests: `python -m pytest tests/test_bench_gates.py tests/test_agi_bench_report.py tests/test_bench_scoring.py -q`

## 1) Smoke (fast)
- [ ] Full quick smoke: `python bench.py --suite quick --seeds 0 --report reports/bench_quick_smoke_seed0.json`
- [ ] AGI quick smoke: `python bench.py --suite agi_v1 --quick --seeds 0 --report reports/agi_v1.quick.smoke.seed0.json`

## 2) Gate2 Candidate Runs
- [ ] Run 1: `python bench.py --suite agi_v1 --quick --seeds 0,1,2 --report reports/agi_v1.quick.gate2.seed012.run1.json`
- [ ] Run 2: `python bench.py --suite agi_v1 --quick --seeds 0,1,2 --report reports/agi_v1.quick.gate2.seed012.run2.json`
- [ ] Confirm both reports show `overall.gates.gate2 == pass`

## 3) Gate3 Candidate Runs (robustness)
- [ ] 5-seed full: `python bench.py --suite agi_v1 --seeds 0,1,2,3,4 --report reports/agi_v1.full.gate3.seed01234.json`
- [ ] 5-seed OOD: `python bench.py --suite agi_v1 --ood --seeds 0,1,2,3,4 --report reports/agi_v1.full.ood.seed01234.json`
- [ ] Confirm `overall.gates.gate3 == pass` and CI half-width constraints per suite

## 4) Regression Discipline
- [ ] Run targeted suites before/after risky changes:
  - core: `python bench.py --suite core --quick --seeds 0,1,2 --report reports/bench_core_quick_seed012.regression.json`
  - tools: `python bench.py --suite tools --quick --seeds 0,1,2 --report reports/bench_tools_quick_seed012.regression.json`
  - language: `python bench.py --suite language --quick --seeds 0,1,2 --report reports/bench_language_quick_seed012.regression.json`
- [ ] Keep previous baseline report for diff-based review

## 5) OOD Pack
- [ ] core OOD: `python bench.py --suite core --ood --seeds 0,1,2 --report reports/bench_core_ood_seed012.json`
- [ ] tools OOD: `python bench.py --suite tools --ood --seeds 0,1,2 --report reports/bench_tools_ood_seed012.json`
- [ ] language OOD: `python bench.py --suite language --ood --seeds 0,1,2 --report reports/bench_language_ood_seed012.json`
- [ ] social OOD: `python bench.py --suite social --ood --seeds 0,1,2 --report reports/bench_social_ood_seed012.json`

## 6) Acceptance Snapshot
- [ ] Gate2 stable pass (two repeated runs)
- [ ] Gate3 stable pass (5 seeds + OOD)
- [ ] Capability vector populated in report (`overall.capabilities`)
- [ ] Confidence available and >= threshold for Gate4 candidate
