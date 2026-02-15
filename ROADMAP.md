# ROADMAP (Synced)

Updated: 2026-02-15

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed01234.stab2.json`.
- Current gates: `gate0=pass`, `gate1=pass`, `gate2=pass`, `gate3=pass`, `gate4=fail`.
- Immediate blocker to Gate4:
  - `overall.confidence = 0.7846` (threshold: `>= 0.80`).

## Active Priority Order
1. Long-horizon planning quality (Mountain #2).
2. Lifelong adaptation/forgetting stability (Mountain #3).
3. Safety/alignment hardening (Mountain #7).
4. Then expansion of remaining mountains.

## Confirmed Metrics Snapshot
- capabilities:
  - `generalization_score = 0.8314`
  - `sample_efficiency_score = 0.8548`
  - `robustness_score = 0.8864`
  - `tool_workflow_score = 0.95`
- key suite CI half-width:
  - `core = 1.3202` (<= 1.50)
  - `tools = 0.0952` (<= 0.10)
  - `language = 0.0135` (<= 0.10)
  - `social = 0.0995` (<= 0.10)
  - `lifelong = 0.2011` (<= 0.75)

## What Is Already Implemented
- Report schema `0.2` in `bench.py` with:
  - `meta.run_manifest`
  - `suites[].ci`
  - `overall.capabilities`
  - `overall.confidence`
  - explicit `gate3`, `gate4`
- Lifelong variance stabilization:
  - stratified scenario scheduling
  - policy-controlled lifelong eval
- Long-horizon benchmark wiring:
  - dedicated `long_horizon` suite
  - per-case `max_steps_env` control in `run_experiment`
  - horizon/planner metrics and scoring in `bench.py`
- Safety metric wiring:
  - `trainer.evaluate()` now exports `constraint_compliance`, `catastrophic_fail_rate`, `death_rate`, `reason_counts`
  - `safety` suite now uses real eval metrics + planner smoke check
- Quick language/lifelong stabilization for Gate3 reliability.
- Lifelong robustness instrumentation refresh:
  - `lifelong` CI now tracks per-run lifelong score (not raw transfer delta)
  - quick lifelong eval uses sampled policy for less brittle seed behavior
  - `run_lifelong_eval` now records uncertainty consistently with train path

## Immediate Next Milestones
1. Lift lifelong `forward_transfer` while keeping `forgetting_gap` near zero after long-horizon tuning.
2. Reduce safety `catastrophic_fail_rate` and raise `constraint_compliance` from current weak baseline.
3. After 1-2 stabilize, re-run full Gate4 confidence push (`overall.confidence >= 0.80`).

## Priority Smoke (2026-02-15)
- Reference artifact: `reports/bench_priority_quick_seed0.autonomy2.json`
- `long_horizon.score = 0.7063`
- `lifelong.score = 0.6612`
- `safety.score = 0.8178`
- 2-seed check artifact: `reports/bench_priority_quick_seed01.autonomy1.json`
  - `long_horizon.score = 0.7264`
  - `lifelong.score = 0.5718`
  - `safety.score = 0.7703`
- Remaining gap:
  - convert seed0 gains into stable multi-seed (`0..4`) behavior before claiming milestone closure.
- 5-seed quick snapshots (priority suites, isolated runs):
  - `reports/bench_long_horizon_quick_seed01234.autonomy4.json`
    - `long_horizon.score = 0.7609`
    - `long_horizon.ci.half_width = 0.0258`
  - `reports/bench_lifelong_quick_seed01234.autonomy4.json`
    - `lifelong.score = 0.5092`
    - `lifelong.forgetting_gap = 0.9794`
    - `lifelong.forward_transfer = 0.1842`
    - `lifelong.ci.half_width = 0.0727`
  - `reports/bench_safety_quick_seed01234.autonomy4.json`
    - `safety.score = 0.7444`
    - `constraint_compliance = 0.55`
    - `catastrophic_fail_rate = 0.25`

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced.
