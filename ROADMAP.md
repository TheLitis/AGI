# ROADMAP (Synced)

Updated: 2026-02-17

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed01234.rebaseline_phase2.retune1.json`.
- Current gates: `gate0=pass`, `gate1=pass`, `gate2=pass`, `gate3=pass`, `gate4=pass`.
- Independent reproducibility rerun: `reports/agi_v1.quick.seed01234.rebaseline_phase2.rerun2.json` (`gate0..gate4=pass`).
- Internal mountain status (Gate2-Strict, isolated 5-seed):
  - Mountain #2 (`long_horizon`) = `open` (`score = 0.6847`, threshold `>= 0.65`) via `reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json`.
  - Mountain #3 (`lifelong`) = `open` (`forgetting_gap = 0.7799`, `forward_transfer = 0.6950`) via `reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json`.
- Canonical baseline health:
  - `core` suite = `ok`, `language` suite = `ok` (no runtime `OSError: [Errno 22] Invalid argument`).
  - `overall.confidence = 0.8836` (above Gate4 threshold `>= 0.80`).

## Active Priority Order
1. Keep Gate0-3 stable on repeated 5-seed runs and full/OOD acceptance packs.
2. Safety/alignment hardening (Mountain #7), especially catastrophic-failure reduction.
3. Preserve Mountain #2/#3 quality while extending Mountain #1/#4/#6 depth.
4. Maintain infra reliability guardrails (including Windows-safe run logging).

## Confirmed Metrics Snapshot
- overall:
  - `agi_score = 0.7980`
  - `overall.confidence = 0.8836`
- capabilities:
  - `generalization_score = 0.8484`
  - `sample_efficiency_score = 0.8548`
  - `robustness_score = 0.9327`
  - `tool_workflow_score = 0.9583`
- required suite status on canonical validated baseline:
  - `long_horizon = ok`, `score = 0.8135`, `ci.half_width = 0.0144`
  - `tools = ok`, `score = 0.9167`, `ci.half_width = 0.0517`
  - `social = ok`, `score = 0.8500`, `ci.half_width = 0.0995`
  - `lifelong = ok`, `score = 0.5852`, `ci.half_width = 0.0648`
  - `safety = ok`, `score = 0.7830`, `ci.half_width = 0.0623`
  - `core = ok`, `score = 1.0000`, `ci.half_width = 0.0202`
  - `language = ok`, `score = 0.7095`, `ci.half_width = 0.0135`

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
1. Run full 5-seed AGI acceptance (`quick/full/OOD`) and keep gate consistency.
2. Reduce catastrophic safety failures and improve constraint compliance under harder distributions.
3. Expand 8-mountain depth (multimodality, stronger ToM/social transfer, language abstraction quality) without regressing Gate0-3.

## Historical Priority Smoke (2026-02-15)
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
- note:
  - these priority snapshots are directional diagnostics, not the canonical validated AGI reference.
- 5-seed quick snapshots (priority suites, isolated runs, latest):
  - `reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json`
    - `long_horizon.score = 0.6847`
    - `long_horizon.goal_completion_rate = 0.7875`
    - `long_horizon.timeout_rate = 0.4375`
  - `reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json`
    - `lifelong.score = 0.5347`
    - `lifelong.forgetting_gap = 0.7799`
    - `lifelong.forward_transfer = 0.6950`
  - `reports/bench_safety_quick_seed01234.autonomy4.json`
    - `safety.score = 0.7444`
    - `constraint_compliance = 0.55`
    - `catastrophic_fail_rate = 0.25`
- mountain opener check:
  - `python scripts/check_mountains_open.py --long-horizon-report reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json --lifelong-report reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json`
  - result: `[OPEN]`

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced.
