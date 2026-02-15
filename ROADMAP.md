# ROADMAP (Synced)

Updated: 2026-02-15

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed0.codex_audit.json`.
- Current gates: `gate0=fail`, `gate1=fail`, `gate2=fail`, `gate3=fail`, `gate4=fail`.
- Internal mountain status (Gate2-Strict, isolated 5-seed):
  - Mountain #2 (`long_horizon`) = `open` (`score = 0.8135`, threshold `>= 0.65`).
  - Mountain #3 (`lifelong`) = `open` (`forgetting_gap = 0.2031`, `forward_transfer = 1.7203`).
- Immediate blockers:
  - `core` suite runtime failure: `OSError: [Errno 22] Invalid argument`.
  - `language` suite runtime failure: `OSError: [Errno 22] Invalid argument`.
  - `overall.confidence = 0.6381` (below Gate4 threshold `>= 0.80`).

## Active Priority Order
1. Restore canonical baseline health: fix `core/language` runtime failures (`OSError: [Errno 22] Invalid argument`).
2. Preserve Mountain #2/#3 gains while revalidating on canonical AGI quick runs.
3. Safety/alignment hardening (Mountain #7).
4. Then expansion of remaining mountains.

## Confirmed Metrics Snapshot
- capabilities:
  - `generalization_score = 0.9701`
  - `sample_efficiency_score = 1.0000`
  - `robustness_score = 0.9266`
  - `tool_workflow_score = 0.9167`
- required suite status on validated baseline:
  - `long_horizon = ok`, `score = 0.7063`
  - `tools = ok`, `score = 0.8333`
  - `social = ok`, `score = 1.0000`
  - `lifelong = ok`, `score = 0.5866`
  - `safety = ok`, `score = 0.8178`
  - `core = error` (`OSError: [Errno 22] Invalid argument`)
  - `language = error` (`OSError: [Errno 22] Invalid argument`)

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
1. Fix runtime failures in `core` and `language` suites (`OSError: [Errno 22] Invalid argument`) and restore `gate0=pass`.
2. Regenerate a validated multi-seed AGI quick baseline (`seeds 0..4`) including all required suites.
3. Reconfirm Gate2/Gate3 on canonical baseline after runtime fix, then continue confidence push (`overall.confidence >= 0.80`).

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
- note:
  - these priority snapshots are directional diagnostics, not the canonical validated AGI reference.
- 5-seed quick snapshots (priority suites, isolated runs, latest):
  - `reports/bench_long_horizon_quick_seed01234.phase1_m23.json`
    - `long_horizon.score = 0.8135`
    - `long_horizon.ci.half_width = 0.0144`
  - `reports/bench_lifelong_quick_seed01234.phase1_m23.json`
    - `lifelong.score = 0.5852`
    - `lifelong.forgetting_gap = 0.2031`
    - `lifelong.forward_transfer = 1.7203`
    - `lifelong.ci.half_width = 0.0648`
  - `reports/bench_safety_quick_seed01234.autonomy4.json`
    - `safety.score = 0.7444`
    - `constraint_compliance = 0.55`
    - `catastrophic_fail_rate = 0.25`
- mountain opener check:
  - `python scripts/check_mountains_open.py --long-horizon-report reports/bench_long_horizon_quick_seed01234.phase1_m23.json --lifelong-report reports/bench_lifelong_quick_seed01234.phase1_m23.json`
  - result: `[OPEN]`

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced.
