# ROADMAP (Synced)

Updated: 2026-02-15

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed01234.stab2.json`.
- Current gates: `gate0=pass`, `gate1=pass`, `gate2=pass`, `gate3=pass`, `gate4=fail`.
- Immediate blocker to Gate4:
  - `overall.confidence = 0.7846` (threshold: `>= 0.80`).

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
- Quick language/lifelong stabilization for Gate3 reliability.

## Immediate Next Milestones
1. Raise `overall.confidence` to `>= 0.80` (primary Gate4 blocker).
2. Reproduce current Gate3 pass in at least one additional independent 5-seed rerun.
3. Complete full + OOD 5-seed acceptance runs (`--suite agi_v1`, non-quick).
4. Expand mountain coverage where tests are still shallow (long-horizon, safety/adversarial, ToM).

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced.
