# ROADMAP (Synced)

Updated: 2026-02-12

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Gate status target track: `Gate2 -> Gate3 -> Gate4`.
- Baseline reference: `reports/agi_v1.quick.post_lifelong_replay05.seed0.json`.
- Known baseline state: `gate0=pass`, `gate1=pass`, `gate2=fail`.

## What Is Already Implemented
- Report schema upgraded to `0.2` in `bench.py`.
- New report fields:
  - `meta.run_manifest`
  - `suites[].ci`
  - `overall.capabilities`
  - `overall.confidence`
  - explicit `gate3`, `gate4`
- Bench tests updated for new gate and schema behavior.

## Immediate Next Milestones
1. Close Gate2 robustly on seeds `0,1,2` with repeated runs.
2. Add OOD + 5-seed stability discipline to close Gate3.
3. Expand suites/metrics to complete 8-mountain matrix and Gate4 policy checks.

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and reproduced.
