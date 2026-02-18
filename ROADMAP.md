# ROADMAP (Synced)

Updated: 2026-02-18

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed01234.safetygate_v1.json`.
- Current gates: `gate0=pass`, `gate1=fail`, `gate2=fail`, `gate3=fail`, `gate4=fail`.
- Safety-blocking cutover is active:
  - Gate2 requires `constraint_compliance >= 0.85` and `catastrophic_fail_rate <= 0.05`.
  - Gate4 requires `constraint_compliance >= 0.90` and `catastrophic_fail_rate <= 0.02`.
- Restored safety snapshot artifact: `reports/bench_safety_quick_seed01234.autonomy4.json`.
- Internal mountain opener status (isolated 5-seed reports):
  - Mountain #2 (`long_horizon`) remains `open` via `reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json` (`score = 0.6847`).
  - Mountain #3 (`lifelong`) remains `open` via `reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json` (`forgetting_gap = 0.7799`, `forward_transfer = 0.6950`).

## Confirmed Metrics Snapshot (Canonical Safety Gate v1)
- overall:
  - `agi_score = 0.6472`
  - `overall.confidence = 0.8357`
- capabilities:
  - `generalization_score = 0.8701`
  - `sample_efficiency_score = 0.8715`
  - `robustness_score = 0.9111`
  - `tool_workflow_score = 0.6167`
- suite status on primary reference:
  - `long_horizon = ok`, `score = 0.7054`, `ci.half_width = 0.2427`, `catastrophic_fail_rate = 0.15`, `timeout_rate = 0.45`
  - `tools = ok`, `score = 0.2333`, `pass_rate_unmasked = 0.2333`, `mean_steps_to_pass_unmasked = 3.0`
  - `social = ok`, `score = 0.8500`, `transfer_rate = 0.8867`
  - `lifelong = ok`, `score = 0.5689`, `forgetting_gap = 2.6738`, `forward_transfer = 1.3876`
  - `safety = ok`, `score = 0.8041`, `constraint_compliance = 0.65`, `catastrophic_fail_rate = 0.20`
  - `core = ok`, `score = 1.0000`
  - `language = ok`, `score = 0.7429`, `pass_rate = 0.7429`, `causal_drop = 0.0`

## Active Priority Order
1. Mountain #7 first: ship `Safety+OOD` pack and reduce catastrophic failures/compliance gap to blocking thresholds.
2. Recover `tools` reliability under the stricter policy (`pass_rate_unmasked` and workflow stability).
3. Keep Mountain #2/#3 openness while increasing planner reality signal and lifelong stability.
4. Expand mountain depth (#1/#4/#6) without regressing reproducibility and CI health.

## Immediate Next Milestones
1. Run full 5-seed acceptance with safety-blocking policy: `quick/full/OOD` consistency.
2. Add adversarial safety/OOD pack and make it mandatory in acceptance gates.
3. Add planner “pay rent” blocking diagnostics (`corr_adv`, `top1_advantage`, `planner_gain`).
4. Start multimodal/tokenized-world expansion with explicit transfer metrics.

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced under the safety-blocking policy.
