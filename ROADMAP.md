# ROADMAP (Synced)

Updated: 2026-02-21

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed01234.safetygate_v1.json`.
- Current gates: `gate0=pass`, `gate1=fail`, `gate2=fail`, `gate3=fail`, `gate4=fail`.
- Safety-blocking cutover is active:
  - Gate2 requires `constraint_compliance >= 0.85` and `catastrophic_fail_rate <= 0.05`.
  - Gate4 requires `constraint_compliance >= 0.90` and `catastrophic_fail_rate <= 0.02`.
- Safety quick artifacts (5 seeds):
  - `reports/milestones/20260220_phase1_safety_checkpoint_select.quick.cuda.json`
  - `reports/milestones/20260220_phase1_safety_ood_checkpoint_select.quick.cuda.json`
- Internal mountain opener status (isolated 5-seed reports):
  - Mountain #2 (`long_horizon`) remains `open` via `reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json` (`score = 0.6847`).
  - Mountain #3 (`lifelong`) remains `open` via `reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json` (`forgetting_gap = 0.7799`, `forward_transfer = 0.6950`).

## Confirmed Metrics Snapshot (Latest Quick 5-Seed)
- overall:
  - `agi_score = 0.0000` (geometric mean zero due `tools.score = 0.0`)
  - `overall.confidence = 0.8436`
- capabilities:
  - `generalization_score = 0.6716`
  - `sample_efficiency_score = 0.8640`
  - `robustness_score = 0.6946`
  - `tool_workflow_score = 0.7917`
- suite status on primary reference:
  - `long_horizon = ok`, `score = 0.3814`, `catastrophic_fail_rate = 0.1750`
  - `tools = ok`, `score = 0.0000`, `pass_rate_unmasked = 0.5833`, `mean_steps_to_pass_unmasked = 9.2857`, `invalid_action_rate = 0.0473`
  - `social = ok`, `score = 0.7625`, `success_rate = 0.7625`, `transfer_rate = 0.8317`
  - `lifelong = ok`, `score = 0.4997`, `forgetting_gap = 0.0213`, `forward_transfer = -0.0066`
  - `safety = ok`, `score = 0.9565`, `constraint_compliance = 0.8750`, `catastrophic_fail_rate = 0.0000`
  - `safety_ood = ok`, `score = 0.9705`, `constraint_compliance = 0.9375`, `catastrophic_fail_rate = 0.0250`
  - `core = ok`, `score = 0.5003`
  - `language = ok`, `score = 0.7279`, `pass_rate = 0.7279`, `causal_drop = 0.0`

## Active Priority Order
1. Keep Mountain #7 safety blocker closed on canonical quick/full/OOD reports.
2. Recover `tools` reliability under stricter policy (`pass_rate_unmasked` and workflow stability).
3. Recover `long_horizon` score and positive `lifelong.forward_transfer` on canonical runs.
4. Expand mountain depth (#1/#4/#6) without regressing reproducibility and CI health.

## Immediate Next Milestones
1. Run full 5-seed acceptance with safety-blocking policy: `quick/full/OOD` consistency.
2. Add adversarial safety/OOD pack and make it mandatory in acceptance gates.
3. Improve planner “pay rent” diagnostics (`corr_adv`, `top1_advantage`, `planner_gain`).
4. Continue multimodal/tokenized-world expansion with explicit transfer metrics.

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced under the safety-blocking policy.
