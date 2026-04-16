# ROADMAP (Synced)

Updated: 2026-04-16

Source of truth: `ROADMAP_v2.md`.

## Current Stage
- Primary reference: `reports/agi_v1.quick.seed01234.safetygate_v1.json`.
- Current gates on the canonical quick 5-seed snapshot: `gate0=pass`, `gate1=pass`, `gate2=pass`, `gate3=pass`, `gate4=fail`.
- Safety-blocking cutover is active:
  - Gate2 requires `constraint_compliance >= 0.85` and `catastrophic_fail_rate <= 0.05`.
  - Gate4 requires `constraint_compliance >= 0.90` and `catastrophic_fail_rate <= 0.02`.
- Safety quick artifacts (5 seeds):
  - `reports/milestones/20260220_phase1_safety_checkpoint_select.quick.cuda.json`
  - `reports/milestones/20260220_phase1_safety_ood_checkpoint_select.quick.cuda.json`
- Acceptance backlog still open:
  - `full` 5-seed validation is incomplete and must be resumed from `reports/milestones/20260301_phase3_gate2_closure.full.cuda.iter1.json`.
  - `ood` 5-seed validation still needs a fresh milestone artifact after `full`.
- Internal mountain opener status (isolated 5-seed reports):
  - Mountain #2 (`long_horizon`) remains `open` via `reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json` (`score = 0.6847`).
  - Mountain #3 (`lifelong`) remains `open` via `reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json` (`forgetting_gap = 0.7799`, `forward_transfer = 0.6950`).

## Confirmed Metrics Snapshot (Latest Quick 5-Seed)
- overall:
  - `agi_score = 0.8272`
  - `overall.confidence = 0.9041`
- capabilities:
  - `generalization_score = 0.8781`
  - `sample_efficiency_score = 0.8733`
  - `robustness_score = 0.8812`
  - `tool_workflow_score = 1.0000`
- suite status on primary reference:
  - `long_horizon = ok`, `score = 0.7159`, `catastrophic_fail_rate = 0.0250`, `planner_gain = 4.6712`
  - `tools = ok`, `score = 1.0000`, `pass_rate_unmasked = 1.0000`, `mean_steps_to_pass_unmasked = 2.0000`, `invalid_action_rate = 0.0000`
  - `social = ok`, `score = 0.8500`, `success_rate = 0.8500`, `transfer_rate = 0.9067`
  - `lifelong = ok`, `score = 0.5384`, `forgetting_gap = 1.2881`, `forward_transfer = 0.7696`
  - `safety = ok`, `score = 0.9393`, `constraint_compliance = 0.8500`, `catastrophic_fail_rate = 0.0250`
  - `safety_ood = ok`, `score = 0.9621`, `constraint_compliance = 0.9375`, `catastrophic_fail_rate = 0.0500`
  - `core = ok`, `score = 1.0000`
  - `language = ok`, `score = 0.7403`, `pass_rate = 0.7467`, `causal_drop = 0.0063`

## Active Priority Order
1. Reproduce the canonical quick `gate2=pass` / `gate3=pass` state on `full` and `ood` 5-seed acceptance reports.
2. Keep Mountain #7 safety blocker closed while validating `full` and `ood`.
3. Improve planner "pay rent" diagnostics, which still lag despite the recovered `long_horizon` score.
4. Expand mountain depth (#1/#4/#6) without regressing reproducibility and CI health.

## Immediate Next Milestones
1. Resume and finish the staged `full` 5-seed acceptance run with `--resume`, validate it, and snapshot gate status.
2. Produce the matching `ood` 5-seed acceptance report and run the same validator / mountain / planner checks.
3. Keep regression guardrails on `safety`, `safety_ood`, and `tools` when comparing new acceptance artifacts to canonical quick.
4. Continue multimodal/tokenized-world expansion with explicit transfer metrics after acceptance artifacts are stable.

## AGI Claim Rule
Use "AGI-ready research prototype" wording until all `ROADMAP_v2.md` Gate4 conditions are satisfied and independently reproduced under the safety-blocking policy.
