# ROADMAP v2: Gate-Driven AGI-Ready Program

Updated: 2026-02-21

## 1. Goal
- Maintain a reproducible AGI-ready research contour with measurable criteria across all 8 technical mountains.
- Keep proof-by-evaluation as the governing rule: stable gates, multi-seed robustness, OOD checks, and capability metrics.
- Enforce safety as a blocking condition for higher gates (not a soft side-metric).

## 2. Current Snapshot (Fact)
- Primary reference report: `reports/milestones/20260220_phase3_gate2_snapshot.quick.cuda.json`.
- Current gates:
  - `gate0=pass`
  - `gate1=fail`
  - `gate2=fail`
  - `gate3=fail`
  - `gate4=fail`
- Canonical quick state:
  - `overall.confidence = 0.8436`
  - `generalization_score = 0.6716`
  - `sample_efficiency_score = 0.8640`
  - `robustness_score = 0.6946`
  - `tool_workflow_score = 0.7917`
- Canonical suite snapshot:
  - `long_horizon.score = 0.3814` (`catastrophic_fail_rate = 0.1750`, `ci.half_width = 0.1782`)
  - `lifelong.score = 0.4997` (`forgetting_gap = 0.0213`, `forward_transfer = -0.0066`, `ci.half_width = 0.0008`)
  - `safety.score = 0.9565` (`constraint_compliance = 0.8750`, `catastrophic_fail_rate = 0.0000`, `ci.half_width = 0.1096`)
  - `safety_ood.score = 0.9705` (`constraint_compliance = 0.9375`, `catastrophic_fail_rate = 0.0250`, `ci.half_width = 0.0706`)
  - `tools.score = 0.0000` (`pass_rate_unmasked = 0.5833`, `mean_steps_to_pass_unmasked = 9.2857`, `invalid_action_rate = 0.0473`, `ci.half_width = 0.1265`)
  - `core.score = 0.5003` (`ci.half_width = 0.0004`)
  - `language.score = 0.7279` (`causal_drop = 0.0`, `ci.half_width = 0.0444`)
  - `social.score = 0.7625` (`transfer_rate = 0.8317`, `ci.half_width = 0.0900`)
- Dedicated safety milestone artifacts:
  - `reports/milestones/20260220_phase1_safety_checkpoint_select.quick.cuda.json` (`compliance = 0.8750`, `catastrophic = 0.0000`)
  - `reports/milestones/20260220_phase1_safety_ood_checkpoint_select.quick.cuda.json` (`compliance = 0.9375`, `catastrophic = 0.0250`)
- Internal Gate2-Strict mountain opener (isolated reports) remains open:
  - `python scripts/check_mountains_open.py --long-horizon-report reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json --lifelong-report reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json`
  - result: `[OPEN]`

## 2.1 Active Execution Priority
1. Mountain #7 first: ship mandatory `Safety+OOD` pack and hit blocking safety thresholds.
2. Recover tool-workflow reliability under new safety-blocking policy.
3. Preserve Mountain #2/#3 openness while improving planner reality diagnostics.
4. Expand mountain depth (#1/#4/#6) without regressing reproducibility.

## 3. Gate Model (DoD)

### Gate2 (performance + safety-blocking)
Pass conditions:
- core:
  - `core.score >= 0.90`
- long horizon:
  - `long_horizon.score >= 0.65`
- tools:
  - `tools.pass_rate_unmasked >= 0.85`
  - `tools.mean_steps_to_pass_unmasked <= 10.0`
- language:
  - `language.pass_rate >= 0.70`
  - `language.causal_drop <= 0.10`
- social:
  - `social.success_rate >= 0.75`
  - `social.transfer_rate >= 0.70`
- lifelong:
  - `lifelong.forgetting_gap >= -1.0`
  - `lifelong.forward_transfer >= 0.5`
- safety (blocking):
  - `safety.constraint_compliance >= 0.85`
  - `safety.catastrophic_fail_rate <= 0.05`
  - missing/non-finite/out-of-range safety metrics => fail

### Gate3 (robustness and variance)
Pass conditions:
- Gate2 already pass.
- `seed_count >= 5`.
- 95% CI half-width by suite:
  - core: `<= 1.50`
  - tools: `<= 0.10`
  - language: `<= 0.10`
  - social: `<= 0.10`
  - lifelong: `<= 0.75`

### Gate4 (capability-complete + strict safety)
Pass conditions:
- Gate3 already pass.
- Capability vector thresholds:
  - `generalization_score >= 0.80`
  - `sample_efficiency_score >= 0.75`
  - `robustness_score >= 0.75`
  - `tool_workflow_score >= 0.80`
- `overall.confidence >= 0.80`
- strict safety (blocking):
  - `safety.constraint_compliance >= 0.90`
  - `safety.catastrophic_fail_rate <= 0.02`
  - missing/non-finite/out-of-range safety metrics => fail

## 4. Phase Plan and Status

### Phase A (Reproducibility hygiene)
Status: active and healthy.
- Report manifest discipline implemented (`config_hash`, `seed_list`, environment fingerprint).
- Test/validator hardening expanded (`safety` metric checks and `.venv` pytest recursion guard).

### Phase B (Close Gate2)
Status: reopened under safety-blocking cutover.
- Prior Gate2 pass is superseded by stricter safety-blocking policy.
- Safety blockers on canonical quick are now closed (`safety` + `safety_ood` pass thresholds).
- Current blockers: `core.score`, `long_horizon.score`, `tools.pass_rate_unmasked`, and `lifelong.forward_transfer`.

### Phase C (Close Gate3)
Status: blocked by Gate2 failure.
- CI/variance checks remain in place; cannot close until Gate2 recovers under new policy.

### Phase D (8-mountain completeness)
Status: in progress.
- Mountain #2/#3 opener checks remain open in isolated reports.
- Critical blockers: Safety+OOD depth, planner reality quality, stronger ToM/social transfer, and multimodal breadth.

## 5. 8-Mountain Matrix (metric -> mechanism -> tests -> DoD)

| Mountain | Primary metrics | Mechanism (roadmap) | Required tests | DoD |
|---|---|---|---|---|
| 1. Cross-domain generality | transfer gaps, generalization score | Tokenized World Interface | core/tools/language/social + multimodal transfer pack | one architecture across >=3 observation families |
| 2. Long horizon planning | horizon success, planner reality metrics | Planner must pay rent | long_horizon + planning_diag | `corr_adv>=0`, `top1_adv>=0`, `planner_gain>=0` |
| 3. Lifelong learning | forgetting/transfer + stability | Stability triangle (replay quota + EWC/SI + adapters) | lifelong stress + replay ablations | stable return after R1→R2→R3→R1_return |
| 4. Language/abstraction | instruction success + compositional generalization | Curriculum Language Scaffolding | language IID/OOD/composition | robust composition beyond templates |
| 5. Tools/workflows | unmasked pass-rate + invalid mass | Mask internalization distillation | tools/tools_open repo loops | high unmasked success without mask crutch |
| 6. Social reasoning | success/transfer + ToM proxy | Explicit belief head | social basic + transfer/OOD | improved transfer with ToM auxiliary objective |
| 7. Safety/alignment | compliance + catastrophic rate | Constrained safety loop + Safety+OOD pack | safety + adversarial/OOD packs | blocking thresholds pass on quick/full/OOD |
| 8. Engineering/scale | reproducibility + CI health | Hardening loop protocol | smoke/gate/regression pipelines | low-flake reproducible benchmark ops |

## 6. Public Interfaces (required)
- `bench.py` schema `0.2`:
  - `overall.capabilities`
  - `overall.confidence`
  - `suites[].ci`
  - `meta.run_manifest`
  - explicit `gate2/gate3/gate4` with safety-blocking semantics
- `trainer.py`:
  - structured safety/planning/lifelong metrics from runtime eval
- `validate_bench_report.py`:
  - structural checks for required safety metrics when `safety` suite is required
- `ROADMAP.md`, `ROADMAP_v2.md`, `CHECKLIST.md`:
  - synced to current canonical artifact and gate policy

## 7. Mandatory Test Packs
- Unit:
  - gate logic correctness including safety-blocking conditions
  - report validator checks for required safety metrics
- Integration:
  - quick/full: long_horizon/core/tools/language/social/lifelong/safety
  - mandatory Safety+OOD pack in acceptance phase
- Regression:
  - baseline artifact comparison + degradation guardrails

## 8. AGI Claim Policy (for this repository)
A "real AGI" claim is allowed only if all conditions hold:
1. Gate4 pass on current main branch under safety-blocking policy.
2. Reproduced in at least two independent reruns.
3. Multi-seed (`0..4`) and OOD packs pass with bounded variance.
4. 8-mountain matrix has no missing mandatory metric/test/threshold.
5. Evidence artifacts are present and auditable in reports.

If any condition fails, wording must be "AGI-ready research prototype" and include exact missing conditions.

## 9. Commit and Execution Protocol
- One logical step = one commit.
- Push after each accepted step.
- No mixed-scope monolithic commits.
- Every milestone produces:
  - benchmark report(s) in `reports/`
  - short changelog summary
  - explicit gate status delta
