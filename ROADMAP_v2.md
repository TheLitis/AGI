# ROADMAP v2: Gate-Driven AGI-Ready Program

Updated: 2026-02-15

## 1. Goal
- Maintain a reproducible AGI-ready research contour with measurable criteria across all 8 technical mountains.
- Keep proof-by-evaluation as the governing rule: stable gates, multi-seed robustness, OOD checks, and capability metrics.

## 2. Current Snapshot (Fact)
- Primary reference report: `reports/agi_v1.quick.seed01234.stab2.json`.
- Current gates:
  - `gate0=pass`
  - `gate1=pass`
  - `gate2=pass`
  - `gate3=pass`
  - `gate4=fail`
- Gate4 blocker:
  - `overall.confidence = 0.7846` (< `0.80`).
- Capability vector (already above thresholds):
  - `generalization_score = 0.8314`
  - `sample_efficiency_score = 0.8548`
  - `robustness_score = 0.8864`
  - `tool_workflow_score = 0.95`

## 3. Gate Model (DoD)

### Gate2 (performance thresholds)
Pass conditions:
- core:
  - `core.score >= 0.90`
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

### Gate4 (capability-complete contour)
Pass conditions:
- Gate3 already pass.
- Capability vector thresholds:
  - `generalization_score >= 0.80`
  - `sample_efficiency_score >= 0.75`
  - `robustness_score >= 0.75`
  - `tool_workflow_score >= 0.80`
- `overall.confidence >= 0.80`

## 4. Phase Plan and Status

### Phase A (Reproducibility hygiene)
Status: mostly complete, keep regression discipline.
- Report manifest discipline implemented (`config_hash`, `seed_list`, environment fingerprint).
- Remaining: keep rerun discipline and reduce flakiness in heavy suites.

### Phase B (Close Gate2)
Status: complete on current quick 5-seed reference.
- Tools/language/core tuning integrated.
- Lifelong and social thresholds hold on current reference.

### Phase C (Close Gate3)
Status: complete for current quick 5-seed reference.
- CI thresholds met for all Gate3 suites.
- Remaining hardening: independent rerun confirmation and full+OOD parity.

### Phase D (8-mountain completeness)
Status: in progress.
- Current blockers are depth/coverage (especially long-horizon, safety/adversarial, rich ToM, stronger multimodality), not only gate math.

## 5. 8-Mountain Matrix (metric -> test -> DoD)

| Mountain | Primary metrics | Required tests | DoD |
|---|---|---|---|
| 1. Cross-domain generality | `generalization_score`, per-suite transfer gaps | core/tools/language/social OOD runs | Shared architecture passes OOD tolerances |
| 2. Long horizon planning | horizon success at 100+ steps, subgoal completion | dedicated long-horizon suite | stable score with bounded variance |
| 3. Lifelong learning | `forgetting_gap`, `forward_transfer` | lifelong stress suite, replay ablations | no catastrophic forgetting under drift |
| 4. Language/abstraction | instruction success, causal drop, explanation checks | language in/out-of-distribution tasks | stable instruction execution + auditable reasoning traces |
| 5. Tools/workflows | unmasked pass rate, steps-to-pass, workflow completion | repo loop and open workflow suites | multi-step code workflow success without mask dependence |
| 6. Social reasoning | social success/transfer, ToM proxy metrics | social basic + transfer/OOD scenarios | stable cooperation/competition transfer |
| 7. Safety/alignment | constraint compliance, catastrophic fail rate, anti-hacking score | safety/OOD/adversarial packs | constraints respected across novel settings |
| 8. Engineering/scale | run reproducibility, CI health, regression rate | smoke/gate/regression pipelines | low-flake reproducible benchmark ops |

## 6. Public Interfaces (required)
- `bench.py` schema `0.2`:
  - `overall.capabilities`
  - `overall.confidence`
  - `suites[].ci`
  - `meta.run_manifest`
  - explicit `gate2/gate3/gate4`
- `trainer.py`:
  - structured stability/internalization/planning metrics
  - episode-level trace export for analysis
- `ROADMAP.md`, `ROADMAP_v2.md`, `CHECKLIST.md`:
  - synced DoD, commands, acceptance criteria

## 7. Mandatory Test Packs
- Unit:
  - gate logic correctness
  - schema `0.2` fields and backward-safe handling
- Integration:
  - quick/full: core/tools/tools_open/language/social/lifelong/safety
  - OOD pack for critical suites
- Regression:
  - baseline commit comparison and degradation guardrails
- Acceptance:
  - stable Gate2 pass
  - stable Gate3 pass
  - 8-mountain matrix fully populated and auto-validated

## 8. AGI Claim Policy (for this repository)
A "real AGI" claim is allowed only if all conditions hold:
1. Gate4 pass on the current main branch.
2. Reproduced in at least two independent reruns.
3. Multi-seed (`0..4`) and OOD packs pass with bounded variance.
4. 8-mountain matrix has no missing mandatory metric/test/threshold.
5. Evidence artifacts are present and auditable in reports.

If any condition fails, wording must be "AGI-ready research prototype" and include the exact missing conditions.

## 9. Commit and Execution Protocol
- One logical step = one commit.
- Push immediately after each commit to `main`.
- No monolithic mixed-scope commits.
- Every milestone produces:
  - benchmark report(s) in `reports/`
  - short changelog summary
  - explicit gate status delta
