# ROADMAP v2: Gate-Driven AGI-Ready Program

Updated: 2026-02-17

## 1. Goal
- Maintain a reproducible AGI-ready research contour with measurable criteria across all 8 technical mountains.
- Keep proof-by-evaluation as the governing rule: stable gates, multi-seed robustness, OOD checks, and capability metrics.

## 2. Current Snapshot (Fact)
- Primary reference report: `reports/agi_v1.quick.seed01234.rebaseline_phase2.retune1.json`.
- Current gates:
  - `gate0=pass`
  - `gate1=pass`
  - `gate2=pass`
  - `gate3=pass`
  - `gate4=pass`
- Independent rerun #2 (same config): `reports/agi_v1.quick.seed01234.rebaseline_phase2.rerun2.json` (`gate0..gate4=pass`).
- Canonical baseline health:
  - `core` and `language` are `status=ok` (no active `OSError: [Errno 22] Invalid argument`).
  - `overall.confidence = 0.8836` (>= `0.80`).
- Capability vector:
  - `generalization_score = 0.8484`
  - `sample_efficiency_score = 0.8548`
  - `robustness_score = 0.9327`
  - `tool_workflow_score = 0.9583`
- Canonical 5-seed suite snapshot:
  - `long_horizon.score = 0.8135` (`ci.half_width = 0.0144`)
  - `lifelong.score = 0.5852` (`forgetting_gap = 0.2033`, `forward_transfer = 1.7203`, `ci.half_width = 0.0648`)
  - `safety.score = 0.7830` (`ci.half_width = 0.0623`)
  - `tools.score = 0.9167` (`ci.half_width = 0.0517`)
  - `core.score = 1.0000` (`ci.half_width = 0.0202`)
  - `language.score = 0.7095` (`ci.half_width = 0.0135`)
  - `social.score = 0.8500` (`ci.half_width = 0.0995`)
- Historical priority-suite seed0 smoke after infrastructure/tuning:
  - artifact: `reports/bench_priority_quick_seed0.autonomy2.json`
  - `long_horizon.score = 0.7063`
  - `lifelong.score = 0.6612`
  - `safety.score = 0.8178`
- Historical priority-suite 2-seed sanity:
  - artifact: `reports/bench_priority_quick_seed01.autonomy1.json`
  - `long_horizon.score = 0.7264`
  - `lifelong.score = 0.5718`
  - `safety.score = 0.7703`
- Historical priority-suite 5-seed quick snapshots (isolated):
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
- Internal Gate2-Strict mountain opener:
  - `python scripts/check_mountains_open.py --long-horizon-report reports/bench_long_horizon_quick_seed01234.p0s2_rewardaware.json --lifelong-report reports/bench_lifelong_quick_seed01234.p0s2_rewardaware.json`
  - result: `[OPEN]` (Mountain #2/#3 opened on isolated suites)
- Infra update completed:
  - dedicated `long_horizon` suite added to AGI-bench runs
  - `trainer.evaluate()` now emits operational safety metrics (`constraint_compliance`, `catastrophic_fail_rate`, `death_rate`, `reason_counts`)
  - `safety` suite now combines planner smoke + runtime safety metrics (no longer placeholder-only)
  - lifelong CI now computed from per-run lifelong score instead of raw transfer deltas
  - `ExperimentLogger` now sanitizes `run_id` for Windows filename safety

## 2.1 Active Execution Priority
1. Keep Gate0-3 stability under repeated runs and acceptance packs (`quick/full/OOD`).
2. Mountain #7: safety/alignment hardening (catastrophic failure reduction, stronger compliance).
3. Expand mountain depth (#1/#4/#6) while preserving Mountain #2/#3 openness.
4. Maintain low-flake infra and regression guardrails.

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
Status: confirmed on canonical 5-seed quick baseline.
- Gate2 pass is established on `reports/agi_v1.quick.seed01234.rebaseline_phase2.retune1.json`.
- Maintain regression checks to keep suite thresholds stable after new tuning cycles.

### Phase C (Close Gate3)
Status: confirmed on canonical 5-seed quick baseline + independent rerun #2.
- Gate3 CI thresholds pass on both `retune1` and `rerun2` artifacts.
- Keep variance controls and CI sampling discipline under future changes.

### Phase D (8-mountain completeness)
Status: in progress.
- Long-horizon and safety base suites are now wired into bench; Mountain #2/#3 are open in isolated 5-seed checks.
- Current blockers are depth and breadth beyond gate math: stronger OOD evidence, catastrophic safety reduction, deeper ToM/social transfer, and richer multimodality.

## 5. 8-Mountain Matrix (metric -> test -> DoD)

| Mountain | Primary metrics | Required tests | DoD |
|---|---|---|---|
| 1. Cross-domain generality | `generalization_score`, per-suite transfer gaps | core/tools/language/social OOD runs | Shared architecture passes OOD tolerances |
| 2. Long horizon planning | horizon success at 100+ steps, subgoal completion | dedicated long-horizon suite (implemented) | stable score with bounded variance |
| 3. Lifelong learning | `forgetting_gap`, `forward_transfer` | lifelong stress suite, replay ablations | no catastrophic forgetting under drift |
| 4. Language/abstraction | instruction success, causal drop, explanation checks | language in/out-of-distribution tasks | stable instruction execution + auditable reasoning traces |
| 5. Tools/workflows | unmasked pass rate, steps-to-pass, workflow completion | repo loop and open workflow suites | multi-step code workflow success without mask dependence |
| 6. Social reasoning | social success/transfer, ToM proxy metrics | social basic + transfer/OOD scenarios | stable cooperation/competition transfer |
| 7. Safety/alignment | constraint compliance, catastrophic fail rate, anti-hacking score | safety suite + safety/OOD/adversarial packs | constraints respected across novel settings |
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
  - quick/full: long_horizon/core/tools/tools_open/language/social/lifelong/safety
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
