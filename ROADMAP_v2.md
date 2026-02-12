# ROADMAP v2: Gate-Driven AGI-Ready Program

Updated: 2026-02-12

## 1. Goal
- Move from current baseline (`Gate0=pass`, `Gate1=pass`, `Gate2=fail`) to a reproducible AGI-ready contour with measurable criteria across all 8 technical mountains.
- Prioritize proof-by-evaluation: stable gates, multi-seed robustness, OOD checks, and capability-level metrics.

## 2. Baseline Snapshot
- Baseline reference report: `reports/agi_v1.quick.post_lifelong_replay05.seed0.json`.
- Known blockers on Gate2 path:
  - `core.score < 0.90`
  - `tools.pass_rate_unmasked < 0.85`
  - `language.pass_rate < 0.70`
- Already near/at threshold in baseline:
  - `tools.mean_steps_to_pass_unmasked <= 10`
  - social metrics above Gate2 thresholds
  - lifelong metrics above Gate2 thresholds

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

## 4. Phase Plan

### Phase A (1 week): Reproducibility hygiene
- Standardized artifact naming and temporary-file policy.
- Report manifest discipline (`commit`, `config_hash`, `seed_list`, environment fingerprint).
- Stable multi-seed smoke in local/CI checks.
Exit criteria:
- repeated quick/full runs reproducible within agreed tolerance on seeds `0,1,2`.

### Phase B (1-3 weeks): Close Gate2
- Tools:
  - targeted tuning for repo-loop settings (BC schedule, replay, mask dropout).
  - internalization metrics tracked (`invalid_mass` trend, masked/unmasked divergence).
- Language:
  - increase instruction-signal budget.
  - enforce consistent aggregation of success rate vs return.
- Core:
  - improve quick/full stability and MiniGrid readiness.
  - preserve optional dependency fallback behavior.
Exit criteria:
- `python bench.py --suite agi_v1 --quick --seeds 0,1,2` gives `Gate2=pass` in at least two repeated runs.

### Phase C (2-4 weeks): Close Gate3
- Move from single-pass to robust capability pass.
- Enforce 5-seed stability plus OOD pack checks.
- Track capability vector as first-class output.
Exit criteria:
- Gate3 pass under documented thresholds.

### Phase D (1-3 months): Close all 8 mountains in-repo
- Add required tests/metrics/thresholds per mountain.
- Automate pass/fail in bench pipeline.
Exit criteria:
- full matrix coverage and automatic validation.

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
- `repo_tool_env.py`:
  - protocol extension for longer workflow action chains
- `ROADMAP.md`, `ROADMAP_v2.md`, `CHECKLIST.md`:
  - synced DoD, commands, acceptance criteria

## 7. Mandatory Test Packs
- Unit:
  - gate logic correctness
  - schema 0.2 fields and backward-safe handling
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
