# ARC-AGI-3 Strategy

Last reviewed: 2026-05-05

## Why This Matters

ARC-AGI-3 moves the target from static grid transformations to interactive reasoning. The official competition page describes agents that must interact with novel environments without instructions, and identifies four target capabilities: exploration, modeling, goal-setting, and planning/execution. This is aligned with the project's existing eight-mountain roadmap, especially long-horizon planning, world modeling, lifelong learning, tools, and safety.

Primary references:

- ARC-AGI-3 competition: https://arcprize.org/competitions/2026/arc-agi-3
- ARC-AGI-3 docs: https://docs.arcprize.org/
- ARC-AGI toolkit: https://github.com/arcprize/arc-agi
- ARC-AGI-3 technical report: https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf

## Competition Constraints We Must Respect

- Evaluation has no internet access.
- Prize-eligible solutions must open-source code and methods.
- Submission goes through Kaggle.
- ARC-AGI-3 rewards efficient adaptation in novel interactive environments; public-game overfitting is not a winning research strategy.

## Research Hypothesis

The shortest credible path is not "attach a bigger LLM." It is an agent loop specialized for interactive abstraction:

1. Perception: convert ARC frame layers into compact tokens and object/event summaries.
2. Exploration: actively probe action affordances and map causal effects.
3. World model: learn transition/event predictions from replay.
4. Goal acquisition: infer progress signals from level changes, state transitions, toggles, and novelty collapse.
5. Planning: search over learned macro-actions, not only primitive actions.
6. Memory: keep per-game causal graphs and reusable strategy templates.

## Engineering Milestones

### M0: Official Harness

- Add official `arc-agi` SDK dependency.
- Add adapter from `FrameDataRaw` to `ObsPacket`.
- Add bounded smoke runner that writes JSON reports.
- Add CI tests that do not require network or API keys.

Status: implemented as initial scaffold.

### M1: Baseline Agents

- Random legal action baseline.
- First/last legal action deterministic baselines.
- Curiosity baseline: maximize frame delta and new object/event observations.
- Reset-aware baseline: avoid repeated game-over loops.

Acceptance:

- Run on all public games available to our key.
- Produce per-game replay + score artifacts.
- Dashboard includes ARC score, win rate, steps, action entropy, frame novelty.

### M2: Causal Mapper

- Learn action -> frame delta signatures.
- Detect controllable sprites, toggles, keys/doors, hazards, level completion events.
- Build per-game transition graph.

Acceptance:

- Causal graph artifact per game.
- Replay visualization shows action effects and discovered affordances.

### M3: Macro Planner

- Compose macros from repeated action-effect chains.
- Search over macros with budgeted rollouts.
- Prefer actions that increase predicted progress or unlock new observations.

Acceptance:

- Beat random/curiosity baselines on public games.
- Positive planner gain on held-out seeds.

### M4: Submission Track

- Freeze dependency versions.
- Implement Kaggle-compatible offline runner.
- Remove any network/runtime assumptions.
- Open-source methods and write reproducible report.

## Immediate Implementation Notes

- The adapter is separate from `bench.py` for now so Gate2 work is not destabilized.
- Next integration step is an `arc_agi3` optional suite after baseline artifacts are stable.
- Our existing visual replay machinery should be reused for ARC frame timelines.
