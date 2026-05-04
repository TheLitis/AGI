# Research Alignment Notes

Last reviewed: 2026-05-04

This document maps current frontier-agent directions to concrete project work. It is not a claim that the project matches frontier labs; it is a guardrail against optimizing only local sandbox gates.

## Current Gap

Our strongest engineering base is `bench -> suites -> gates -> reports`, plus sandbox environments and a cognitive stack with world/self/planner/policy modules. The largest gap versus frontier systems is not a single model size issue. It is the combination of:

- Real tool and computer-use workflows instead of mostly discrete sandbox actions.
- Rich world models and multimodal inputs instead of fixed grid patches plus hashed text.
- High-throughput experiment infrastructure instead of serial, single-step rollouts.
- Visual/trace observability instead of mostly scalar report metrics.
- Stronger evaluation integrity, including strict gate recomputation and required OOD/safety suites.

## Frontier Signals To Track

### OpenAI: Computer-Using and Unified Agents

Primary references:

- OpenAI Computer-Using Agent: https://openai.com/index/computer-using-agent/
- OpenAI ChatGPT Agent: https://openai.com/index/introducing-chatgpt-agent/

Relevant direction:

- Agents operate through browser/computer environments, not only friendly APIs.
- Toolboxes combine browsing, research synthesis, code execution, connectors, and user-facing deliverables.
- Robustness is trained with reinforcement learning and instruction hierarchy.

Project implication:

- `RepoToolEnv` should evolve from candidate patch selection to typed tool calls, edit/generate/debug loops, and sandboxed computer-like workflows.
- Bench needs workflow traces: action, tool name, args, result, recovery, regression, and safety events.
- Add visual replay and tool timeline panels before making claims about tool intelligence.

### Google DeepMind: World Models and Discovery Agents

Primary references:

- Genie 3: https://deepmind.google/models/genie/
- Genie 3 blog: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- AlphaEvolve: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/

Relevant direction:

- Interactive world models are becoming a central route for scalable agent training.
- Discovery agents combine LLM generation, evaluation loops, and search/evolution over candidate solutions.

Project implication:

- The current world model needs better trace-based diagnostics: prediction error by scenario, horizon, event type, and planner regret.
- Add an `algorithm_discovery` or `repo_search` suite where the agent proposes variants, runs tests/benchmarks, and keeps the best candidate.
- Long-horizon planning should be evaluated by realized planner gain and regret, not only return.

### Anthropic: Computer Use, MCP, Interpretability, and Coding Agents

Primary references:

- Developing computer use: https://www.anthropic.com/news/developing-computer-use
- Computer use docs: https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- Model Context Protocol: https://www.anthropic.com/news/model-context-protocol
- Tracing thoughts in language models: https://www.anthropic.com/research/tracing-thoughts-language-model
- Claude Code: https://www.anthropic.com/product/claude-code

Relevant direction:

- Tool interfaces are moving toward typed, inspectable protocols and secure boundaries.
- Agentic coding products are expected to read code, edit files, run tests, iterate, and deliver commits.
- Interpretability is becoming a practical alignment layer, not a side quest.

Project implication:

- Treat `ToolCallEnvelope` as the start of an MCP-like internal protocol: typed calls, args, return schema, sandbox policy, and audit log.
- Add a `coding_agent_trace` artifact for repo tasks: files read, patch generated/applied, test output summary, failure recovery.
- Add introspection panels for traits, risk, planner influence, and self-reflection. Explanations without traces are not enough.

## Immediate Research-Engineering Backlog

1. Strict metrics and acceptance validation.
   - `validate_bench_report.py --strict-acceptance` must be required for canonical/milestone acceptance.
   - Gate values must be recomputed from metrics.
   - `safety_ood` must be required in acceptance.

2. Visual observability.
   - Generate `reports/visualizations/dashboard.html` for every milestone.
   - Add trajectory export next; current reports do not contain enough data for true replay.
   - Add planner influence panels before calling planner behavior interpretable.

3. Experiment throughput.
   - Remove per-step CPU/GPU synchronizations.
   - Batch environment collection or parallelize independent seeds/cases.
   - Increase update batch sizes after rollout collection is no longer the bottleneck.

4. Tool/computer-use ladder.
   - Tier 0: current candidate patch actions.
   - Tier 1: typed tool calls with args and audit log.
   - Tier 2: edit/generate code, run tests, parse logs, recover.
   - Tier 3: real sandboxed desktop/browser/computer-use tasks.

5. World-model diagnostics.
   - Log next-event prediction metrics by scenario and horizon.
   - Add planner realized-gain and regret breakdowns by scenario.
   - Add OOD world-model stress tests before larger model rewrites.

## Non-Negotiable Measurement Rules

- No gate claim without report, validator output, and gate snapshot.
- No safety claim without `safety` and `safety_ood`.
- No tools claim without unmasked pass rate, steps to pass, recovery rate, invalid action rate, and sandbox escape rate.
- No planning claim without planner gain/regret and a visual planner influence trace.
- No AGI-v2 claim without real tool/computer-use workflows and trajectory-level observability.
