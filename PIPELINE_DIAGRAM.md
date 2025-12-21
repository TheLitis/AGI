# Pipeline note

The existing pipeline diagram stays the same; the `EnvPool` node now has multiple backends:

- `gridworld` (toy default)
- `minigrid` (benchmark tasks: empty/doorkey/lava/multiroom, selectable via `--minigrid-scenarios`)
- `tools` (arithmetic toy tool-loop)
- `computer` (simulated project workflow tasks, selectable via `--computer-scenarios`)
- `repo` (on-disk "real tool loop": apply patch candidates + run `pytest`, selectable via `--repo-scenarios`; supports procedural scenarios like `proc_arith`)
- `mixed` (hybrid pool with GridWorld + MiniGrid + Computer; optionally RepoTool when `--repo-scenarios` is set)
- Optional regime-aware replay mixes current/past regime experience for lifelong stability (`--regime-aware-replay`).

All downstream components (world model, self-model, traits, reflection, planner) are unchanged. Only the observation wiring and env selection differ via `--env-type`.
