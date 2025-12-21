"""
Small benchmark battery runner for the proto-creature AGI project.

Goal: make progress measurable across env families (gridworld/minigrid/repo/mixed)
without rewriting ad-hoc commands each time.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment import run_experiment


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, int)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    return obj


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts or None


@dataclass(frozen=True)
class BenchCase:
    name: str
    env_type: str
    minigrid_scenarios: Optional[List[str]] = None
    computer_scenarios: Optional[List[str]] = None
    repo_scenarios: Optional[List[str]] = None


DEFAULT_CASES: List[BenchCase] = [
    BenchCase(name="gridworld", env_type="gridworld"),
    BenchCase(name="instruction", env_type="instruction"),
    BenchCase(name="social", env_type="social"),
    BenchCase(
        name="minigrid",
        env_type="minigrid",
        minigrid_scenarios=["minigrid-empty", "minigrid-doorkey", "test:minigrid-lavacrossing"],
    ),
    BenchCase(
        name="repo",
        env_type="repo",
        repo_scenarios=["train:calc_add", "train:calc_pow", "test:calc_div", "test:calc_bundle"],
    ),
    BenchCase(
        name="mixed",
        env_type="mixed",
        minigrid_scenarios=["minigrid-empty", "minigrid-doorkey"],
        computer_scenarios=["simple_project", "refactor_project"],
        repo_scenarios=["train:calc_add", "test:calc_div"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small benchmark battery across env families.")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: bench_results/bench_<ts>.json).")
    parser.add_argument("--log-dir", type=str, default="bench_logs", help="Per-run JSONL logs directory.")
    parser.add_argument("--mode", type=str, default="stage4", choices=["stage4", "lifelong"], help="Experiment mode.")
    parser.add_argument(
        "--variants",
        type=str,
        default="full,no_reflection,no_self",
        help="Comma-separated agent variants to run.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1], help="Seeds to run (default: 0 1).")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of cases to run (gridworld|instruction|social|minigrid|repo|mixed).",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller/faster settings (good for smoke checks).")
    parser.add_argument("--use-skills", action="store_true", help="Enable hierarchical skills.")
    parser.add_argument(
        "--skill-mode",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "latent", "mixed"],
        help="Skill backend (only used when --use-skills).",
    )
    parser.add_argument("--n-latent-skills", type=int, default=0, help="Number of latent skills.")

    # Optional scenario overrides
    parser.add_argument("--minigrid-scenarios", type=str, default=None, help="Override default MiniGrid scenarios (CSV).")
    parser.add_argument("--computer-scenarios", type=str, default=None, help="Override default Computer scenarios (CSV).")
    parser.add_argument("--repo-scenarios", type=str, default=None, help="Override default Repo scenarios (CSV).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ts = int(time.time())
    out_path = Path(args.out or (Path("bench_results") / f"bench_{ts}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    seeds = list(args.seeds or [])
    only = set(_split_csv(args.only) or [])

    override_minigrid = _split_csv(args.minigrid_scenarios)
    override_computer = _split_csv(args.computer_scenarios)
    override_repo = _split_csv(args.repo_scenarios)

    cases: List[BenchCase] = []
    for c in DEFAULT_CASES:
        if only and c.name not in only:
            continue
        cases.append(
            BenchCase(
                name=c.name,
                env_type=c.env_type,
                minigrid_scenarios=override_minigrid if c.env_type in {"minigrid", "mixed"} and override_minigrid else c.minigrid_scenarios,
                computer_scenarios=override_computer if c.env_type in {"computer", "mixed"} and override_computer else c.computer_scenarios,
                repo_scenarios=override_repo if c.env_type in {"repo", "mixed"} and override_repo else c.repo_scenarios,
            )
        )

    if not cases:
        raise SystemExit("No benchmark cases selected.")

    # Settings tuned for practicality: not a full research run, but stable enough to compare changes.
    episodes_per_phase = 50
    n_steps = 1024
    planning_horizon = 12
    planner_rollouts = 4
    planning_coef = 0.3
    lifelong_eps = 50
    if args.quick:
        episodes_per_phase = 15
        n_steps = 256
        planning_horizon = 8
        planner_rollouts = 2
        planning_coef = 0.25
        lifelong_eps = 20

    results: List[Dict[str, Any]] = []
    for case in cases:
        for variant in variants:
            for seed in seeds:
                run_id = f"bench_{case.name}_{variant}_seed{seed}_{args.mode}"
                print(f"[BENCH] case={case.name} env={case.env_type} variant={variant} seed={seed}")
                res = run_experiment(
                    seed=int(seed),
                    mode=str(args.mode),
                    agent_variant=str(variant),
                    env_type=str(case.env_type),
                    schedule_mode="iid",
                    episodes_per_phase=int(episodes_per_phase),
                    n_steps=int(n_steps),
                    planning_horizon=int(planning_horizon),
                    planner_mode="rollout",
                    planner_rollouts=int(planner_rollouts),
                    planning_coef=float(planning_coef),
                    lifelong_episodes_per_chapter=int(lifelong_eps),
                    minigrid_scenarios=case.minigrid_scenarios,
                    computer_scenarios=case.computer_scenarios,
                    repo_scenarios=case.repo_scenarios,
                    log_dir=str(args.log_dir),
                    run_id=run_id,
                    use_skills=bool(args.use_skills),
                    skill_mode=str(args.skill_mode),
                    n_latent_skills=int(args.n_latent_skills),
                )
                res = dict(res or {})
                res["bench_case"] = case.name
                results.append(res)

    payload = {
        "timestamp": ts,
        "mode": str(args.mode),
        "quick": bool(args.quick),
        "variants": variants,
        "seeds": seeds,
        "cases": [c.name for c in cases],
        "results": results,
    }
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp_path.replace(out_path)
    print(f"[BENCH] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
