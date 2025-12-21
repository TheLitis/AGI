"""
Sweep runner across agent variants/seeds for stage4 + lifecycle evaluation.
Ensures JSON output is valid (no NaN/Inf) and written atomically.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from experiment import run_experiment

VARIANTS = ["full", "no_reflection", "no_self"]
LIFELONG_VARIANTS = ["full", "no_reflection", "no_self"]
SEEDS = [0, 1, 2, 3, 4]


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, int)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run sweeps over variants/seeds.")
    parser.add_argument(
        "--lifelong",
        action="store_true",
        help="Shortcut for --mode lifelong.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="stage4",
        choices=["stage4", "lifelong"],
        help="Sweep mode: stage4 (default) or lifelong non-stationary eval.",
    )
    parser.add_argument(
        "--lifelong-episodes-per-chapter",
        type=int,
        default=50,
        help="Episodes per regime chapter when --mode lifelong.",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="gridworld",
        choices=["gridworld", "minigrid", "tools", "computer", "repo", "mixed"],
        help="Environment family: gridworld (toy default), minigrid, tools, computer, repo, or mixed.",
    )
    parser.add_argument(
        "--minigrid-scenarios",
        type=str,
        default=None,
        help="Comma-separated MiniGrid scenario aliases; only used when env-type != gridworld.",
    )
    parser.add_argument(
        "--computer-scenarios",
        type=str,
        default=None,
        help="Comma-separated ComputerEnv scenario aliases (only used when env-type is computer/mixed).",
    )
    parser.add_argument(
        "--repo-scenarios",
        type=str,
        default=None,
        help="Comma-separated RepoToolEnv task aliases (supports train:/test: prefixes).",
    )
    parser.add_argument(
        "--regime-aware-replay",
        action="store_true",
        help="Enable regime-aware replay mixing current/past regimes during training.",
    )
    parser.add_argument(
        "--skill-mode",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "latent", "mixed"],
        help="Skill selection backend: handcrafted (default), latent, or mixed.",
    )
    parser.add_argument(
        "--n-latent-skills",
        type=int,
        default=0,
        help="Number of latent skills (used when skill-mode=latent/mixed).",
    )
    parser.add_argument(
        "--use-skills",
        action="store_true",
        help="Enable hierarchical skills and high-level policy during sweeps.",
    )
    args = parser.parse_args()

    sweep_mode = "lifelong" if args.lifelong else args.mode
    lifelong_eps = args.lifelong_episodes_per_chapter
    env_type = args.env_type
    scenario_list = None
    if args.minigrid_scenarios:
        scenario_list = [s.strip() for s in args.minigrid_scenarios.split(",") if s.strip()]
    computer_list = None
    if args.computer_scenarios:
        computer_list = [s.strip() for s in args.computer_scenarios.split(",") if s.strip()]
    repo_list = None
    if args.repo_scenarios:
        repo_list = [s.strip() for s in args.repo_scenarios.split(",") if s.strip()]
    regime_aware_replay = bool(args.regime_aware_replay)
    skill_mode = args.skill_mode
    n_latent_skills = args.n_latent_skills
    use_skills = bool(args.use_skills)

    results = []
    sweep_out_dir = Path("sweep_results")
    sweep_out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("sweep_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print("===============================================")
    print(" Lifelong regime sweep" if sweep_mode == "lifelong" else " Stage4 + lifecycle sweep")
    variants = LIFELONG_VARIANTS if sweep_mode == "lifelong" else VARIANTS

    print(" Variants:", variants)
    print(" Seeds:", SEEDS)
    print(" Env:", env_type)
    print("===============================================")

    for variant in variants:
        for seed in SEEDS:
            run_id_base = f"{variant}_seed{seed}_{sweep_mode}"
            run_id = f"{run_id_base}_{env_type}" if env_type != "gridworld" else run_id_base
            print("\n-----------------------------------------------")
            print(f"START variant={variant}, seed={seed}")
            print("-----------------------------------------------")
            try:
                res = run_experiment(
                    seed=seed,
                    mode=sweep_mode if sweep_mode == "lifelong" else "stage4",
                    agent_variant=variant,
                    env_type=env_type,
                    schedule_mode="iid",
                    episodes_per_phase=50,
                    planning_horizon=12,
                    planner_mode="rollout",
                    planner_rollouts=4,
                    log_dir=str(log_dir),
                    run_id=run_id,
                    lifelong_episodes_per_chapter=lifelong_eps,
                    minigrid_scenarios=scenario_list,
                    computer_scenarios=computer_list,
                    repo_scenarios=repo_list,
                    regime_aware_replay=regime_aware_replay,
                    skill_mode=skill_mode,
                    n_latent_skills=n_latent_skills,
                    use_skills=use_skills,
                )
                results.append(res)
            except Exception as e:
                print(f"[ERROR] variant={variant}, seed={seed} crashed: {e}")
                continue
            print(f"FINISH variant={variant}, seed={seed}")

    suffix = "" if env_type == "gridworld" else f"_{env_type}"
    if sweep_mode == "lifelong":
        out_name = f"lifelong_sweep{suffix}.json"
    else:
        out_name = f"stage4_lifecycle_sweep{suffix}.json"
    out_path = sweep_out_dir / out_name
    tmp_path = out_path.with_suffix(".json.tmp")
    sanitized = _sanitize(results)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp_path.replace(out_path)
    print(f"\n[LOG] Saved sweep results to {out_path}")

    if sweep_mode == "stage4":
        print("\n=== Summary: Phase C (test) mean_return, use_self=False, planner on ===")
        for variant in variants:
            vals = []
            for res in results:
                if res.get("agent_variant") != variant:
                    continue
                sm = res.get("stage_metrics", {})
                phase_c = sm.get("lifecycle_phaseC_no_self", {})
                if isinstance(phase_c, dict) and "mean_return" in phase_c:
                    vals.append(phase_c["mean_return"])
            if vals:
                mean_v = statistics.mean(vals)
                std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                print(f"{variant}: n={len(vals)}, mean={mean_v:.3f}, std={std_v:.3f}")
            else:
                print(f"{variant}: no data")
    else:
        print("\n=== Summary: Lifelong mean returns per chapter ===")
        for variant in variants:
            rows = []
            for res in results:
                if res.get("agent_variant") != variant:
                    continue
                per_chapter = (
                    res.get("stage_metrics", {})
                    .get("lifelong_eval", {})
                    .get("lifelong_per_chapter", [])
                    or []
                )
                if per_chapter:
                    rows.append([ch.get("mean_return") for ch in per_chapter])
            if rows:
                avg_per_chapter = [
                    statistics.mean([r[i] for r in rows if len(r) > i and isinstance(r[i], (int, float))])
                    for i in range(len(rows[0]))
                    if any(len(r) > i and isinstance(r[i], (int, float)) for r in rows)
                ]
                formatted = ", ".join(f"{v:.3f}" for v in avg_per_chapter)
                print(f"{variant}: {formatted} (n={len(rows)})")
            else:
                print(f"{variant}: no data")


if __name__ == "__main__":
    main()
