"""
CLI wrapper around run_experiment for staged training and lifecycle eval.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from experiment import run_experiment


def main(
    seed: int = None,
    mode: str = None,
    log_dir: str = None,
    run_id: str = None,
    env_type: str = None,
    schedule_mode: str = None,
    episodes_per_phase: int = None,
    planning_horizon: int = None,
    planner_mode: str = None,
    planner_rollouts: int = None,
    agent_variant: str = None,
    lifelong_episodes_per_chapter: int = None,
    use_skills: bool = None,
    minigrid_scenarios: str = None,
    regime_aware_replay: bool = None,
    replay_frac_current: float = None,
    planning_coef: float = None,
    beta_conflict: float = None,
    beta_uncertainty: float = None,
    action_mask_internalization_coef: float = None,
    action_mask_dropout_prob: float = None,
    action_mask_prediction_coef: float = None,
    repo_online_bc_coef: float = None,
    repo_bc_pretrain_episodes: int = None,
    repo_bc_pretrain_max_steps: int = None,
    computer_scenarios: str = None,
    repo_scenarios: str = None,
    skill_mode: str = None,
    n_latent_skills: int = None,
    train_latent_skills: bool = None,
    resume_from: str = None,
    checkpoint_path: str = None,
    checkpoint_save_optim: bool = None,
    deterministic_torch: bool = None,
    force_cpu: bool = None,
    rl_steps: int = None,
    stage2_updates: int = None,
    stage4_updates: int = None,
    eval_policy: str = None,
):
    is_cli = len(sys.argv) > 1 and seed is None and mode is None

    if is_cli:
        parser = argparse.ArgumentParser(
            description=(
                "Proto-creature pipeline runner (Stage1-4 + lifecycle). "
                "Agent variants: full (self-model + reflection + planner), "
                "no_reflection (self-model + planner, traits fixed), "
                "no_self (planner only, no self-model). "
                "Env choices: gridworld (toy) or minigrid (benchmark)."
            )
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--mode",
            type=str,
            default="stage4",
            choices=["all", "stage1", "stage2", "stage3", "stage3b", "stage3c", "stage4", "lifelong", "lifelong_train"],
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--run-id",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--env-type",
            type=str,
            default="gridworld",
            choices=["gridworld", "minigrid", "tools", "instruction", "social", "computer", "repo", "mixed"],
            help=(
                "Environment family to use "
                "(gridworld=toy default, minigrid=MiniGrid backend, "
                "tools=arithmetic env, instruction=language-conditioned env, "
                "social=multi-agent env, computer=simulated coding tasks, "
                "repo=sandbox repo tasks (real pytest), "
                "mixed=mixture of supported env families)."
            ),
        )
        parser.add_argument(
            "--schedule-mode",
            type=str,
            default="iid",
            choices=["iid", "round_robin", "curriculum"],
        )
        parser.add_argument(
            "--episodes-per-phase",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--planning-horizon",
            type=int,
            default=12,
        )
        parser.add_argument(
            "--planner-mode",
            type=str,
            default="rollout",
            choices=["none", "repeat", "rollout", "beam", "skills"],
        )
        parser.add_argument(
            "--planner-rollouts",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--agent-variant",
            type=str,
            default="full",
            choices=["full", "no_reflection", "no_self"],
            help="full=self-model+reflection+planner | no_reflection=self-model+planner | no_self=planner only",
        )
        parser.add_argument(
            "--lifelong-episodes-per-chapter",
            type=int,
            default=50,
            help="Number of episodes per regime chapter when mode=lifelong.",
        )
        parser.add_argument(
            "--planning-coef",
            type=float,
            default=0.3,
            help="Weight of planner logits vs policy logits when mixing actions.",
        )
        parser.add_argument(
            "--beta-conflict",
            type=float,
            default=0.05,
            help="Regularizer weight for conflict penalty in A2C loss.",
        )
        parser.add_argument(
            "--beta-uncertainty",
            type=float,
            default=0.05,
            help="Regularizer weight for uncertainty penalty in A2C loss.",
        )
        parser.add_argument(
            "--invalid-action-coef",
            type=float,
            default=0.10,
            help=(
                "Penalty weight for probability mass on invalid actions when the environment exposes an action mask "
                "(helps the policy 'internalize' tool UI constraints)."
            ),
        )
        parser.add_argument(
            "--action-mask-dropout",
            type=float,
            default=0.0,
            help=(
                "With probability p, sample actions from the *unmasked* policy even when an action mask is available "
                "(while still computing the mask penalty). This reduces reliance on the UI mask and improves unmasked eval."
            ),
        )
        parser.add_argument(
            "--action-mask-pred-coef",
            type=float,
            default=0.10,
            help=(
                "Auxiliary BCE weight for predicting action-mask validity from policy features. "
                "Helps internalize tool UI constraints."
            ),
        )
        parser.add_argument(
            "--repo-online-bc-coef",
            type=float,
            default=0.10,
            help="Auxiliary online BC loss weight on RepoToolEnv using expert actions during A2C rollouts.",
        )
        parser.add_argument(
            "--repo-bc-episodes",
            type=int,
            default=0,
            help="Episodes of expert behavior-cloning pretrain on RepoToolEnv before stage4 (0 disables).",
        )
        parser.add_argument(
            "--repo-bc-max-steps",
            type=int,
            default=24,
            help="Max steps per episode during RepoToolEnv BC pretrain.",
        )
        parser.add_argument(
            "--use-skills",
            action="store_true",
            help="Enable hierarchical skills and high-level policy.",
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
            "--train-latent-skills",
            dest="train_latent_skills",
            action="store_true",
            default=None,
            help="Run supervised distillation stage for latent skills (Stage 2.5).",
        )
        parser.add_argument(
            "--no-train-latent-skills",
            dest="train_latent_skills",
            action="store_false",
            help="Disable latent skill distillation even when latent skills are present.",
        )
        parser.add_argument(
            "--resume-from",
            type=str,
            default=None,
            help="Path to a checkpoint (.pt) to load before running the experiment.",
        )
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help="Path to save a checkpoint (.pt) after the experiment finishes.",
        )
        parser.add_argument(
            "--checkpoint-save-optim",
            action="store_true",
            help="Also save/load optimizer state in checkpoints (larger files).",
        )
        parser.add_argument(
            "--deterministic-torch",
            action="store_true",
            help="Enable deterministic PyTorch execution when possible (better reproducibility, potentially slower).",
        )
        parser.add_argument(
            "--force-cpu",
            action="store_true",
            help="Force CPU execution even when CUDA is available (useful for stability/debugging).",
        )
        parser.add_argument(
            "--minigrid-scenarios",
            type=str,
            default=None,
            help=(
                "Comma-separated MiniGrid scenario aliases "
                "(e.g., 'minigrid-empty,minigrid-doorkey,test:minigrid-lavacrossing'). "
                "Only used when env-type is minigrid or mixed."
            ),
        )
        parser.add_argument(
            "--computer-scenarios",
            type=str,
            default=None,
            help="Comma-separated ComputerEnv scenario aliases (e.g., 'simple_project,refactor_project').",
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
            "--replay-frac-current",
            type=float,
            default=0.5,
            help="When --regime-aware-replay is enabled, fraction of samples drawn from the current regime.",
        )
        parser.add_argument(
            "--rl-steps",
            type=int,
            default=1024,
            help="Number of on-policy steps per A2C update in stage2/stage4.",
        )
        parser.add_argument(
            "--stage2-updates",
            type=int,
            default=1,
            help="How many A2C updates to run in stage2 (policy without self-model).",
        )
        parser.add_argument(
            "--stage4-updates",
            type=int,
            default=1,
            help="How many A2C updates to run in stage4 (policy with self-model/planner).",
        )
        parser.add_argument(
            "--eval-policy",
            type=str,
            default="sample",
            choices=["sample", "greedy"],
            help="Evaluation action selection: sample (stochastic) or greedy (argmax).",
        )
        args = parser.parse_args()

        seed = args.seed
        mode = args.mode
        log_dir = args.log_dir
        run_id = args.run_id
        env_type = args.env_type
        schedule_mode = args.schedule_mode
        episodes_per_phase = args.episodes_per_phase
        planning_horizon = args.planning_horizon
        planner_mode = args.planner_mode
        planner_rollouts = args.planner_rollouts
        planning_coef = args.planning_coef
        beta_conflict = args.beta_conflict
        beta_uncertainty = args.beta_uncertainty
        action_mask_internalization_coef = args.invalid_action_coef
        action_mask_dropout_prob = args.action_mask_dropout
        action_mask_prediction_coef = args.action_mask_pred_coef
        repo_online_bc_coef = args.repo_online_bc_coef
        repo_bc_pretrain_episodes = args.repo_bc_episodes
        repo_bc_pretrain_max_steps = args.repo_bc_max_steps
        agent_variant = args.agent_variant
        lifelong_episodes_per_chapter = args.lifelong_episodes_per_chapter
        use_skills = args.use_skills
        minigrid_scenarios = args.minigrid_scenarios
        computer_scenarios = args.computer_scenarios
        repo_scenarios = args.repo_scenarios
        regime_aware_replay = args.regime_aware_replay
        replay_frac_current = args.replay_frac_current
        skill_mode = args.skill_mode
        n_latent_skills = args.n_latent_skills
        train_latent_skills = args.train_latent_skills
        resume_from = args.resume_from
        checkpoint_path = args.checkpoint_path
        checkpoint_save_optim = args.checkpoint_save_optim
        deterministic_torch = args.deterministic_torch
        force_cpu = args.force_cpu
        rl_steps = args.rl_steps
        stage2_updates = args.stage2_updates
        stage4_updates = args.stage4_updates
        eval_policy = args.eval_policy
    else:
        if seed is None:
            seed = 0
        if mode is None:
            mode = "all"
        if schedule_mode is None:
            schedule_mode = "iid"
        if env_type is None:
            env_type = "gridworld"
        if episodes_per_phase is None:
            episodes_per_phase = 50
        if planning_horizon is None:
            planning_horizon = 12
        if planner_mode is None:
            planner_mode = "rollout"
        if planner_rollouts is None:
            planner_rollouts = 4
        if planning_coef is None:
            planning_coef = 0.3
        if beta_conflict is None:
            beta_conflict = 0.05
        if beta_uncertainty is None:
            beta_uncertainty = 0.05
        if action_mask_internalization_coef is None:
            action_mask_internalization_coef = 0.10
        if action_mask_dropout_prob is None:
            action_mask_dropout_prob = 0.0
        if action_mask_prediction_coef is None:
            action_mask_prediction_coef = 0.10
        if repo_online_bc_coef is None:
            repo_online_bc_coef = 0.10
        if repo_bc_pretrain_episodes is None:
            repo_bc_pretrain_episodes = 0
        if repo_bc_pretrain_max_steps is None:
            repo_bc_pretrain_max_steps = 24
        if agent_variant is None:
            agent_variant = "full"
        if lifelong_episodes_per_chapter is None:
            lifelong_episodes_per_chapter = 50
        if use_skills is None:
            use_skills = False
        if minigrid_scenarios is None:
            minigrid_scenarios = None
        if computer_scenarios is None:
            computer_scenarios = None
        if repo_scenarios is None:
            repo_scenarios = None
        if regime_aware_replay is None:
            regime_aware_replay = False
        if replay_frac_current is None:
            replay_frac_current = 0.5
        if skill_mode is None:
            skill_mode = "handcrafted"
        if n_latent_skills is None:
            n_latent_skills = 0
        # keep train_latent_skills as None to allow auto-enable when latent skills are present
        if resume_from is None:
            resume_from = None
        if checkpoint_path is None:
            checkpoint_path = None
        if checkpoint_save_optim is None:
            checkpoint_save_optim = False
        if deterministic_torch is None:
            deterministic_torch = False
        if force_cpu is None:
            force_cpu = False
        if rl_steps is None:
            rl_steps = 1024
        if stage2_updates is None:
            stage2_updates = 1
        if stage4_updates is None:
            stage4_updates = 1
        if eval_policy is None:
            eval_policy = "sample"

    scenario_list = None
    if minigrid_scenarios:
        scenario_list = [s.strip() for s in minigrid_scenarios.split(",") if s.strip()]
    computer_list = None
    if computer_scenarios:
        computer_list = [s.strip() for s in computer_scenarios.split(",") if s.strip()]
    repo_list = None
    if repo_scenarios:
        repo_list = [s.strip() for s in repo_scenarios.split(",") if s.strip()]

    result = run_experiment(
        seed=seed,
        mode=mode,
        agent_variant=agent_variant,
        env_type=env_type,
        schedule_mode=schedule_mode,
        episodes_per_phase=episodes_per_phase,
        minigrid_scenarios=scenario_list,
        computer_scenarios=computer_list,
        repo_scenarios=repo_list,
        regime_aware_replay=regime_aware_replay,
        replay_frac_current=replay_frac_current,
        n_steps=int(rl_steps),
        stage2_updates=int(stage2_updates),
        stage4_updates=int(stage4_updates),
        eval_policy=eval_policy,
        skill_mode=skill_mode,
        n_latent_skills=n_latent_skills,
        train_latent_skills=train_latent_skills,
        resume_from=resume_from,
        checkpoint_path=checkpoint_path,
        checkpoint_save_optim=checkpoint_save_optim,
        deterministic_torch=bool(deterministic_torch),
        force_cpu=bool(force_cpu),
        planning_horizon=planning_horizon,
        planner_mode=planner_mode,
        planner_rollouts=planner_rollouts,
        planning_coef=planning_coef,
        beta_conflict=beta_conflict,
        beta_uncertainty=beta_uncertainty,
        action_mask_internalization_coef=float(action_mask_internalization_coef),
        action_mask_dropout_prob=float(action_mask_dropout_prob),
        action_mask_prediction_coef=float(action_mask_prediction_coef),
        repo_online_bc_coef=float(repo_online_bc_coef),
        repo_bc_pretrain_episodes=int(repo_bc_pretrain_episodes),
        repo_bc_pretrain_max_steps=int(repo_bc_pretrain_max_steps),
        log_dir=log_dir,
        run_id=run_id,
        lifelong_episodes_per_chapter=lifelong_episodes_per_chapter,
        use_skills=use_skills,
    )

    if mode == "lifelong":
        ll = (result or {}).get("stage_metrics", {}).get("lifelong_eval", {})
        per_chapter = ll.get("lifelong_per_chapter", []) or []
        if per_chapter:
            print("\n[Lifelong] Per-chapter mean returns:")
            for ch in per_chapter:
                mean_r = ch.get("mean_return")
                std_r = ch.get("std_return")
                regime = ch.get("regime", "?")
                if isinstance(mean_r, (int, float)) and isinstance(std_r, (int, float)):
                    print(f"  {regime}: {mean_r:.3f} +/- {std_r:.3f}")
                else:
                    print(f"  {regime}: {mean_r} +/- {std_r}")
        gap = ll.get("lifelong_forgetting_R1_gap")
        if isinstance(gap, (int, float)):
            print(f"[Lifelong] Forgetting gap (R1_return - R1): {gap:.3f}")
    elif mode == "lifelong_train":
        ll = (result or {}).get("stage_metrics", {}).get("lifelong_train", {})
        per_chapter = ll.get("lifelong_per_chapter", []) or []
        if per_chapter:
            print("\n[Lifelong-Train] Per-chapter mean returns:")
            for ch in per_chapter:
                mean_r = ch.get("mean_return")
                std_r = ch.get("std_return")
                regime = ch.get("regime", "?")
                if isinstance(mean_r, (int, float)) and isinstance(std_r, (int, float)):
                    print(f"  {regime}: {mean_r:.3f} +/- {std_r:.3f}")
                else:
                    print(f"  {regime}: {mean_r} +/- {std_r}")
        gap = ll.get("lifelong_forgetting_R1_gap")
        if isinstance(gap, (int, float)):
            print(f"[Lifelong-Train] Forgetting gap (R1_return - R1): {gap:.3f}")

    # Save results if requested
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        run_id_effective = run_id or f"s{seed}_{mode}_{int(time.time())}"
        out_path = log_dir_path / f"run_{run_id_effective}_result.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"[LOG] Saved run_experiment result to {out_path}")

    return result


if __name__ == "__main__":
    main()
