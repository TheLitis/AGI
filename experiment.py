"""
Experiment orchestration for the 4-stage proto-creature pipeline.

Stages:
1) Random exploration + world-model pretraining
2) A2C without self-model
3) Self-model offline + (optional) self-reflection on traits
4) A2C with self-model + (optional) planner + lifecycle evaluation
"""

from typing import Any, Dict, List, Optional
import random
import numpy as np
import torch

from agent import ProtoCreatureAgent
from env import GridWorldEnv, EnvPool
from tool_env import ToolEnv, ToolEnvConfig
from computer_env import ComputerEnv, ComputerEnvConfig, build_computer_taskset
from models import traits_to_preference_weights
from trainer import Trainer, ExperimentLogger
from skills import get_default_skills
from checkpointing import load_checkpoint, save_checkpoint


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_env_pool(
    seed: int,
    schedule_mode: str,
    episodes_per_phase: int,
    max_steps_env: int = 50,
) -> EnvPool:
    train_envs: List[GridWorldEnv] = []
    test_envs: List[GridWorldEnv] = []

    for idx in range(5):
        train_envs.append(
            GridWorldEnv(
                max_steps=max_steps_env,
                seed=seed + idx,
                multi_task=True,
                schedule_mode=schedule_mode,
                episodes_per_phase=episodes_per_phase,
                env_id=idx,
                env_name=f"train_{idx}",
            )
        )

    for idx in range(1):
        eid = len(train_envs) + idx
        test_envs.append(
            GridWorldEnv(
                max_steps=max_steps_env,
                seed=seed + eid,
                multi_task=True,
                schedule_mode=schedule_mode,
                episodes_per_phase=episodes_per_phase,
                env_id=eid,
                env_name=f"test_{idx}",
            )
        )

    envs = train_envs + test_envs
    train_ids = [e.env_id for e in train_envs]
    test_ids = [e.env_id for e in test_envs]

    return EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )


def _build_minigrid_env_pool(
    seed: int,
    schedule_mode: str,
    scenario_names: Optional[List[str]] = None,
) -> EnvPool:
    from minigrid_env import MiniGridEnvPool

    return MiniGridEnvPool(
        seed=seed,
        schedule_mode=schedule_mode,
        scenario_names=scenario_names,
    )


def _build_tool_env_pool(
    seed: int,
    schedule_mode: str,
) -> EnvPool:
    cfg = ToolEnvConfig()
    train_envs = [
        ToolEnv(config=cfg, env_id=0, env_name="tools_basic_train", seed=seed + 0),
    ]
    test_envs = [
        ToolEnv(config=cfg, env_id=1, env_name="tools_basic_test", seed=seed + 1),
    ]
    envs = train_envs + test_envs
    train_ids = [e.env_id for e in train_envs]
    test_ids = [e.env_id for e in test_envs]
    pool = EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )
    return pool


def _build_computer_env_pool(
    seed: int,
    schedule_mode: str,
    scenario_names: Optional[List[str]] = None,
) -> EnvPool:
    cfg = ComputerEnvConfig(rng_seed=seed)
    train_taskset = build_computer_taskset(scenario_names)
    test_taskset = build_computer_taskset(scenario_names, difficulty_shift=0.1)
    train_envs = [
        ComputerEnv(task_set=train_taskset, config=cfg, env_id=0, env_name="computer_train", seed=seed + 0),
    ]
    test_envs = [
        ComputerEnv(task_set=test_taskset, config=cfg, env_id=1, env_name="computer_test", seed=seed + 1),
    ]
    envs = train_envs + test_envs
    train_ids = [e.env_id for e in train_envs]
    test_ids = [e.env_id for e in test_envs]
    return EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )


def _build_instruction_env_pool(
    seed: int,
    schedule_mode: str,
) -> EnvPool:
    from instruction_env import InstructionEnv, InstructionEnvConfig

    cfg = InstructionEnvConfig()
    train_env = InstructionEnv(config=cfg, env_id=0, env_name="instruction_train", seed=seed + 0)
    test_env = InstructionEnv(config=cfg, env_id=1, env_name="instruction_test", seed=seed + 1)

    # Make test instructions slightly different (OOD-ish phrasing).
    if getattr(test_env, "scenario_configs", None):
        for conf in test_env.scenario_configs:
            if isinstance(conf, dict) and "description" in conf:
                conf["description"] = str(conf["description"]).replace("go to", "navigate to").replace("TAKE", "pick")

    envs = [train_env, test_env]
    train_ids = [train_env.env_id]
    test_ids = [test_env.env_id]
    pool = EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )
    pool.task_metadata_instruction = [dict(x) for x in (train_env.scenario_configs + test_env.scenario_configs)]  # type: ignore[attr-defined]
    return pool


def _build_social_env_pool(
    seed: int,
    schedule_mode: str,
) -> EnvPool:
    from social_env import SocialEnv, SocialEnvConfig

    cfg = SocialEnvConfig()
    train_env = SocialEnv(config=cfg, env_id=0, env_name="social_train", seed=seed + 0)
    test_env = SocialEnv(config=cfg, env_id=1, env_name="social_test", seed=seed + 1)

    envs = [train_env, test_env]
    train_ids = [train_env.env_id]
    test_ids = [test_env.env_id]
    pool = EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )
    pool.task_metadata_social = [dict(x) for x in (train_env.scenario_configs + test_env.scenario_configs)]  # type: ignore[attr-defined]
    return pool


def _build_repo_env_pool(
    seed: int,
    schedule_mode: str,
    scenario_names: Optional[List[str]] = None,
) -> EnvPool:
    from repo_tool_env import RepoToolEnv, RepoToolEnvConfig, build_repo_taskset

    train_names: List[str] = []
    test_names: List[str] = []
    for raw in scenario_names or []:
        name = str(raw).strip()
        if not name:
            continue
        if name.startswith("test:"):
            test_names.append(name[len("test:") :])
        elif name.startswith("train:"):
            train_names.append(name[len("train:") :])
        else:
            train_names.append(name)

    if not train_names and not test_names:
        train_names = ["calc_add"]
        test_names = ["calc_div"]
    elif not test_names and len(train_names) > 1:
        test_names = [train_names[-1]]
        train_names = train_names[:-1]
    elif not test_names:
        test_names = list(train_names)

    cfg = RepoToolEnvConfig()
    train_tasks = build_repo_taskset(train_names)
    test_tasks = build_repo_taskset(test_names)

    train_env = RepoToolEnv(task_set=train_tasks, config=cfg, env_id=0, env_name="repo_train", seed=seed + 0)
    test_env = RepoToolEnv(task_set=test_tasks, config=cfg, env_id=1, env_name="repo_test", seed=seed + 1)

    envs = [train_env, test_env]
    train_ids = [train_env.env_id]
    test_ids = [test_env.env_id]
    pool = EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )
    # small metadata for analysis/logging
    pool.task_metadata_repo = [dict(x) for x in (train_env.scenario_configs + test_env.scenario_configs)]  # type: ignore[attr-defined]
    return pool


def _build_mixed_env_pool(
    seed: int,
    schedule_mode: str,
    episodes_per_phase: int,
    minigrid_scenarios: Optional[List[str]] = None,
    computer_scenarios: Optional[List[str]] = None,
    repo_scenarios: Optional[List[str]] = None,
    max_steps_env: int = 50,
) -> EnvPool:
    """Combine GridWorld, MiniGrid, (optional) Computer, and (optional) RepoTool envs into a single EnvPool."""
    grid_pool = _build_env_pool(
        seed=seed,
        schedule_mode=schedule_mode,
        episodes_per_phase=episodes_per_phase,
        max_steps_env=max_steps_env,
    )
    mg_pool = _build_minigrid_env_pool(
        seed=seed,
        schedule_mode=schedule_mode,
        scenario_names=minigrid_scenarios,
    )
    comp_pool = _build_computer_env_pool(
        seed=seed,
        schedule_mode=schedule_mode,
        scenario_names=computer_scenarios,
    )
    repo_pool = (
        _build_repo_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
            scenario_names=repo_scenarios,
        )
        if repo_scenarios
        else None
    )

    offset = len(grid_pool.envs)
    mg_envs = []
    for i, env in enumerate(mg_pool.envs):
        # rebase env_id to avoid collisions
        new_id = offset + i
        if hasattr(env, "_env_id"):
            env._env_id = new_id
        if hasattr(env, "_env_name"):
            env._env_name = f"mg_{getattr(env, '_env_name', env.env_name)}"
        mg_envs.append(env)

    offset_comp = offset + len(mg_envs)
    comp_envs = []
    for i, env in enumerate(comp_pool.envs):
        new_id = offset_comp + i
        if hasattr(env, "_env_id"):
            env._env_id = new_id
        if hasattr(env, "_env_name"):
            env._env_name = f"comp_{getattr(env, '_env_name', env.env_name)}"
        comp_envs.append(env)

    offset_repo = offset_comp + len(comp_envs)
    repo_envs = []
    if repo_pool is not None:
        for i, env in enumerate(repo_pool.envs):
            new_id = offset_repo + i
            if hasattr(env, "_env_id"):
                env._env_id = new_id
            if hasattr(env, "_env_name"):
                env._env_name = f"repo_{getattr(env, '_env_name', env.env_name)}"
            repo_envs.append(env)

    envs = grid_pool.envs + mg_envs + comp_envs + repo_envs
    grid_train = list(grid_pool.train_env_ids or [])
    grid_test = list(grid_pool.test_env_ids or [])
    mg_train = [offset + i for i in (mg_pool.train_env_ids or [])]
    mg_test = [offset + i for i in (mg_pool.test_env_ids or [])]
    comp_train = [offset_comp + i for i in (comp_pool.train_env_ids or [])]
    comp_test = [offset_comp + i for i in (comp_pool.test_env_ids or [])]
    repo_train = [offset_repo + i for i in (repo_pool.train_env_ids or [])] if repo_pool is not None else []
    repo_test = [offset_repo + i for i in (repo_pool.test_env_ids or [])] if repo_pool is not None else []

    train_ids = grid_train + mg_train + comp_train + repo_train
    test_ids = grid_test + mg_test + comp_test + repo_test

    env_pool = EnvPool(
        envs=envs,
        schedule_mode=schedule_mode,
        seed=seed,
        train_env_ids=train_ids,
        test_env_ids=test_ids,
    )
    # preserve MiniGrid metadata for logging/analysis
    if hasattr(mg_pool, "task_metadata"):
        env_pool.task_metadata = getattr(mg_pool, "task_metadata")  # type: ignore[attr-defined]
    if hasattr(comp_pool, "task_metadata"):
        env_pool.task_metadata_computer = getattr(comp_pool, "task_metadata")  # type: ignore[attr-defined]
    if repo_pool is not None and hasattr(repo_pool, "task_metadata_repo"):
        env_pool.task_metadata_repo = getattr(repo_pool, "task_metadata_repo")  # type: ignore[attr-defined]
    return env_pool


def _sanitize(obj: Any) -> Any:
    """Recursively convert to JSON-safe types (replace NaN/Inf with None)."""
    import math

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (float, int)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    return obj


def run_experiment(
    seed: int = 0,
    mode: str = "all",
    agent_variant: str = "full",
    env_type: str = "gridworld",
    schedule_mode: str = "iid",
    episodes_per_phase: int = 50,
    minigrid_scenarios: Optional[List[str]] = None,
    computer_scenarios: Optional[List[str]] = None,
    repo_scenarios: Optional[List[str]] = None,
    regime_aware_replay: bool = False,
    replay_frac_current: float = 0.5,
    planning_horizon: int = 12,
    planner_mode: str = "rollout",
    planner_rollouts: int = 4,
    log_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    n_steps: int = 1024,
    gamma: float = 0.99,
    entropy_coef: float = 0.003,
    curiosity_beta: float = 0.1,
    beta_conflict: float = 0.05,
    beta_uncertainty: float = 0.05,
    planning_coef: float = 0.3,
    lifelong_episodes_per_chapter: int = 50,
    use_skills: bool = False,
    skill_mode: str = "handcrafted",
    n_latent_skills: int = 0,
    train_latent_skills: Optional[bool] = None,
    resume_from: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_save_optim: bool = True,
) -> Dict[str, Any]:
    """
    Run the staged training pipeline and return metrics.

    Args:
        seed: global seed for envs, numpy, torch.
        mode: which stage(s) to run (all|stage1|stage2|stage3|stage3b|stage3c|stage4).
        agent_variant: full | no_reflection | no_self.
        env_type: gridworld | minigrid | tools | mixed (gridworld + minigrid).
        minigrid_scenarios: optional list of scenario aliases (e.g., ["minigrid-empty", "minigrid-doorkey", "test:minigrid-lavacrossing"]).
        computer_scenarios: optional list of ComputerEnv scenario aliases (e.g., ["simple_project", "refactor_project"]).
        schedule_mode: env scheduling across scenarios ("iid", "round_robin", "curriculum").
        episodes_per_phase: curriculum phase length for scenario scheduling.
        planning_horizon / planner_mode / planner_rollouts: planner settings.
        log_dir: optional directory to save per-run JSONL logs (ExperimentLogger).
        run_id: optional run identifier for logs.
        n_steps, gamma, entropy_coef, curiosity_beta, beta_conflict, beta_uncertainty,
        planning_coef: RL hyperparameters (kept identical to original code).
        train_latent_skills: whether to run Stage 2.5 latent skill distillation (auto-enables for latent/mixed skills if None).

    Returns:
        Dict with "config" and "stage_metrics" plus metadata; all values are JSON-safe.
    """
    _set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "[Config] "
        f"planner_mode={planner_mode}, planner_rollouts={planner_rollouts}, "
        f"planning_coef={planning_coef:.3f}, "
        f"beta_conflict={beta_conflict:.3f}, beta_uncertainty={beta_uncertainty:.3f}"
    )

    env_choice = (env_type or "gridworld").lower()
    if env_choice == "gridworld" or env_choice == "toy":
        env_pool = _build_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
            episodes_per_phase=episodes_per_phase,
            max_steps_env=50,
        )
    elif env_choice == "tools":
        env_pool = _build_tool_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
        )
    elif env_choice == "instruction":
        env_pool = _build_instruction_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
        )
    elif env_choice == "social":
        env_pool = _build_social_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
        )
    elif env_choice == "minigrid":
        env_pool = _build_minigrid_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
            scenario_names=minigrid_scenarios,
        )
    elif env_choice == "computer":
        env_pool = _build_computer_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
            scenario_names=computer_scenarios,
        )
    elif env_choice == "repo":
        env_pool = _build_repo_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
            scenario_names=repo_scenarios,
        )
    elif env_choice == "mixed":
        env_pool = _build_mixed_env_pool(
            seed=seed,
            schedule_mode=schedule_mode,
            episodes_per_phase=episodes_per_phase,
            minigrid_scenarios=minigrid_scenarios,
            computer_scenarios=computer_scenarios,
            repo_scenarios=repo_scenarios,
            max_steps_env=50,
        )
    else:
        raise ValueError(
            f"Unknown env_type '{env_type}'; expected 'gridworld', 'minigrid', 'tools', 'instruction', 'social', 'computer', 'repo', or 'mixed'."
        )

    train_latent_flag = train_latent_skills
    if train_latent_flag is None:
        train_latent_flag = bool(use_skills and skill_mode in {"latent", "mixed"} and n_latent_skills > 0)

    env_desc_np = [env.get_env_descriptor() for env in env_pool.envs]
    max_len = max(desc.shape[0] for desc in env_desc_np)
    env_desc_np = [np.pad(desc, (0, max_len - desc.shape[0]), constant_values=0.0) for desc in env_desc_np]
    env_descriptors = torch.tensor(np.stack(env_desc_np), dtype=torch.float32, device=device)

    skills_list = get_default_skills() if use_skills else []

    agent = ProtoCreatureAgent(
        n_cell_types=env_pool.n_cell_types,
        n_scenarios=env_pool.n_scenarios,
        env_descriptors=env_descriptors,
        device=device,
        n_actions=env_pool.n_actions,
        use_skills=use_skills,
        n_skills=len(skills_list),
        skill_mode=skill_mode,
        n_latent_skills=n_latent_skills,
    )

    logger = ExperimentLogger(run_id=run_id, logdir=log_dir) if log_dir else None
    trainer = Trainer(
        env=env_pool,
        agent=agent,
        planning_horizon=planning_horizon,
        planner_gamma=gamma,
        planner_mode=planner_mode,
        planner_rollouts=planner_rollouts,
        train_env_ids=env_pool.train_env_ids,
        test_env_ids=env_pool.test_env_ids,
        env_descriptors=env_descriptors,
        logger=logger,
        use_skills=use_skills,
        skills=skills_list,
        regime_aware_replay=regime_aware_replay,
        replay_frac_current=replay_frac_current,
        skill_mode=skill_mode,
        n_latent_skills=n_latent_skills,
    )

    if resume_from:
        load_checkpoint(
            resume_from,
            agent=agent,
            trainer=trainer,
            load_optim=bool(checkpoint_save_optim),
            strict=False,
        )

    mode = (mode or "all").lower()
    stage_metrics: Dict[str, Any] = {}
    stage_metrics["trait_reflection_debug"] = trainer.trait_reflection_debug
    use_self_flag = agent_variant != "no_self"
    def _attach_trait_logs(limit: int = 200) -> None:
        summary = trainer.get_trait_reflection_summary()
        if summary:
            stage_metrics["trait_reflection_summary"] = summary
        log_entries: List[Dict[str, Any]] = []
        if getattr(trainer, "trait_reflection_log", None) is not None:
            log_entries = trainer.get_trait_reflection_log(limit=limit)
        if log_entries:
            stage_metrics["trait_reflection_log"] = log_entries

    # Initial evaluation
    stage_metrics["eval_before_rl"] = trainer.evaluate(
        n_episodes=10,
        max_steps=200,
        use_self=False,
        planning_coef=0.0,
    )

    # Stage 1: random exploration + world model
    if mode in {"all", "stage1", "stage2", "stage3", "stage3b", "stage3c", "stage4", "lifelong", "lifelong_train"}:
        trainer.collect_random_experience(n_steps=5000)
        trainer.train_world_model()
        if mode == "stage1":
            _attach_trait_logs()
            result = {
                "seed": seed,
                "mode": mode,
                "env_type": env_choice,
                "device": str(device),
                "n_scenarios": env_pool.n_scenarios,
                "max_steps_env": env_pool.max_steps,
                "stage_metrics": _sanitize(stage_metrics),
            }
            if logger:
                logger.close()
            return result

    # Stage 2: policy without self
    if mode in {"all", "stage2", "stage3", "stage3b", "stage3c", "stage4", "lifelong", "lifelong_train"}:
        trainer.train_policy_a2c(
            n_steps=n_steps,
            gamma=gamma,
            entropy_coef=entropy_coef,
            use_self=False,
            curiosity_beta=curiosity_beta,
            beta_conflict=beta_conflict,
            beta_uncertainty=beta_uncertainty,
            planning_coef=0.0,
        )
        stage_metrics["eval_after_stage2"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=0.0,
        )
        if train_latent_flag and trainer.skill_library is not None and trainer.n_latent_skills > 0:
            trainer.train_latent_skills_supervised()
            stage_metrics["latent_skill_training"] = {
                "enabled": True,
                "n_latent_skills": trainer.n_latent_skills,
                "demos_per_skill": trainer.latent_skill_training.demos_per_skill,
                "epochs": trainer.latent_skill_training.epochs,
            }
        if mode == "stage2":
            _attach_trait_logs()
            result = {
                "seed": seed,
                "mode": mode,
                "env_type": env_choice,
                "device": str(device),
                "n_scenarios": env_pool.n_scenarios,
                "max_steps_env": env_pool.max_steps,
                "stage_metrics": _sanitize(stage_metrics),
            }
            if logger:
                logger.close()
            return result

    # Stage 3: self-model offline
    if agent_variant != "no_self" and mode in {"all", "stage3", "stage3b", "stage3c", "stage4", "lifelong", "lifelong_train"}:
        trainer.train_self_model_offline()
        stage_metrics["self_model_probe_after_stage3"] = trainer.probe_self_model(gamma=gamma)
        stage_metrics["eval_after_stage3_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=0.0,
        )
        stage_metrics["eval_after_stage3_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=True,
            planning_coef=0.0,
        )
        if mode == "stage3":
            _attach_trait_logs()
            result = {
                "seed": seed,
                "mode": mode,
                "env_type": env_choice,
                "device": str(device),
                "n_scenarios": env_pool.n_scenarios,
                "max_steps_env": env_pool.max_steps,
                "stage_metrics": _sanitize(stage_metrics),
            }
            if logger:
                logger.close()
            return result

    # Stage 3b/c: self-reflection on traits
    if agent_variant == "full" and mode in {"all", "stage3b", "stage3c", "stage4", "lifelong", "lifelong_train"}:
        trainer.self_reflect_on_traits()
        stage_metrics["eval_after_stage3c_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=0.0,
        )
        stage_metrics["eval_after_stage3c_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=True,
            planning_coef=0.0,
        )
        if mode == "stage3b":
            _attach_trait_logs()
            result = {
                "seed": seed,
                "mode": mode,
                "env_type": env_choice,
                "device": str(device),
                "n_scenarios": env_pool.n_scenarios,
                "max_steps_env": env_pool.max_steps,
                "stage_metrics": _sanitize(stage_metrics),
            }
            if logger:
                logger.close()
            return result

    # Stage 3c: self-model <-> traits co-learning
    if agent_variant != "no_self" and mode in {"all", "stage3c", "stage4", "lifelong", "lifelong_train"}:
        co_learn = trainer.run_stage3c_self_model_trait_co_learning(
            n_collect_episodes=10,
            max_steps=200,
            use_self_for_collection=True,
            planning_coef=0.0,
            split="train",
            lr_scale=0.5,
            probe_gamma=gamma,
        )
        if isinstance(co_learn, dict):
            stage_metrics["self_model_probe_after_stage3c"] = co_learn.get("probe_after")
            stage_metrics["stage3c_self_model_trait_co_learning"] = {
                k: v
                for k, v in co_learn.items()
                if k not in {"probe_after"}
            }
        stage_metrics["eval_after_stage3c_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=0.0,
        )
        stage_metrics["eval_after_stage3c_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=True,
            planning_coef=0.0,
        )
        if mode in {"stage3c"}:
            _attach_trait_logs()
            result = {
                "seed": seed,
                "mode": mode,
                "env_type": env_choice,
                "device": str(device),
                "n_scenarios": env_pool.n_scenarios,
                "max_steps_env": env_pool.max_steps,
                "stage_metrics": _sanitize(stage_metrics),
            }
            if logger:
                logger.close()
            return result

    # Stage 4: policy with self + planner (optional)
    if mode in {"all", "stage4", "lifelong", "lifelong_train"}:
        planning_coef_eff = planning_coef if use_self_flag else 0.0

        trainer.train_policy_a2c(
            n_steps=n_steps,
            gamma=gamma,
            entropy_coef=entropy_coef,
            use_self=use_self_flag,
            curiosity_beta=curiosity_beta,
            beta_conflict=beta_conflict,
            beta_uncertainty=beta_uncertainty,
            planning_coef=planning_coef_eff,
        )
        stage_metrics["eval_after_stage4_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=True,
            planning_coef=planning_coef_eff,
        )
        stage_metrics["eval_after_stage4_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=planning_coef_eff,
        )
        stage_metrics["self_model_probe_after_stage4"] = trainer.probe_self_model(gamma=gamma)

        # Lifecycle phases A/B/C (train/test splits)
        env_pool.set_phase("A")
        stage_metrics["lifecycle_phaseA_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=planning_coef_eff,
        )

        env_pool.set_phase("B")
        stage_metrics["lifecycle_phaseB_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=planning_coef_eff,
        )
        stage_metrics["lifecycle_phaseB_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=True,
            planning_coef=planning_coef_eff,
        )

        env_pool.set_phase("C")
        stage_metrics["lifecycle_phaseC_no_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=False,
            planning_coef=planning_coef_eff,
        )
        stage_metrics["lifecycle_phaseC_self"] = trainer.evaluate(
            n_episodes=20,
            max_steps=200,
            use_self=True,
            planning_coef=planning_coef_eff,
        )
        # Online Phase C adaptation (Phase D)
        reflect_enabled = agent_variant == "full"
        stage_metrics["lifecycle_phaseC_online_no_self"] = trainer.run_online_phaseC_adaptation(
            env=env_pool,
            n_episodes=50,
            use_self=False,
            planning_coef=planning_coef_eff,
            phase_label="phaseC_online_no_self",
            allow_reflection=False,
        )
        stage_metrics["lifecycle_phaseC_online_self"] = trainer.run_online_phaseC_adaptation(
            env=env_pool,
            n_episodes=50,
            use_self=True,
            planning_coef=planning_coef_eff,
            phase_label="phaseC_online_self",
            allow_reflection=reflect_enabled,
        )

        # reset to default scheduling
        env_pool.set_phase("")

    if mode == "lifelong":
        stage_metrics["lifelong_eval"] = trainer.run_lifelong_eval(
            episodes_per_chapter=lifelong_episodes_per_chapter,
            planning_coef=planning_coef_eff if use_self_flag else 0.0,
            agent_variant=agent_variant,
            allow_online_reflection=agent_variant == "full",
        )
    elif mode == "lifelong_train":
        stage_metrics["lifelong_train"] = trainer.run_lifelong_train(
            episodes_per_chapter=lifelong_episodes_per_chapter,
            planning_coef=planning_coef_eff if use_self_flag else 0.0,
            agent_variant=agent_variant,
            allow_online_reflection=agent_variant == "full",
            allow_online_model_updates=True,
            lr_policy=3e-4,
            lr_models=1e-4,
            replay_ratio_old=0.5,
            regularization_coef=1e-3,
        )

    if env_choice in {"minigrid", "mixed"} and hasattr(env_pool, "task_metadata"):
        stage_metrics["minigrid_task_metadata"] = env_pool.task_metadata
    if env_choice in {"computer", "mixed"} and hasattr(env_pool, "task_metadata_computer"):
        stage_metrics["computer_task_metadata"] = getattr(env_pool, "task_metadata_computer")

    _attach_trait_logs(limit=200)
    stage_metrics["skill_mode"] = skill_mode
    if getattr(trainer, "skill_usage_counts", None):
        stage_metrics["skill_usage_counts"] = dict(trainer.skill_usage_counts)

    result = {
        "seed": seed,
        "mode": mode,
        "agent_variant": agent_variant,
        "device": str(device),
        "n_scenarios": env_pool.n_scenarios,
        "n_envs": getattr(env_pool, "n_envs", 0),
        "schedule_mode": schedule_mode,
        "episodes_per_phase": episodes_per_phase,
        "planning_horizon": planning_horizon,
        "planner_mode": planner_mode,
        "planner_rollouts": planner_rollouts,
        "meta_conflict_ma": trainer.meta_conflict_ma,
        "meta_uncertainty_ma": trainer.meta_uncertainty_ma,
        "max_steps_env": env_pool.max_steps,
        "final_traits": agent.traits.detach().cpu().numpy().tolist(),
        "final_preference_weights": traits_to_preference_weights(agent.traits)
        .detach()
        .cpu()
        .numpy()
        .tolist(),
        "stage_metrics": stage_metrics,
        "env_type": env_choice,
        "minigrid_scenarios": minigrid_scenarios,
        "computer_scenarios": computer_scenarios,
        "repo_scenarios": repo_scenarios,
        "config": {
            "seed": seed,
            "device": str(device),
            "n_scenarios": env_pool.n_scenarios,
            "n_envs": getattr(env_pool, "n_envs", 0),
            "schedule_mode": schedule_mode,
            "episodes_per_phase": episodes_per_phase,
            "planning_horizon": planning_horizon,
            "planner_mode": planner_mode,
            "planner_rollouts": planner_rollouts,
            "n_steps": n_steps,
            "gamma": gamma,
            "entropy_coef": entropy_coef,
            "curiosity_beta": curiosity_beta,
            "beta_conflict": beta_conflict,
            "beta_uncertainty": beta_uncertainty,
            "planning_coef": planning_coef,
            "use_self": use_self_flag if mode in {"all", "stage4"} else agent_variant != "no_self",
            "use_planner": planning_coef > 0.0,
            "do_self_reflection": agent_variant == "full",
            "mode": mode,
            "agent_variant": agent_variant,
            "logdir": log_dir,
            "exp_name": run_id,
            "env_type": env_choice,
            "minigrid_scenarios": minigrid_scenarios,
            "computer_scenarios": computer_scenarios,
            "repo_scenarios": repo_scenarios,
            "regime_aware_replay": regime_aware_replay,
            "skill_mode": skill_mode,
            "n_latent_skills": n_latent_skills,
            "train_latent_skills": bool(train_latent_flag),
        },
    }

    if logger:
        logger.close()

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            agent=agent,
            trainer=trainer,
            save_optim=bool(checkpoint_save_optim),
            extra={
                "seed": seed,
                "mode": mode,
                "agent_variant": agent_variant,
                "env_type": env_choice,
                "run_id": run_id,
            },
        )

    return _sanitize(result)
