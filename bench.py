"""
AGI-Bench runner: standardized suites, JSON report, and a skeleton AGI score.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment import run_experiment

SCHEMA_VERSION = "0.1"
SUITE_ORDER = ["core", "tools", "tools_open", "language", "social", "lifelong", "safety"]


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


def _parse_seeds(values: Optional[List[str]]) -> List[int]:
    if not values:
        return [0, 1]
    raw: List[str] = []
    for item in values:
        raw.extend([p for p in str(item).split(",") if p.strip()])
    seeds: List[int] = []
    for tok in raw:
        try:
            seeds.append(int(tok))
        except ValueError:
            continue
    return seeds or [0, 1]


def _safe_mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _geometric_mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and v >= 0.0]
    if not clean:
        return None
    if any(v <= 0.0 for v in clean):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in clean) / len(clean)))


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "unknown"


def _atomic_write_report(path: Path, payload: Dict[str, Any]) -> None:
    partial_path = path.with_suffix(path.suffix + ".partial")
    partial_path.write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    partial_path.replace(path)


def _refresh_overall(report: Dict[str, Any]) -> None:
    suites = report.get("suites", [])
    scores_used = [s.get("score") for s in suites if s.get("score") is not None and s.get("status") == "ok"]
    agi_score = _geometric_mean([float(x) for x in scores_used if x is not None])
    notes: List[str] = []
    if agi_score is None:
        agi_score = 0.0
        notes.append("no_suite_scores_available")
    report["overall"] = {
        "agi_score": agi_score,
        "notes": notes,
        "gates": {
            "gate0": "pass" if suites else "fail",
            "gate1": "na",
            "gate2": "na",
        },
    }


def _save_report(report_path: Path, report: Dict[str, Any]) -> None:
    _refresh_overall(report)
    _atomic_write_report(report_path, report)


@dataclass(frozen=True)
class BenchCase:
    name: str
    env_type: str
    minigrid_scenarios: Optional[List[str]] = None
    computer_scenarios: Optional[List[str]] = None
    repo_scenarios: Optional[List[str]] = None
    description: str = ""


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    cases: List[BenchCase]
    implemented: bool = True
    description: str = ""


SUITE_METRICS_KEYS: Dict[str, List[str]] = {
    "core": [
        "mean_return",
        "test_mean_return",
        "ood_gap",
    ],
    "tools": [
        "pass_rate_masked",
        "pass_rate_unmasked",
        "mean_steps_to_pass_unmasked",
        "invalid_action_rate",
        "mask_pred_f1",
        "mask_pred_auc",
        "bc_pretrain_used",
        "action_mask_dropout_prob",
        "mean_invalid_mass",
        "ood_gap",
    ],
    "tools_open": [
        "pass_rate_unmasked",
        "mean_steps_to_pass_unmasked",
        "invalid_action_rate",
        "ood_gap",
    ],
    "language": [
        "pass_rate",
        "ood_pass_rate",
        "causal_drop",
    ],
    "social": [
        "success_rate",
        "transfer_rate",
    ],
    "lifelong": [
        "forgetting_gap",
        "forward_transfer",
    ],
    "safety": [
        "safety_planner_ok",
        "constraint_compliance",
        "catastrophic_fail_rate",
    ],
}


def _metric_template(name: str) -> Dict[str, Any]:
    keys = SUITE_METRICS_KEYS.get(name, [])
    return {k: None for k in keys}


def _case_label(case: BenchCase) -> str:
    if case.env_type == "repo":
        scenarios = ",".join(case.repo_scenarios or [])
        return f"repo_tool_env/{scenarios or case.name}"
    if case.env_type == "tools":
        return "tool_env/basic"
    if case.env_type == "minigrid":
        scenarios = ",".join(case.minigrid_scenarios or [])
        return f"minigrid/{scenarios or case.name}"
    if case.env_type == "gridworld":
        return "gridworld/basic"
    return f"{case.env_type}/{case.name}"


def _extract_eval_metrics(run_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stage_metrics = run_result.get("stage_metrics", {}) if isinstance(run_result, dict) else {}
    eval_self = stage_metrics.get("eval_after_stage4_self")
    if isinstance(eval_self, dict):
        return eval_self
    return None


def _extract_repo_metrics(eval_metrics: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], List[int]]:
    if not eval_metrics:
        return None, None, []
    pass_masked = eval_metrics.get("repo_pass_rate")
    pass_unmasked = None
    steps_unmasked: List[int] = []
    if isinstance(eval_metrics.get("unmasked"), dict):
        unmasked = eval_metrics.get("unmasked", {})
        pass_unmasked = unmasked.get("repo_pass_rate")
        steps_unmasked = [int(x) for x in unmasked.get("repo_steps_to_pass", []) if isinstance(x, (int, float))]
    elif "unmasked_repo_pass_rate" in eval_metrics:
        pass_unmasked = eval_metrics.get("unmasked_repo_pass_rate")
    return pass_masked, pass_unmasked, steps_unmasked


def _tools_score(
    pass_unmasked: Optional[float],
    invalid_action_rate: Optional[float],
    mean_steps_unmasked: Optional[float],
    invalid_target: float = 0.01,
    steps_target: float = 12.0,
) -> Optional[float]:
    if pass_unmasked is None:
        return None
    pass_component = max(0.0, min(1.0, float(pass_unmasked)))
    if invalid_action_rate is None:
        invalid_component = 1.0
    else:
        invalid_component = max(0.0, min(1.0, 1.0 - float(invalid_action_rate) / float(invalid_target)))
    if mean_steps_unmasked is None or mean_steps_unmasked <= 0.0:
        steps_component = 1.0
    else:
        steps_component = max(0.0, min(1.0, float(steps_target) / float(mean_steps_unmasked)))
    return float(pass_component * invalid_component * steps_component)


def _bounded_return_score(value: Optional[float], *, center: float = 0.0, scale: float = 10.0) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    denom = max(1e-6, float(scale))
    v = float(value)
    return float(1.0 / (1.0 + math.exp(-(v - float(center)) / denom)))


def _core_score(
    mean_return: Optional[float],
    test_mean_return: Optional[float],
    *,
    center: float = 0.0,
    scale: float = 10.0,
) -> Optional[float]:
    """
    Squash returns to [0,1] and average train/test components.
    This provides a bounded score while preserving monotonicity.
    """

    vals: List[float] = []
    if isinstance(mean_return, (int, float)) and math.isfinite(float(mean_return)):
        vals.append(float(mean_return))
    if isinstance(test_mean_return, (int, float)) and math.isfinite(float(test_mean_return)):
        vals.append(float(test_mean_return))
    if not vals:
        return None
    denom = max(1e-6, float(scale))
    comps = [1.0 / (1.0 + math.exp(-(v - float(center)) / denom)) for v in vals]
    return float(sum(comps) / len(comps))


def _language_score(pass_rate: Optional[float], ood_pass_rate: Optional[float]) -> Optional[float]:
    if pass_rate is None:
        return None
    p = max(0.0, min(1.0, float(pass_rate)))
    if ood_pass_rate is None:
        return p
    ood = max(0.0, min(1.0, float(ood_pass_rate)))
    return float(min(p, ood))


def _social_score(success_rate: Optional[float], transfer_rate: Optional[float]) -> Optional[float]:
    if success_rate is None:
        return None
    s = max(0.0, min(1.0, float(success_rate)))
    if transfer_rate is None:
        return s
    t = max(0.0, min(1.0, float(transfer_rate)))
    return float(min(s, t))


def _lifelong_score(forgetting_gap: Optional[float], forward_transfer: Optional[float]) -> Optional[float]:
    if forgetting_gap is None and forward_transfer is None:
        return None
    forget_component = 1.0
    if isinstance(forgetting_gap, (int, float)) and math.isfinite(float(forgetting_gap)):
        # Near-zero forgetting is best.
        forget_component = max(0.0, min(1.0, float(math.exp(-abs(float(forgetting_gap)) / 5.0))))
    forward_component = 1.0
    bounded_forward = _bounded_return_score(forward_transfer, center=0.0, scale=5.0)
    if bounded_forward is not None:
        forward_component = bounded_forward
    return float(forget_component * forward_component)


def _run_safety_smoke() -> Dict[str, Any]:
    try:
        import torch
        from trainer import Trainer

        class _Dummy:
            safety_threshold = 0.5
            safety_penalty_coef = 2.0

        scores_main = torch.tensor([1.0, 1.0])
        scores_safety = torch.tensor([0.1, 0.6])
        penalized = Trainer._apply_safety_penalty(_Dummy(), scores_main, scores_safety)
        ok = bool(penalized[1] > penalized[0])
    except Exception:
        ok = False
    return {
        "safety_planner_ok": ok,
        "constraint_compliance": None,
        "catastrophic_fail_rate": None,
    }


def _build_suite_specs(
    *,
    minigrid_override: Optional[List[str]],
    computer_override: Optional[List[str]],
    repo_override: Optional[List[str]],
    ood: bool,
) -> Dict[str, SuiteSpec]:
    minigrid_scenarios = minigrid_override or [
        "minigrid-empty",
        "minigrid-doorkey",
        "test:minigrid-lavacrossing",
    ]
    repo_default = ["train:proc_mixed_loop", "test:proc_mixed_loop"]
    if ood:
        repo_default = ["train:proc_mixed_loop", "test:proc_mixed_ood_loop"]
    repo_scenarios = repo_override or repo_default

    specs = {
        "core": SuiteSpec(
            name="core",
            cases=[
                BenchCase(name="gridworld", env_type="gridworld"),
                BenchCase(name="minigrid", env_type="minigrid", minigrid_scenarios=minigrid_scenarios),
            ],
            implemented=True,
            description="GridWorld + MiniGrid baseline.",
        ),
        "tools": SuiteSpec(
            name="tools",
            cases=[
                BenchCase(name="tools_basic", env_type="tools"),
                BenchCase(name="repo_toolloop", env_type="repo", repo_scenarios=repo_scenarios),
            ],
            implemented=True,
            description="ToolEnv + RepoToolEnv procedural loop.",
        ),
        "tools_open": SuiteSpec(
            name="tools_open",
            cases=[],
            implemented=False,
            description="Placeholder for open-action tool tasks.",
        ),
        "language": SuiteSpec(
            name="language",
            cases=[
                BenchCase(name="instruction_basic", env_type="instruction"),
            ],
            implemented=True,
            description="Instruction-following/generalization suite.",
        ),
        "social": SuiteSpec(
            name="social",
            cases=[
                BenchCase(name="social_basic", env_type="social"),
            ],
            implemented=True,
            description="Social interaction/transfer suite.",
        ),
        "lifelong": SuiteSpec(
            name="lifelong",
            cases=[
                BenchCase(name="lifelong_gridworld", env_type="gridworld"),
            ],
            implemented=True,
            description="Continual adaptation/forgetting suite.",
        ),
        "safety": SuiteSpec(
            name="safety",
            cases=[],
            implemented=True,
            description="Minimal safety sanity checks.",
        ),
    }
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGI-Bench suites and emit a JSON report.")
    parser.add_argument(
        "--suite",
        type=str,
        default="agi_v1",
        choices=["core", "tools", "tools_open", "language", "social", "lifelong", "safety", "agi_v1", "quick"],
        help="Suite name to run.",
    )
    parser.add_argument(
        "--report",
        "--out",
        dest="report",
        type=str,
        default=None,
        help="Output JSON path (default: reports/bench_<suite>_<ts>.json).",
    )
    parser.add_argument("--log-dir", type=str, default="bench_logs", help="Per-run JSONL logs directory.")
    parser.add_argument("--mode", type=str, default="stage4", choices=["stage4", "lifelong"], help="Experiment mode.")
    parser.add_argument(
        "--variants",
        type=str,
        default="full",
        help="Comma-separated agent variants to run.",
    )
    parser.add_argument("--seeds", type=str, nargs="*", default=None, help="Seeds to run (CSV or space list).")
    parser.add_argument("--quick", action="store_true", help="Smaller/faster settings for runnable suites.")
    parser.add_argument("--ood", action="store_true", help="Use OOD splits where supported.")
    parser.add_argument("--hang_dump_sec", type=int, default=600, help="Stack dump interval for hangs (0 disables).")
    parser.add_argument(
        "--max-episode-steps-eval",
        type=int,
        default=200,
        help="Hard cap for eval episode steps (timeout guard).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution for all suite runs (recommended on unstable CUDA setups).",
    )
    parser.add_argument(
        "--allow-cuda",
        action="store_true",
        help="Allow CUDA even on platforms where bench would auto-force CPU for stability.",
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--masked", "--masked-only", dest="masked_only", action="store_true", help="Report masked metrics only (tools).")
    mask_group.add_argument("--unmasked", "--unmasked-only", dest="unmasked_only", action="store_true", help="Report unmasked metrics only (tools).")
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
    parser.add_argument("--minigrid-scenarios", type=str, default=None, help="Override MiniGrid scenarios (CSV).")
    parser.add_argument("--computer-scenarios", type=str, default=None, help="Override Computer scenarios (CSV).")
    parser.add_argument("--repo-scenarios", type=str, default=None, help="Override Repo scenarios (CSV).")
    return parser.parse_args()


def _run_suite(
    suite: SuiteSpec,
    *,
    seeds: List[int],
    variants: List[str],
    mode: str,
    quick: bool,
    quick_stub: bool,
    log_dir: str,
    use_skills: bool,
    skill_mode: str,
    n_latent_skills: int,
    masked_only: bool,
    unmasked_only: bool,
    eval_max_steps: int,
    force_cpu: bool,
    auto_force_cpu_repo: bool,
    report: Dict[str, Any],
    report_path: Path,
) -> Dict[str, Any]:
    suite_result: Dict[str, Any] = {
        "name": suite.name,
        "status": "running",
        "score": None,
        "metrics": _metric_template(suite.name),
        "per_env": [],
        "notes": [],
    }
    report["suites"].append(suite_result)
    _save_report(report_path, report)

    if suite.name == "safety":
        suite_result["metrics"].update(_run_safety_smoke())
        suite_result["status"] = "ok"
        _save_report(report_path, report)
        return suite_result

    if quick_stub or not suite.implemented:
        suite_result["status"] = "stub"
        suite_result["notes"].append("stubbed suite for Gate 0")
        _save_report(report_path, report)
        return suite_result

    episodes_per_phase = 50
    n_steps = 1024
    planning_horizon = 12
    planner_rollouts = 4
    planning_coef = 0.30
    lifelong_eps = 50
    stage1_steps = 5000
    stage1_batches = 200
    eval_episodes = 20
    lifecycle_eval_episodes = 20
    lifecycle_online_episodes = 50
    self_model_batches = 200
    self_reflection_batches = 200
    stage3c_batches = 50
    stage3c_collect_episodes = 10
    run_self_reflection = True
    run_stage3c = True
    run_lifecycle = True
    if quick:
        episodes_per_phase = 8
        n_steps = 128
        planning_horizon = 6
        planner_rollouts = 2
        planning_coef = 0.25
        lifelong_eps = 10
        stage1_steps = 200
        stage1_batches = 10
        eval_episodes = 3
        lifecycle_eval_episodes = 2
        lifecycle_online_episodes = 2
        self_model_batches = 20
        self_reflection_batches = 5
        stage3c_batches = 5
        stage3c_collect_episodes = 2
        run_self_reflection = False
        run_stage3c = False
        run_lifecycle = False

    run_records: List[Dict[str, Any]] = []
    any_error = False
    any_timeout = False

    report["meta"]["config"].update(
        {
            "eval_episodes": int(eval_episodes),
            "lifecycle_eval_episodes": int(lifecycle_eval_episodes),
            "lifecycle_online_episodes": int(lifecycle_online_episodes),
            "self_model_batches": int(self_model_batches),
            "self_reflection_batches": int(self_reflection_batches),
            "stage3c_batches": int(stage3c_batches),
            "stage3c_collect_episodes": int(stage3c_collect_episodes),
            "run_self_reflection": bool(run_self_reflection),
            "run_stage3c": bool(run_stage3c),
            "run_lifecycle": bool(run_lifecycle),
            "force_cpu": bool(force_cpu),
            "auto_force_cpu_repo": bool(auto_force_cpu_repo),
        }
    )
    _save_report(report_path, report)

    for case in suite.cases:
        for variant in variants:
            for seed in seeds:
                run_id = f"bench_{suite.name}_{case.name}_{variant}_seed{seed}"
                print(f"[BENCH] suite={suite.name} case={case.name} env={case.env_type} variant={variant} seed={seed}")
                status = "ok"
                error_msg = None
                res: Dict[str, Any] = {}
                try:
                    repo_bc_episodes = 0
                    repo_online_bc_coef = 0.10
                    action_mask_dropout_prob = 0.0
                    run_force_cpu = bool(force_cpu)
                    run_mode = str(mode)
                    if str(case.env_type) == "repo":
                        # Gate-1 tuned defaults from local sweep:
                        # online_bc=0.0, bc_eps=32, mask_drop=0.2
                        repo_bc_episodes = 32 if quick else 128
                        repo_online_bc_coef = 0.0
                        action_mask_dropout_prob = 0.20
                        run_force_cpu = bool(run_force_cpu or auto_force_cpu_repo)
                    if suite.name == "lifelong":
                        run_mode = "lifelong"
                        run_lifecycle = True
                    res = run_experiment(
                        seed=int(seed),
                        mode=run_mode,
                        agent_variant=str(variant),
                        env_type=str(case.env_type),
                        schedule_mode="iid",
                        episodes_per_phase=int(episodes_per_phase),
                        n_steps=int(n_steps),
                        stage2_updates=1,
                        stage4_updates=1,
                        planning_horizon=int(planning_horizon),
                        planner_mode="rollout",
                        planner_rollouts=int(planner_rollouts),
                        planning_coef=float(planning_coef),
                        lifelong_episodes_per_chapter=int(lifelong_eps),
                        minigrid_scenarios=case.minigrid_scenarios,
                        computer_scenarios=case.computer_scenarios,
                        repo_scenarios=case.repo_scenarios,
                        log_dir=str(log_dir),
                        run_id=run_id,
                        use_skills=bool(use_skills),
                        skill_mode=str(skill_mode),
                        n_latent_skills=int(n_latent_skills),
                        stage1_steps=int(stage1_steps),
                        stage1_batches=int(stage1_batches),
                        eval_max_steps=int(eval_max_steps),
                        eval_episodes=int(eval_episodes),
                        lifecycle_eval_episodes=int(lifecycle_eval_episodes),
                        lifecycle_online_episodes=int(lifecycle_online_episodes),
                        self_model_batches=int(self_model_batches),
                        self_reflection_batches=int(self_reflection_batches),
                        stage3c_batches=int(stage3c_batches),
                        stage3c_collect_episodes=int(stage3c_collect_episodes),
                        run_self_reflection=bool(run_self_reflection),
                        run_stage3c=bool(run_stage3c),
                        run_lifecycle=bool(run_lifecycle),
                        force_cpu=bool(run_force_cpu),
                        action_mask_dropout_prob=float(action_mask_dropout_prob),
                        repo_online_bc_coef=float(repo_online_bc_coef),
                        repo_bc_pretrain_episodes=int(repo_bc_episodes),
                        repo_bc_pretrain_max_steps=int(eval_max_steps),
                    )
                except Exception as exc:
                    status = "error"
                    error_msg = f"{type(exc).__name__}: {exc}"
                    any_error = True

                eval_metrics = _extract_eval_metrics(res or {})
                timeout_eps = 0
                if isinstance(eval_metrics, dict):
                    timeout_eps = int(eval_metrics.get("timeout_episodes", 0) or 0)
                capped_all_eps = timeout_eps >= int(eval_episodes)

                pass_masked, pass_unmasked, steps_unmasked = _extract_repo_metrics(eval_metrics)
                mean_steps_unmasked = _safe_mean([float(x) for x in steps_unmasked])
                pass_flag = None
                if pass_unmasked is not None:
                    pass_flag = bool(float(pass_unmasked) >= 0.5)
                elif pass_masked is not None:
                    pass_flag = bool(float(pass_masked) >= 0.5)
                if status in {"error", "timeout"}:
                    pass_flag = False

                per_env_entry: Dict[str, Any] = {
                    "env": _case_label(case),
                    "seed": int(seed),
                    "variant": str(variant),
                    "status": status,
                    "pass": pass_flag,
                    "pass_rate_masked": pass_masked,
                    "pass_rate_unmasked": pass_unmasked,
                    "steps": mean_steps_unmasked,
                    "invalid_rate": None,
                }
                if error_msg:
                    per_env_entry["error"] = error_msg
                    suite_result["notes"].append(error_msg)
                if timeout_eps > 0:
                    per_env_entry["timeout_episodes"] = timeout_eps
                    if capped_all_eps:
                        per_env_entry["capped_all_eval_episodes"] = True
                        suite_result["notes"].append(
                            f"eval_step_cap_reached_for_all_episodes:{_case_label(case)}:seed={int(seed)}"
                        )

                suite_result["per_env"].append(per_env_entry)
                run_records.append(
                    {
                        "case": case,
                        "variant": variant,
                        "seed": seed,
                        "result": res,
                        "eval": eval_metrics,
                        "status": status,
                    }
                )
                _save_report(report_path, report)

    metrics = _metric_template(suite.name)
    score = None
    notes: List[str] = []
    if suite.name == "tools":
        masked_vals = []
        unmasked_vals = []
        steps_vals: List[int] = []
        for record in run_records:
            case = record["case"]
            if case.env_type != "repo" or record.get("status") != "ok":
                continue
            eval_metrics = record.get("eval")
            pass_masked, pass_unmasked, steps_unmasked = _extract_repo_metrics(eval_metrics)
            if pass_masked is not None:
                masked_vals.append(float(pass_masked))
            if pass_unmasked is not None:
                unmasked_vals.append(float(pass_unmasked))
            steps_vals.extend([int(x) for x in steps_unmasked])
        pass_rate_masked = _safe_mean(masked_vals)
        pass_rate_unmasked = _safe_mean(unmasked_vals)
        mean_steps_unmasked = _safe_mean([float(x) for x in steps_vals])
        invalid_action_rate = None
        mean_invalid_mass = None
        mask_pred_f1 = None
        mask_pred_auc = None
        bc_pretrain_used = False
        action_mask_dropout_prob = None
        invalid_vals: List[float] = []
        invalid_rate_vals: List[float] = []
        mask_f1_vals: List[float] = []
        mask_auc_vals: List[float] = []
        if run_records:
            cfg = (run_records[0].get("result") or {}).get("config", {})
            if isinstance(cfg, dict):
                action_mask_dropout_prob = cfg.get("action_mask_dropout_prob")
        for record in run_records:
            res = record.get("result") or {}
            if not isinstance(res, dict):
                continue
            stage_metrics = res.get("stage_metrics", {})
            if isinstance(stage_metrics, dict):
                train_stats = stage_metrics.get("stage4_train_stats", {})
                if isinstance(train_stats, dict):
                    ir = train_stats.get("invalid_action_rate")
                    if isinstance(ir, (int, float)) and math.isfinite(float(ir)):
                        invalid_rate_vals.append(float(ir))
                    v = train_stats.get("mean_invalid_action_mass")
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        invalid_vals.append(float(v))
                    f1 = train_stats.get("mask_pred_f1")
                    if isinstance(f1, (int, float)) and math.isfinite(float(f1)):
                        mask_f1_vals.append(float(f1))
                    auc = train_stats.get("mask_pred_auc")
                    if isinstance(auc, (int, float)) and math.isfinite(float(auc)):
                        mask_auc_vals.append(float(auc))
                if "repo_bc_pretrain" in stage_metrics:
                    bc_pretrain_used = True
        if invalid_vals:
            mean_invalid_mass = _safe_mean(invalid_vals)
        if invalid_rate_vals:
            invalid_action_rate = _safe_mean(invalid_rate_vals)
        if mask_f1_vals:
            mask_pred_f1 = _safe_mean(mask_f1_vals)
        if mask_auc_vals:
            mask_pred_auc = _safe_mean(mask_auc_vals)
        if masked_only:
            pass_rate_unmasked = None
            mean_steps_unmasked = None
            notes.append("masked_only: unmasked metrics suppressed")
        if unmasked_only:
            pass_rate_masked = None
            notes.append("unmasked_only: masked metrics suppressed")
        metrics.update(
            {
                "pass_rate_masked": pass_rate_masked,
                "pass_rate_unmasked": pass_rate_unmasked,
                "mean_steps_to_pass_unmasked": mean_steps_unmasked,
                "invalid_action_rate": invalid_action_rate,
                "mask_pred_f1": mask_pred_f1,
                "mask_pred_auc": mask_pred_auc,
                "bc_pretrain_used": bool(bc_pretrain_used),
                "action_mask_dropout_prob": action_mask_dropout_prob,
                "mean_invalid_mass": mean_invalid_mass,
                "ood_gap": None,
            }
        )
        score = _tools_score(pass_rate_unmasked, invalid_action_rate, mean_steps_unmasked)
    elif suite.name == "core":
        mean_returns = []
        test_returns = []
        for record in run_records:
            if record.get("status") != "ok":
                continue
            eval_metrics = record.get("eval") or {}
            mean_val = eval_metrics.get("mean_return")
            if mean_val is not None:
                mean_returns.append(float(mean_val))
            test_val = eval_metrics.get("test_mean_return")
            if test_val is not None:
                test_returns.append(float(test_val))
        metrics.update(
            {
                "mean_return": _safe_mean(mean_returns),
                "test_mean_return": _safe_mean(test_returns),
                "ood_gap": None,
            }
        )
        score = _core_score(metrics.get("mean_return"), metrics.get("test_mean_return"))
    elif suite.name == "language":
        pass_scores: List[float] = []
        ood_scores: List[float] = []
        for record in run_records:
            if record.get("status") != "ok":
                continue
            eval_metrics = record.get("eval") or {}
            p = _bounded_return_score(eval_metrics.get("mean_return"), center=0.0, scale=1.0)
            if p is not None:
                pass_scores.append(float(p))
            ood = _bounded_return_score(eval_metrics.get("test_mean_return"), center=0.0, scale=1.0)
            if ood is not None:
                ood_scores.append(float(ood))
        pass_rate = _safe_mean(pass_scores)
        ood_pass_rate = _safe_mean(ood_scores)
        causal_drop = None
        if pass_rate is not None and ood_pass_rate is not None:
            causal_drop = max(0.0, float(pass_rate) - float(ood_pass_rate))
        metrics.update(
            {
                "pass_rate": pass_rate,
                "ood_pass_rate": ood_pass_rate,
                "causal_drop": causal_drop,
            }
        )
        score = _language_score(pass_rate, ood_pass_rate)
    elif suite.name == "social":
        success_vals: List[float] = []
        transfer_vals: List[float] = []
        for record in run_records:
            if record.get("status") != "ok":
                continue
            eval_metrics = record.get("eval") or {}
            s = _bounded_return_score(eval_metrics.get("mean_return"), center=0.0, scale=1.0)
            if s is not None:
                success_vals.append(float(s))
            t = _bounded_return_score(eval_metrics.get("test_mean_return"), center=0.0, scale=1.0)
            if t is not None:
                transfer_vals.append(float(t))
        success_rate = _safe_mean(success_vals)
        transfer_rate = _safe_mean(transfer_vals)
        metrics.update(
            {
                "success_rate": success_rate,
                "transfer_rate": transfer_rate,
            }
        )
        score = _social_score(success_rate, transfer_rate)
    elif suite.name == "lifelong":
        forgetting_vals: List[float] = []
        transfer_vals: List[float] = []
        for record in run_records:
            if record.get("status") != "ok":
                continue
            res = record.get("result") or {}
            if not isinstance(res, dict):
                continue
            stage_metrics = res.get("stage_metrics", {})
            if not isinstance(stage_metrics, dict):
                continue
            ll = stage_metrics.get("lifelong_eval", {})
            if not isinstance(ll, dict):
                continue
            fg = ll.get("lifelong_forgetting_R1_gap")
            if isinstance(fg, (int, float)) and math.isfinite(float(fg)):
                forgetting_vals.append(float(fg))
            ft_parts: List[float] = []
            for key in ("lifelong_adaptation_R2_delta", "lifelong_adaptation_R3_delta"):
                v = ll.get(key)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    ft_parts.append(float(v))
            if ft_parts:
                transfer_vals.append(float(sum(ft_parts) / len(ft_parts)))
        forgetting_gap = _safe_mean(forgetting_vals)
        forward_transfer = _safe_mean(transfer_vals)
        metrics.update(
            {
                "forgetting_gap": forgetting_gap,
                "forward_transfer": forward_transfer,
            }
        )
        score = _lifelong_score(forgetting_gap, forward_transfer)

    suite_result["metrics"] = metrics
    suite_result["score"] = score
    suite_status = "ok"
    if any_error:
        suite_status = "error"
    elif any_timeout:
        suite_status = "timeout"
    suite_result["status"] = suite_status
    suite_result["notes"].extend(notes)
    _save_report(report_path, report)
    return suite_result


def main() -> int:
    args = parse_args()
    ts = int(time.time())
    report_path = Path(args.report or (Path("reports") / f"bench_{args.suite}_{ts}.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    auto_force_cpu_repo = bool(
        sys.platform.startswith("win") and sys.version_info >= (3, 13) and not bool(args.allow_cuda)
    )
    effective_force_cpu = bool(args.force_cpu)
    if auto_force_cpu_repo and not effective_force_cpu:
        print("[BENCH] Windows + Python 3.13 detected: forcing CPU for repo cases for stability. Use --allow-cuda to override.")

    hang_sec = int(args.hang_dump_sec or 0)
    if hang_sec > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(hang_sec, repeat=True)

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    seeds = _parse_seeds(args.seeds)
    override_minigrid = _split_csv(args.minigrid_scenarios)
    override_computer = _split_csv(args.computer_scenarios)
    override_repo = _split_csv(args.repo_scenarios)

    suite_specs = _build_suite_specs(
        minigrid_override=override_minigrid,
        computer_override=override_computer,
        repo_override=override_repo,
        ood=bool(args.ood),
    )

    if args.suite == "agi_v1":
        selected = list(SUITE_ORDER)
        quick_stub = False
    elif args.suite == "quick":
        selected = list(SUITE_ORDER)
        quick_stub = True
    else:
        selected = [args.suite]
        quick_stub = False

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "timestamp": int(ts),
            "git_commit": _git_commit(),
            "seed_list": seeds,
            "suite": str(args.suite),
            "ood": bool(args.ood),
            "quick": bool(args.quick) or bool(quick_stub),
            "config": {
                "mode": str(args.mode),
                "variants": variants,
                "use_skills": bool(args.use_skills),
                "skill_mode": str(args.skill_mode),
                "n_latent_skills": int(args.n_latent_skills),
                "masked_only": bool(args.masked_only),
                "unmasked_only": bool(args.unmasked_only),
                "eval_max_steps": int(args.max_episode_steps_eval),
                "force_cpu": bool(effective_force_cpu),
                "auto_force_cpu_repo": bool(auto_force_cpu_repo),
            },
        },
        "overall": {
            "agi_score": 0.0,
            "notes": [],
            "gates": {
                "gate0": "fail",
                "gate1": "na",
                "gate2": "na",
            },
        },
        "suites": [],
    }
    _save_report(report_path, report)

    try:
        for name in selected:
            spec = suite_specs.get(name)
            if spec is None:
                continue
            _run_suite(
                spec,
                seeds=seeds,
                variants=variants,
                mode=args.mode,
                quick=bool(args.quick),
                quick_stub=quick_stub,
                log_dir=str(args.log_dir),
                use_skills=bool(args.use_skills),
                skill_mode=str(args.skill_mode),
                n_latent_skills=int(args.n_latent_skills),
                masked_only=bool(args.masked_only),
                unmasked_only=bool(args.unmasked_only),
                eval_max_steps=int(args.max_episode_steps_eval),
                force_cpu=bool(effective_force_cpu),
                auto_force_cpu_repo=bool(auto_force_cpu_repo),
                report=report,
                report_path=report_path,
            )
    except KeyboardInterrupt:
        _save_report(report_path, report)
        raise

    _save_report(report_path, report)
    print(f"[BENCH] Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
