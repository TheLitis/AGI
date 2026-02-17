import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid JSON root in {path}")
    return raw


def _suite(report: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    suites = report.get("suites", [])
    if not isinstance(suites, list):
        return None
    for s in suites:
        if isinstance(s, dict) and s.get("name") == name:
            return s
    return None


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _pick_suite(report: Dict[str, Any], requested: Optional[str]) -> Optional[Dict[str, Any]]:
    if isinstance(requested, str) and requested.strip():
        return _suite(report, requested.strip())
    return _suite(report, "planning_diag") or _suite(report, "long_horizon")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate planner reality-check metrics in a bench report.")
    parser.add_argument("--report", type=Path, required=True, help="Path to bench report JSON.")
    parser.add_argument("--suite", type=str, default=None, help="Suite name (default: planning_diag, fallback long_horizon).")
    parser.add_argument("--min-steps", type=int, default=200, help="Minimum planner reality sample count.")
    parser.add_argument("--min-corr-adv", type=float, default=0.0, help="Minimum planner corr advantage over policy.")
    parser.add_argument("--min-top1-adv", type=float, default=0.0, help="Minimum planner top1 n-step advantage.")
    args = parser.parse_args()

    report = _load_json(args.report)
    suite = _pick_suite(report, args.suite)
    if not isinstance(suite, dict):
        print("[BLOCKED] suite_not_found")
        return 1
    suite_name = str(suite.get("name", "unknown"))
    metrics = suite.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    steps = _as_float(metrics.get("planner_reality_steps"))
    planner_corr = _as_float(metrics.get("planner_score_nstep_corr"))
    policy_corr = _as_float(metrics.get("policy_score_nstep_corr"))
    corr_adv = _as_float(metrics.get("planner_score_corr_advantage"))
    top1_adv = _as_float(metrics.get("planner_top1_advantage_nstep"))
    regret_proxy = _as_float(metrics.get("planner_regret_proxy_nstep"))
    if corr_adv is None and planner_corr is not None and policy_corr is not None:
        corr_adv = float(planner_corr - policy_corr)

    print(f"suite={suite_name}")
    print(f"planner_reality_steps={steps}")
    print(f"planner_score_nstep_corr={planner_corr}")
    print(f"policy_score_nstep_corr={policy_corr}")
    print(f"planner_score_corr_advantage={corr_adv}")
    print(f"planner_top1_advantage_nstep={top1_adv}")
    print(f"planner_regret_proxy_nstep={regret_proxy}")

    checks = {
        "planner_reality_steps": steps is not None and steps >= float(max(1, int(args.min_steps))),
        "planner_score_corr_advantage": corr_adv is not None and corr_adv >= float(args.min_corr_adv),
        "planner_top1_advantage_nstep": top1_adv is not None and top1_adv >= float(args.min_top1_adv),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"[BLOCKED] failed={','.join(failed)}")
        return 1
    print("[OPEN] Planner reality check thresholds satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
