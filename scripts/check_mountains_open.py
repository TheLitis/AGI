import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


THRESH_LONG_HORIZON = 0.65
THRESH_FORGETTING = -1.0
THRESH_FORWARD = 0.5


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


def _extract_metrics(long_report: Dict[str, Any], ll_report: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    long_suite = _suite(long_report, "long_horizon") or {}
    ll_suite = _suite(ll_report, "lifelong") or {}
    ll_metrics = ll_suite.get("metrics", {})
    if not isinstance(ll_metrics, dict):
        ll_metrics = {}
    long_score = _as_float(long_suite.get("score"))
    forgetting_gap = _as_float(ll_metrics.get("forgetting_gap"))
    forward_transfer = _as_float(ll_metrics.get("forward_transfer"))
    return long_score, forgetting_gap, forward_transfer


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether Mountains #2/#3 are open by Gate2 thresholds.")
    parser.add_argument("--long-horizon-report", type=Path, required=True, help="Path to long_horizon bench report JSON.")
    parser.add_argument("--lifelong-report", type=Path, required=True, help="Path to lifelong bench report JSON.")
    args = parser.parse_args()

    long_report = _load_json(args.long_horizon_report)
    lifelong_report = _load_json(args.lifelong_report)
    long_score, forgetting_gap, forward_transfer = _extract_metrics(long_report, lifelong_report)

    print(f"long_horizon.score={long_score} (threshold>={THRESH_LONG_HORIZON})")
    print(f"lifelong.forgetting_gap={forgetting_gap} (threshold>={THRESH_FORGETTING})")
    print(f"lifelong.forward_transfer={forward_transfer} (threshold>={THRESH_FORWARD})")

    checks = {
        "long_horizon.score": long_score is not None and long_score >= THRESH_LONG_HORIZON,
        "lifelong.forgetting_gap": forgetting_gap is not None and forgetting_gap >= THRESH_FORGETTING,
        "lifelong.forward_transfer": forward_transfer is not None and forward_transfer >= THRESH_FORWARD,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"[BLOCKED] failed={','.join(failed)}")
        return 1
    print("[OPEN] Mountains #2/#3 satisfy Gate2 thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
