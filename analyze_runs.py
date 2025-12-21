#!/usr/bin/env python
"""
Utility to summarize individual run JSON logs.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional
from statistics import mean
from statistics import mean as _mean


def find_json_files(log_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.lower().endswith(".json"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def safe_get_mean_return(stage_metrics: Dict[str, Any], key: str) -> Optional[float]:
    m = stage_metrics.get(key)
    if m is None:
        return None
    return m.get("mean_return")


def format_val(x: Optional[float], fmt: str = "{:.3f}") -> str:
    if x is None:
        return "None"
    try:
        return fmt.format(x)
    except Exception:
        return str(x)


def parse_trait_reflection_events(events: Any) -> Dict[str, Any]:
    """
    Extract mean deltas and counts from a structured trait_reflection_log.
    Compatible with legacy string-only logs (returns empty stats).
    """
    deltas_all: Dict[str, List[float]] = {"survival": [], "food": [], "damage": [], "move": []}
    deltas_by_regime: Dict[str, Dict[str, List[float]]] = {}
    counts_by_regime: Dict[str, int] = {}
    structured = False

    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, dict):
                continue
            delta = ev.get("delta_traits")
            if not isinstance(delta, dict):
                continue
            structured = True
            reg = str(ev.get("regime", "unknown"))
            counts_by_regime[reg] = counts_by_regime.get(reg, 0) + 1
            for k in deltas_all.keys():
                v = delta.get(k)
                if isinstance(v, (int, float)):
                    deltas_all[k].append(float(v))
                    deltas_by_regime.setdefault(reg, {}).setdefault(k, []).append(float(v))

    trait_delta_mean = {k: (_mean(v) if v else None) for k, v in deltas_all.items()}
    trait_delta_mean_by_regime = {
        reg: {k: (_mean(vals) if vals else None) for k, vals in ks.items()}
        for reg, ks in deltas_by_regime.items()
    }
    total_updates = int(sum(counts_by_regime.values()))
    return {
        "structured": structured,
        "overall_mean": trait_delta_mean,
        "by_regime_mean": trait_delta_mean_by_regime,
        "counts_by_regime": counts_by_regime,
        "total_updates": total_updates,
    }


def summarize_run(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seed = data.get("seed", None)
    mode = data.get("mode", None)
    device = data.get("device", None)
    # пробуем два варианта ключа, на случай разных версий логгера
    schedule_mode = data.get("schedule_mode", data.get("schedule", "unknown"))
    n_scenarios = data.get("n_scenarios", None)
    meta_conflict = data.get("meta_conflict_ma", None)
    meta_uncertainty = data.get("meta_uncertainty_ma", None)

    stage_metrics = data.get("stage_metrics", {})

    eval_before = safe_get_mean_return(stage_metrics, "eval_before_rl")
    after_s2 = safe_get_mean_return(stage_metrics, "eval_after_stage2")

    s3c_no_self = safe_get_mean_return(stage_metrics, "eval_after_stage3c_no_self")
    s3c_self = safe_get_mean_return(stage_metrics, "eval_after_stage3c_self")

    s4_self = safe_get_mean_return(stage_metrics, "eval_after_stage4_self")
    s4_no_self = safe_get_mean_return(stage_metrics, "eval_after_stage4_no_self")

    # дельты self vs no_self
    delta_s3c_self = None
    if s3c_no_self is not None and s3c_self is not None:
        delta_s3c_self = s3c_self - s3c_no_self

    delta_s4_self = None
    if s4_no_self is not None and s4_self is not None:
        delta_s4_self = s4_self - s4_no_self

    # SelfModel probe
    probe = stage_metrics.get("self_model_probe_after_stage4", {})
    corr_return = probe.get("corr_return")
    corr_survival = probe.get("corr_survival")
    mean_true_return0 = probe.get("mean_true_return0")
    mean_pred_return0 = probe.get("mean_pred_return0")
    mean_true_survival = probe.get("mean_true_survival")
    mean_pred_survival = probe.get("mean_pred_survival")

    forgetting = stage_metrics.get("lifelong_forgetting", {}) or {}
    forgetting_gap: Dict[str, float] = {}
    forgetting_baseline: Dict[str, float] = {}
    forgetting_current: Dict[str, float] = {}
    if isinstance(forgetting, dict):
        gap_raw = forgetting.get("gap", {})
        if isinstance(gap_raw, dict):
            forgetting_gap = {str(k): float(v) for k, v in gap_raw.items() if isinstance(v, (int, float))}
        base_raw = forgetting.get("baseline", {})
        if isinstance(base_raw, dict):
            forgetting_baseline = {str(k): float(v) for k, v in base_raw.items() if isinstance(v, (int, float))}
        cur_raw = forgetting.get("current", {})
        if isinstance(cur_raw, dict):
            forgetting_current = {str(k): float(v) for k, v in cur_raw.items() if isinstance(v, (int, float))}

    gap_values = [v for v in forgetting_gap.values() if isinstance(v, (int, float))]
    mean_gap = float(_mean(gap_values)) if gap_values else None
    max_gap = float(max(gap_values)) if gap_values else None

    # trait reflection events (structured + backward-compatible)
    tr_events = stage_metrics.get("trait_reflection_log", []) or []
    tr_stats = parse_trait_reflection_events(tr_events)
    trait_delta_mean = tr_stats["overall_mean"]
    trait_delta_mean_by_regime = tr_stats["by_regime_mean"]
    trait_counts_by_regime = tr_stats["counts_by_regime"]
    trait_total_updates = tr_stats["total_updates"]
    trait_log_structured = tr_stats["structured"]

    result: Dict[str, Any] = {
        "path": path,
        "seed": seed,
        "mode": mode,
        "device": device,
        "schedule_mode": schedule_mode,
        "n_scenarios": n_scenarios,
        "meta_conflict_ma": meta_conflict,
        "meta_uncertainty_ma": meta_uncertainty,
        "eval_before_rl": eval_before,
        "eval_after_stage2": after_s2,
        "eval_after_stage3c_no_self": s3c_no_self,
        "eval_after_stage3c_self": s3c_self,
        "eval_after_stage4_self": s4_self,
        "eval_after_stage4_no_self": s4_no_self,
        "delta_stage3c_self_minus_no_self": delta_s3c_self,
        "delta_stage4_self_minus_no_self": delta_s4_self,
        # probe
        "corr_return": corr_return,
        "corr_survival": corr_survival,
        "mean_true_return0": mean_true_return0,
        "mean_pred_return0": mean_pred_return0,
        "mean_true_survival": mean_true_survival,
        "mean_pred_survival": mean_pred_survival,
        # traits
        "final_traits": data.get("final_traits", None),
        "final_preference_weights": data.get("final_preference_weights", None),
        # forgetting
        "forgetting_gap_mean": mean_gap,
        "forgetting_gap_max": max_gap,
        "forgetting_gap_per_regime": forgetting_gap,
        "forgetting_baseline": forgetting_baseline,
        "forgetting_current": forgetting_current,
        # reflection stats
        "trait_delta_mean": trait_delta_mean,
        "trait_delta_mean_by_regime": trait_delta_mean_by_regime,
        "trait_reflection_counts_by_regime": trait_counts_by_regime,
        "trait_reflection_total_updates": trait_total_updates,
        "trait_reflection_structured": trait_log_structured,
    }

    return result


def print_run_summary(run: Dict[str, Any], idx: int) -> None:
    print("=" * 80)
    print(f"[{idx}] {run['path']}")
    print(
        f"  seed={run['seed']}, mode={run['mode']}, "
        f"schedule_mode={run['schedule_mode']}, device={run['device']}, "
        f"n_scenarios={run['n_scenarios']}"
    )
    print(
        "  meta_conflict_ma="
        + format_val(run["meta_conflict_ma"])
        + ", meta_uncertainty_ma="
        + format_val(run["meta_uncertainty_ma"])
    )

    print("  mean_return by stage:")
    print("    eval_before_rl          :", format_val(run["eval_before_rl"]))
    print("    eval_after_stage2       :", format_val(run["eval_after_stage2"]))
    print(
        "    eval_after_stage3c (no self):",
        format_val(run["eval_after_stage3c_no_self"]),
    )
    print(
        "    eval_after_stage3c (self)   :",
        format_val(run["eval_after_stage3c_self"]),
    )
    print(
        "    eval_after_stage4 (self)    :",
        format_val(run["eval_after_stage4_self"]),
    )
    print(
        "    eval_after_stage4 (no self) :",
        format_val(run["eval_after_stage4_no_self"]),
    )

    print("  deltas self - no_self:")
    print(
        "    Stage 3c (self - no_self):",
        format_val(run["delta_stage3c_self_minus_no_self"]),
    )
    print(
        "    Stage 4  (self - no_self):",
        format_val(run["delta_stage4_self_minus_no_self"]),
    )

    # SelfModel probe summary
    print("  SelfModel probe (after Stage 4):")
    print("    corr(return_true, return_pred)  :", format_val(run["corr_return"]))
    print("    corr(survival_true, survival_pred):", format_val(run["corr_survival"]))
    print(
        "    mean true R0 / pred R0         :",
        format_val(run["mean_true_return0"]),
        "/",
        format_val(run["mean_pred_return0"]),
    )
    print(
        "    mean true S / pred S           :",
        format_val(run["mean_true_survival"]),
        "/",
        format_val(run["mean_pred_survival"]),
    )

    if run.get("forgetting_gap_mean") is not None:
        print(
            "  forgetting gap (regime-aware): mean=",
            format_val(run["forgetting_gap_mean"]),
            " max=",
            format_val(run["forgetting_gap_max"]),
        )
    gaps = run.get("forgetting_gap_per_regime") or {}
    if gaps:
        print("  forgetting gap per regime:")
        for reg, val in gaps.items():
            print(f"    {reg}: {format_val(val)}")

    # Немного про финальные ценности
    traits = run.get("final_traits")
    weights = run.get("final_preference_weights")
    if traits is not None and len(traits) > 0:
        print("  final traits:", [round(float(x), 4) for x in traits[0]])
    if weights is not None and len(weights) > 0:
        print(
            "  final preference weights (w_survive, w_food, w_danger, w_move):",
            [round(float(x), 4) for x in weights[0]],
        )
    td_mean = run.get("trait_delta_mean") or {}
    total_updates = run.get("trait_reflection_total_updates") or 0
    if td_mean:
        print("  mean trait delta (overall):", {k: format_val(v) for k, v in td_mean.items()})
    if total_updates:
        print(f"  trait reflection updates: {int(total_updates)}")
    per_regime = run.get("trait_delta_mean_by_regime") or {}
    counts = run.get("trait_reflection_counts_by_regime") or {}
    if per_regime:
        print("  trait reflections by regime:")
        for reg, deltas in per_regime.items():
            if not isinstance(deltas, dict):
                continue
            parts = []
            for trait in ["survival", "food", "damage", "move"]:
                v = deltas.get(trait)
                if isinstance(v, (int, float)):
                    parts.append(f"Δ{trait}={float(v):+0.3f}")
            count = counts.get(reg, 0)
            if parts:
                suffix = f" over {int(count)} updates" if count else ""
                print(f"    {reg}: {', '.join(parts)}{suffix}")


def aggregate_by_schedule(runs: List[Dict[str, Any]]) -> None:
    bucket: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        key = r.get("schedule_mode", "unknown")
        bucket.setdefault(key, []).append(r)

    print("\n" + "#" * 80)
    print("# Aggregated by schedule_mode")
    print("#" * 80)

    for sched, rs in bucket.items():
        if not rs:
            continue

        def collect(field: str) -> List[float]:
            vals: List[float] = []
            for rr in rs:
                v = rr.get(field)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals

        s4_self_vals = collect("eval_after_stage4_self")
        s4_noself_vals = collect("eval_after_stage4_no_self")
        delta4_vals = collect("delta_stage4_self_minus_no_self")
        corr_vals = collect("corr_return")

        print(f"\nSchedule mode: {sched} (runs: {len(rs)})")
        if s4_self_vals:
            print("  mean(eval_after_stage4_self)   =", format_val(mean(s4_self_vals)))
        if s4_noself_vals:
            print(
                "  mean(eval_after_stage4_no_self) =",
                format_val(mean(s4_noself_vals)),
            )
        if delta4_vals:
            print(
                "  mean(delta stage4 self-no_self) =",
                format_val(mean(delta4_vals)),
            )
        if corr_vals:
            print(
                "  mean(corr_return)               =",
                format_val(mean(corr_vals)),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Анализ JSON-логов запусков run_all.py (AGI proto-creature)."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        nargs="?",
        default="logs",
        help="Папка с JSON логами (по умолчанию: logs).",
    )
    args = parser.parse_args()

    files = find_json_files(args.log_dir)
    if not files:
        print(f"Не нашёл ни одного .json в {args.log_dir}")
        return

    print(f"Нашёл {len(files)} JSON лог(ов) в {args.log_dir}\n")

    runs: List[Dict[str, Any]] = []
    for i, path in enumerate(files, start=1):
        try:
            run = summarize_run(path)
            runs.append(run)
            print_run_summary(run, i)
        except Exception as e:
            print("=" * 80)
            print(f"[{i}] {path}")
            print("  !!! Ошибка при разборе лога:", e)

    aggregate_by_schedule(runs)


if __name__ == "__main__":
    main()
