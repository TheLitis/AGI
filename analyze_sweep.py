"""
Sweep analysis utilities. Robust to missing keys and prints clearly labelled tables.
Optionally generates matplotlib bar plots summarizing Phase C and Stage 4 results.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_STAGE4_SWEEP = Path("sweep_results") / "stage4_lifecycle_sweep.json"
DEFAULT_LIFELONG_SWEEP = Path("sweep_results") / "lifelong_sweep.json"
FIG_DIR = Path("figures")
EXPECTED_VARIANTS = ["full", "no_reflection", "no_self"]


def _variant_order(found_variants) -> List[str]:
    """Return variants in a stable expected order with any extras appended."""
    order = list(EXPECTED_VARIANTS)
    for v in found_variants:
        if v not in order:
            order.append(v)
    return order


def load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"[ERROR] Sweep file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def find_metric_by_substrings(metrics: Dict[str, Any], substrings: List[str]) -> Optional[Dict[str, Any]]:
    """
    Return the first metric dict whose key contains all given substrings (case-insensitive).
    """
    substrings_lower = [s.lower() for s in substrings]
    for k, v in metrics.items():
        key_l = k.lower()
        if all(s in key_l for s in substrings_lower):
            if isinstance(v, dict):
                return v
    return None


def first_present(metrics: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    """Return first metric dict by exact key match, or None."""
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, dict):
            return v
    return None


def aggregate_phase_c(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect Phase C returns for each variant, split by use_self flag.
    Returns dict: variant -> {"self": [...], "no_self": [...]}
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        phase_c_self = first_present(
            sm,
            ["lifecycle_phaseC_self", "phaseC_self"],
        ) or find_metric_by_substrings(sm, ["phase", "c", "self"])
        phase_c_no_self = first_present(
            sm,
            ["lifecycle_phaseC_no_self", "phaseC_no_self"],
        ) or find_metric_by_substrings(sm, ["phase", "c", "no", "self"])
        out.setdefault(variant, {"self": [], "no_self": []})
        if phase_c_self and isinstance(phase_c_self, dict) and "mean_return" in phase_c_self:
            out[variant]["self"].append(phase_c_self["mean_return"])
        if phase_c_no_self and isinstance(phase_c_no_self, dict) and "mean_return" in phase_c_no_self:
            out[variant]["no_self"].append(phase_c_no_self["mean_return"])
    return out


def aggregate_stage4_eval(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect Stage 4 eval returns (use_self True/False) for each variant.
    Looks for metrics with substrings like 'stage4'/'after_stage4' and 'self' vs 'no_self'.
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        m_self = first_present(
            sm,
            ["eval_after_stage4_self", "stage4_self"],
        ) or find_metric_by_substrings(sm, ["stage4", "self"])
        m_no_self = first_present(
            sm,
            ["eval_after_stage4_no_self", "stage4_no_self"],
        ) or find_metric_by_substrings(sm, ["stage4", "no", "self"])
        out.setdefault(variant, {"self": [], "no_self": []})
        if m_self and isinstance(m_self, dict) and "mean_return" in m_self:
            out[variant]["self"].append(m_self["mean_return"])
        if m_no_self and isinstance(m_no_self, dict) and "mean_return" in m_no_self:
            out[variant]["no_self"].append(m_no_self["mean_return"])
    return out


def aggregate_self_probe(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate self-model probe metrics across seeds per variant.
    Collects all float fields found in the first probe dict encountered per variant.
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    first_keys_printed = set()
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        probe = None
        for k, v in sm.items():
            if "self_model_probe" in k.lower() and isinstance(v, dict):
                probe = v
                break
        if not probe:
            continue
        if variant not in out:
            out[variant] = {}
        if variant not in first_keys_printed:
            print(f"\n[Debug] SelfModel probe keys for variant '{variant}': {list(probe.keys())}")
            first_keys_printed.add(variant)
        for key, val in probe.items():
            if isinstance(val, (int, float)):
                out[variant].setdefault(key, []).append(float(val))
    return out


def print_table(title: str, rows: List[Tuple[str, Optional[float], Optional[float], int]], markdown: bool = False):
    print(f"\n=== {title} ===")
    if markdown:
        print("| Variant | Mean | Std | n |")
        print("|---------|------|-----|---|")
        for variant, mean_v, std_v, n in rows:
            m = f"{mean_v:.3f}" if mean_v is not None else "nan"
            s = f"{std_v:.3f}" if std_v is not None else "nan"
            print(f"| {variant} | {m} | {s} | {n} |")
    else:
        print(f"{'variant':<15} {'mean':>10} {'std':>10} {'n':>5}")
        for variant, mean_v, std_v, n in rows:
            m = f"{mean_v:.3f}" if mean_v is not None else "nan"
            s = f"{std_v:.3f}" if std_v is not None else "nan"
            print(f"{variant:<15} {m:>10} {s:>10} {n:>5}")


def summarize_phase_c(entries: List[Dict[str, Any]]):
    rows_no_self, rows_self = compute_phase_c_rows(entries)
    print_table("Phase C (planner on, use_self=False)", rows_no_self, markdown=False)
    print_table("Phase C (planner on, use_self=False) [Markdown]", rows_no_self, markdown=True)
    print_table("Phase C (planner on, use_self=True)", rows_self, markdown=False)
    print_table("Phase C (planner on, use_self=True) [Markdown]", rows_self, markdown=True)


def summarize_stage4(entries: List[Dict[str, Any]]):
    rows_no_self, rows_self = compute_stage4_rows(entries)
    print_table("Stage 4 eval (planner on, use_self=False)", rows_no_self, markdown=False)
    print_table("Stage 4 eval (planner on, use_self=False) [Markdown]", rows_no_self, markdown=True)
    print_table("Stage 4 eval (planner on, use_self=True)", rows_self, markdown=False)
    print_table("Stage 4 eval (planner on, use_self=True) [Markdown]", rows_self, markdown=True)


def summarize_phase_c_online(entries: List[Dict[str, Any]]):
    rows_no_self, rows_self = compute_phase_c_online_rows(entries)
    print_table("Phase C online adaptation (planner on, use_self=False)", rows_no_self, markdown=False)
    print_table("Phase C online adaptation (planner on, use_self=False) [Markdown]", rows_no_self, markdown=True)
    print_table("Phase C online adaptation (planner on, use_self=True)", rows_self, markdown=False)
    print_table("Phase C online adaptation (planner on, use_self=True) [Markdown]", rows_self, markdown=True)


def summarize_probe(entries: List[Dict[str, Any]]):
    agg = aggregate_self_probe(entries)
    for variant, metrics in agg.items():
        print(f"\n=== SelfModel probe summary: {variant} ===")
        for key, vals in metrics.items():
            mean_v, std_v = mean_std(vals)
            m = f"{mean_v:.3f}" if mean_v is not None else "nan"
            s = f"{std_v:.3f}" if std_v is not None else "nan"
            print(f"{key:<30} mean={m} std={s} n={len(vals)}")


def summarize_self_model_improvement(entries: List[Dict[str, Any]]):
    """
    Optional summary of corr_return improvement from Stage 3a probe to Stage 3c probe.
    Skips silently if the new keys are absent.
    """
    deltas: Dict[str, List[float]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        probe_stage3 = None
        for key in ["self_model_probe_after_stage3", "self_model_probe_after_stage3a"]:
            v = sm.get(key)
            if isinstance(v, dict):
                probe_stage3 = v
                break
        probe_stage3c = sm.get("self_model_probe_after_stage3c")
        if not isinstance(probe_stage3, dict) or not isinstance(probe_stage3c, dict):
            continue
        corr_before = probe_stage3.get("corr_return")
        corr_after = probe_stage3c.get("corr_return")
        if isinstance(corr_before, (int, float)) and isinstance(corr_after, (int, float)):
            if math.isfinite(corr_before) and math.isfinite(corr_after):
                deltas.setdefault(variant, []).append(float(corr_after - corr_before))
    if not any(deltas.values()):
        print("\n[INFO] Stage 3c self-model improvement metrics not found; skipping section.")
        return

    rows: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant in _variant_order(deltas.keys()):
        vals = deltas.get(variant, [])
        mean_v, std_v = mean_std(vals)
        rows.append((variant, mean_v, std_v, len(vals)))
    print_table("SelfModel corr_return improvement (Stage 3 -> Stage 3c)", rows, markdown=False)
    print_table("SelfModel corr_return improvement (Stage 3 -> Stage 3c) [Markdown]", rows, markdown=True)


def _collect_trait_reflection_examples(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pick one non-empty trait_reflection_summary per variant (first found)."""
    per_variant: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        summary = sm.get("trait_reflection_summary")
        if not isinstance(summary, dict) or not summary:
            continue
        if variant not in per_variant:
            per_variant[variant] = summary
    return per_variant


def summarize_trait_reflection(entries: List[Dict[str, Any]]):
    """Print a compact self-report section using trait_reflection_summary."""
    per_variant = _collect_trait_reflection_examples(entries)
    print("\n=== Trait reflection self-report (example summaries) ===")
    if not per_variant:
        print("No trait_reflection_summary found in stage_metrics.")
        return
    for variant in _variant_order(per_variant.keys()):
        example = per_variant.get(variant)
        if not example:
            continue
        print(f"[Variant: {variant}]")
        total_updates = example.get("_total_updates", 0)
        if isinstance(total_updates, (int, float)):
            print(f"  total_updates: {int(total_updates)}")
        regimes = [k for k in example.keys() if not str(k).startswith("_")]
        for reg_name in sorted(regimes):
            reg = example.get(reg_name, {})
            if not isinstance(reg, dict):
                continue
            count = reg.get("count", 0)
            first_msg = reg.get("first", "")
            last_msg = reg.get("last", "")
            print(f"  - regime: {reg_name} (updates={count})")
            if first_msg:
                print(f"      first: {first_msg}")
            if last_msg and last_msg != first_msg:
                print(f"      last:  {last_msg}")
        print()


# ===== MiniGrid Stage 4 breakdowns =====

def _collect_minigrid_metadata(entries: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not entries:
        return None
    metas: List[List[Dict[str, Any]]] = []
    for e in entries:
        sm = e.get("stage_metrics", {}) or {}
        meta = sm.get("minigrid_task_metadata")
        if not isinstance(meta, list) or not meta:
            return None
        metas.append(meta)
    return metas[0] if metas else None


def _extract_per_task_returns(metric: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Extract per-task episode returns from an eval metric dict produced by Trainer.evaluate.
    Expected format:
      metric["per_task"][task_name]["returns"] -> list of floats
    Falls back to empty dict if absent or malformed.
    """
    per_task_raw = metric.get("per_task") if isinstance(metric, dict) else None
    if not isinstance(per_task_raw, dict):
        return {}
    out: Dict[str, List[float]] = {}
    for task_name, data in per_task_raw.items():
        vals: List[float] = []
        if isinstance(data, dict):
            raw = data.get("returns")
            if isinstance(raw, list):
                vals = [float(v) for v in raw if isinstance(v, (int, float))]
            elif isinstance(data.get("mean_return"), (int, float)) and isinstance(data.get("n_episodes"), (int, float)):
                # Reconstruct rough returns using mean if list missing.
                vals = [float(data["mean_return"])] * int(data.get("n_episodes", 0))
        elif isinstance(data, (int, float)):
            vals = [float(data)]
        if vals:
            out[task_name] = vals
    return out


def _collect_stage4_per_task(
    entries: List[Dict[str, Any]], metric_key: str, task_ids: List[str]
) -> Tuple[Dict[str, Dict[str, List[float]]], bool]:
    """
    Return variant -> task_id -> list[float] of returns for the given Stage 4 metric key.
    Also returns a flag indicating if any run lacked per_task data.
    """
    per_variant: Dict[str, Dict[str, List[float]]] = {}
    missing_data = False
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        metric = sm.get(metric_key, {})
        per_task = _extract_per_task_returns(metric)
        if not per_task:
            missing_data = True
            continue
        for tid in task_ids:
            vals = per_task.get(tid, [])
            if not vals:
                continue
            per_variant.setdefault(variant, {}).setdefault(tid, []).extend([float(v) for v in vals])
    return per_variant, missing_data


def _collect_stage4_per_category(
    per_task_stats: Dict[str, Dict[str, List[float]]],
    category_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Convert per-task returns into per-category returns by pooling all task returns within each category.
    Input: variant -> task_name -> list[float]
    Output: variant -> category -> list[float]
    """
    per_variant: Dict[str, Dict[str, List[float]]] = {}
    for variant, task_vals in per_task_stats.items():
        for cat, tasks in category_map.items():
            vals: List[float] = []
            for t in tasks:
                vals.extend(task_vals.get(t, []))
            if vals:
                per_variant.setdefault(variant, {}).setdefault(cat, []).extend([float(v) for v in vals])
    return per_variant


def _rows_with_norm(
    stats: Dict[str, Dict[str, List[float]]],
    keys: List[str],
    label_order: List[str],
) -> List[Tuple[str, str, Optional[float], Optional[float], Optional[float], int]]:
    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float], int]] = []
    max_mean: Dict[str, float] = {}
    for key in keys:
        for variant in label_order:
            vals = (stats.get(variant) or {}).get(key, [])
            if not vals:
                continue
            mean_v, _ = mean_std(vals)
            if mean_v is not None:
                max_mean[key] = max(max_mean.get(key, float("-inf")), mean_v)
    for key in keys:
        for variant in label_order:
            vals = (stats.get(variant) or {}).get(key, [])
            if not vals:
                continue
            mean_v, std_v = mean_std(vals)
            n = len(vals)
            if n <= 0:
                continue
            norm = None
            if mean_v is not None:
                denom = max(max_mean.get(key, 0.0), 1e-8)
                norm = mean_v / denom if denom > 0 else None
            rows.append((key, variant, mean_v, std_v, norm, n))
    return rows


def _print_table_with_norm(title: str, label: str, rows, markdown: bool = False):
    print(f"\n=== {title} ===")
    if markdown:
        print(f"| {label} | Variant | Mean | Std | Norm | n |")
        print(f"|{'-' * (len(label)+2)}|---------|------|-----|------|---|")
        for key, variant, mean_v, std_v, norm, n in rows:
            m = f"{mean_v:.3f}" if mean_v is not None else "nan"
            s = f"{std_v:.3f}" if std_v is not None else "nan"
            r = f"{norm:.3f}" if norm is not None else "nan"
            print(f"| {key} | {variant} | {m} | {s} | {r} | {n} |")
    else:
        print(f"{label:<20} {'variant':<15} {'mean':>10} {'std':>10} {'norm':>10} {'n':>5}")
        for key, variant, mean_v, std_v, norm, n in rows:
            m = f"{mean_v:.3f}" if mean_v is not None else "nan"
            s = f"{std_v:.3f}" if std_v is not None else "nan"
            r = f"{norm:.3f}" if norm is not None else "nan"
            print(f"{key:<20} {variant:<15} {m:>10} {s:>10} {r:>10} {n:>5}")


def print_debug_keys(entries: List[Dict[str, Any]]):
    if not entries:
        return
    sample = entries[0]
    sm = sample.get("stage_metrics", {}) or {}
    print("\n[Debug] Sample stage_metrics keys:")
    for k in sm.keys():
        print(f"  - {k}")


def compute_phase_c_rows(entries: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, Optional[float], Optional[float], int]], List[Tuple[str, Optional[float], Optional[float], int]]]:
    agg = aggregate_phase_c(entries)
    rows_no_self: List[Tuple[str, Optional[float], Optional[float], int]] = []
    rows_self: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant, vals in agg.items():
        mean_ns, std_ns = mean_std(vals["no_self"])
        mean_s, std_s = mean_std(vals["self"])
        rows_no_self.append((variant, mean_ns, std_ns, len(vals["no_self"])))
        rows_self.append((variant, mean_s, std_s, len(vals["self"])))
    return rows_no_self, rows_self


def compute_stage4_rows(entries: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, Optional[float], Optional[float], int]], List[Tuple[str, Optional[float], Optional[float], int]]]:
    agg = aggregate_stage4_eval(entries)
    rows_no_self: List[Tuple[str, Optional[float], Optional[float], int]] = []
    rows_self: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant, vals in agg.items():
        mean_ns, std_ns = mean_std(vals["no_self"])
        mean_s, std_s = mean_std(vals["self"])
        rows_no_self.append((variant, mean_ns, std_ns, len(vals["no_self"])))
        rows_self.append((variant, mean_s, std_s, len(vals["self"])))
    return rows_no_self, rows_self


def get_mean_return(metric: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metric, dict):
        return None
    val = metric.get("mean_return")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def aggregate_phase_c_online(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect Phase C online adaptation returns for each variant, split by use_self flag.
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        m_ns = sm.get("lifecycle_phaseC_online_no_self")
        m_s = sm.get("lifecycle_phaseC_online_self")
        out.setdefault(variant, {"self": [], "no_self": []})
        v_ns = get_mean_return(m_ns)
        v_s = get_mean_return(m_s)
        if v_ns is not None:
            out[variant]["no_self"].append(v_ns)
        if v_s is not None:
            out[variant]["self"].append(v_s)
    return out


def compute_phase_c_online_rows(entries: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, Optional[float], Optional[float], int]], List[Tuple[str, Optional[float], Optional[float], int]]]:
    agg = aggregate_phase_c_online(entries)
    rows_no_self: List[Tuple[str, Optional[float], Optional[float], int]] = []
    rows_self: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant, vals in agg.items():
        mean_ns, std_ns = mean_std(vals["no_self"])
        mean_s, std_s = mean_std(vals["self"])
        rows_no_self.append((variant, mean_ns, std_ns, len(vals["no_self"])))
        rows_self.append((variant, mean_s, std_s, len(vals["self"])))
    return rows_no_self, rows_self


def _lifelong_metric_from_stage(sm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not sm:
        return None
    for k, v in sm.items():
        if "lifelong" in k.lower() and isinstance(v, dict):
            return v
    return None


def _get_lifelong_block(sm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Retrieve the lifelong metrics block from a stage_metrics dict."""
    if not sm:
        return None
    if isinstance(sm.get("lifelong_eval"), dict):
        return sm.get("lifelong_eval")
    return _lifelong_metric_from_stage(sm)


def aggregate_lifelong(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, List[float]]], List[str], Dict[str, List[float]], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """
    Collect per-chapter returns and forgetting/adaptation stats for lifelong eval.
    Returns:
      chapter_returns: variant -> chapter_name -> list[float]
      regime_order: order of chapters observed (fallback default)
      forgetting: variant -> list[float]
      adaptation: variant -> {"R2": [...], "R3": [...]}
      forgetting_per_regime: variant -> regime -> list[float]
    """
    chapter_returns: Dict[str, Dict[str, List[float]]] = {}
    forgetting: Dict[str, List[float]] = {}
    adaptation: Dict[str, Dict[str, List[float]]] = {}
    regime_order: List[str] = []
    forgetting_per_regime: Dict[str, Dict[str, List[float]]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        ll = _get_lifelong_block(sm)
        if not isinstance(ll, dict):
            continue
        regimes = ll.get("lifelong_regimes")
        if not regime_order and isinstance(regimes, list):
            regime_order = [str(x) for x in regimes]
        per_ch = ll.get("lifelong_per_chapter", [])
        if not per_ch:
            continue
        chapter_returns.setdefault(variant, {})
        forgetting.setdefault(variant, [])
        adaptation.setdefault(variant, {"R2": [], "R3": []})
        forgetting_per_regime.setdefault(variant, {})
        for ch in per_ch:
            name = str(ch.get("regime", "chapter"))
            mr = ch.get("mean_return")
            if isinstance(mr, (int, float)):
                chapter_returns[variant].setdefault(name, []).append(float(mr))
        gap = ll.get("lifelong_forgetting_R1_gap")
        if isinstance(gap, (int, float)):
            forgetting[variant].append(float(gap))
        fg_full = ll.get("lifelong_forgetting", {}) or {}
        if isinstance(fg_full, dict):
            gap_map = fg_full.get("gap", {})
            if isinstance(gap_map, dict):
                vals = [v for v in gap_map.values() if isinstance(v, (int, float))]
                if vals:
                    forgetting[variant].append(float(np.mean(vals)))
                for reg_name, v in gap_map.items():
                    if isinstance(v, (int, float)):
                        forgetting_per_regime[variant].setdefault(str(reg_name), []).append(float(v))
        r2_delta = ll.get("lifelong_adaptation_R2_delta")
        r3_delta = ll.get("lifelong_adaptation_R3_delta")
        if isinstance(r2_delta, (int, float)):
            adaptation[variant]["R2"].append(float(r2_delta))
        if isinstance(r3_delta, (int, float)):
            adaptation[variant]["R3"].append(float(r3_delta))
    for v in EXPECTED_VARIANTS:
        chapter_returns.setdefault(v, {})
        forgetting.setdefault(v, [])
        adaptation.setdefault(v, {"R2": [], "R3": []})
        forgetting_per_regime.setdefault(v, {})
    if not regime_order:
        regime_order = ["R1", "R2", "R3", "R1_return"]
    return chapter_returns, regime_order, forgetting, adaptation, forgetting_per_regime


def aggregate_trait_movement(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]], List[str], List[str]]:
    """
    Collect trait movement metrics across regimes.
    Returns:
      trait_change: variant -> regime -> list[float]
      trait_dist_tail: variant -> regime -> list[float]
      regime_order: order of regimes observed
      variants: list of variants observed (unique)
    """
    trait_change: Dict[str, Dict[str, List[float]]] = {}
    trait_dist_tail: Dict[str, Dict[str, List[float]]] = {}
    regime_order: List[str] = []
    variants: List[str] = []
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        if variant not in variants:
            variants.append(variant)
        sm = e.get("stage_metrics", {}) or {}
        ll = _get_lifelong_block(sm)
        if not isinstance(ll, dict):
            continue
        per_ch = ll.get("lifelong_per_chapter", [])
        if per_ch and not regime_order:
            regime_order = [str(ch.get("regime", "chapter")) for ch in per_ch]
        for ch in per_ch or []:
            name = str(ch.get("regime", "chapter"))
            change = ch.get("trait_change_within_regime")
            dist_tail = ch.get("trait_dist_from_init_tail")
            if isinstance(change, (int, float)):
                trait_change.setdefault(variant, {}).setdefault(name, []).append(float(change))
            if isinstance(dist_tail, (int, float)):
                trait_dist_tail.setdefault(variant, {}).setdefault(name, []).append(float(dist_tail))
    if not regime_order:
        regime_order = ["R1", "R2", "R3", "R1_return"]
    for v in EXPECTED_VARIANTS:
        trait_change.setdefault(v, {})
        trait_dist_tail.setdefault(v, {})
    variants_ordered = _variant_order(trait_change.keys() or variants)
    return trait_change, trait_dist_tail, regime_order, list(variants_ordered)


def aggregate_trait_reflections(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate mean delta per trait (overall and per-regime) from structured trait_reflection_log.
    Returns: variant -> {"overall": {...}, "per_regime": {...}, "counts_by_regime": {...}, "total_updates": int}
    """
    raw: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        variant = e.get("agent_variant", "unknown")
        sm = e.get("stage_metrics", {}) or {}
        events = sm.get("trait_reflection_log", []) or []
        if not isinstance(events, list):
            continue
        data = raw.setdefault(
            variant,
            {
                "overall": {"survival": [], "food": [], "damage": [], "move": []},
                "per_regime": {},
                "counts_by_regime": {},
                "total_updates": 0,
            },
        )
        for ev in events:
            if not isinstance(ev, dict):
                continue
            delta = ev.get("delta_traits")
            if not isinstance(delta, dict):
                continue
            reg = str(ev.get("regime", "unknown"))
            data["counts_by_regime"][reg] = data["counts_by_regime"].get(reg, 0) + 1
            data["total_updates"] += 1
            reg_map = data["per_regime"].setdefault(
                reg, {"survival": [], "food": [], "damage": [], "move": []}
            )
            for trait in reg_map.keys():
                v = delta.get(trait)
                if isinstance(v, (int, float)):
                    reg_map[trait].append(float(v))
                    data["overall"][trait].append(float(v))

    out: Dict[str, Dict[str, Any]] = {}
    for variant, data in raw.items():
        overall_mean = {k: (float(np.mean(vals)) if vals else None) for k, vals in data["overall"].items()}
        per_regime_mean: Dict[str, Dict[str, float]] = {}
        for reg, trait_lists in data["per_regime"].items():
            per_regime_mean[reg] = {k: (float(np.mean(vals)) if vals else None) for k, vals in trait_lists.items()}
        out[variant] = {
            "overall": overall_mean,
            "per_regime": per_regime_mean,
            "counts_by_regime": data["counts_by_regime"],
            "total_updates": data["total_updates"],
        }
    return out
def _plot_bar(
    rows: List[Tuple[str, Optional[float], Optional[float], int]],
    title: str,
    ylabel: str,
    outfile: Path,
    variant_order: Optional[List[str]] = None,
) -> None:
    if variant_order:
        row_map = {v: (m, s, n) for v, m, s, n in rows}
        rows = []
        for v in variant_order:
            m, s, n = row_map.get(v, (None, None, 0))
            rows.append((v, m, s, n))
    valid = [(v, m, s) for v, m, s, n in rows if m is not None and n and n > 0]
    if not valid:
        print(f"[WARN] No data to plot for {title}")
        return
    variants, means, stds = zip(*valid)
    x = range(len(variants))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.85)
    plt.xticks(list(x), variants)
    plt.ylabel(ylabel)
    plt.title(title)
    for xi, m in zip(x, means):
        plt.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def summarize_lifelong(entries: List[Dict[str, Any]]):
    chapter_returns, regime_order, forgetting, adaptation, forget_detail = aggregate_lifelong(entries)
    if not chapter_returns:
        print("\n=== Lifelong eval: no data ===")
        return

    variant_order = _variant_order(chapter_returns.keys())
    for reg in regime_order:
        rows: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for variant in variant_order:
            vals = (chapter_returns.get(variant) or {}).get(reg, [])
            mean_v, std_v = mean_std(vals)
            rows.append((variant, mean_v, std_v, len(vals)))
        print_table(f"Lifelong {reg} mean_return", rows, markdown=False)
        print_table(f"Lifelong {reg} mean_return [Markdown]", rows, markdown=True)

    rows_gap: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant in variant_order:
        vals = forgetting.get(variant, [])
        mean_v, std_v = mean_std(vals)
        rows_gap.append((variant, mean_v, std_v, len(vals)))
    print_table("Lifelong forgetting gap (R1_return - R1)", rows_gap, markdown=False)
    print_table("Lifelong forgetting gap (R1_return - R1) [Markdown]", rows_gap, markdown=True)

    # per-regime gap breakdown (if available)
    regime_names = sorted({reg for regs in forget_detail.values() for reg in regs.keys()})
    for reg in regime_names:
        rows_reg: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for variant in variant_order:
            vals = (forget_detail.get(variant) or {}).get(reg, [])
            mean_v, std_v = mean_std(vals)
            rows_reg.append((variant, mean_v, std_v, len(vals)))
        if any(r[3] > 0 for r in rows_reg):
            print_table(f"Lifelong forgetting gap per-regime ({reg})", rows_reg, markdown=False)
            print_table(f"Lifelong forgetting gap per-regime ({reg}) [Markdown]", rows_reg, markdown=True)

    rows_adapt_r2: List[Tuple[str, Optional[float], Optional[float], int]] = []
    rows_adapt_r3: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant in variant_order:
        ad = adaptation.get(variant, {})
        mean_r2, std_r2 = mean_std(ad.get("R2", []))
        mean_r3, std_r3 = mean_std(ad.get("R3", []))
        rows_adapt_r2.append((variant, mean_r2, std_r2, len(ad.get("R2", []))))
        rows_adapt_r3.append((variant, mean_r3, std_r3, len(ad.get("R3", []))))
    print_table("Lifelong adaptation delta R2 (tail - head)", rows_adapt_r2, markdown=False)
    print_table("Lifelong adaptation delta R2 (tail - head) [Markdown]", rows_adapt_r2, markdown=True)
    print_table("Lifelong adaptation delta R3 (tail - head)", rows_adapt_r3, markdown=False)
    print_table("Lifelong adaptation delta R3 (tail - head) [Markdown]", rows_adapt_r3, markdown=True)


def summarize_lifelong_traits(entries: List[Dict[str, Any]]):
    trait_change, trait_dist_tail, regime_order, variants = aggregate_trait_movement(entries)
    if not trait_change and not trait_dist_tail:
        return
    for reg in regime_order:
        rows_change: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for variant in variants:
            vals = (trait_change.get(variant) or {}).get(reg, [])
            mean_v, std_v = mean_std(vals)
            rows_change.append((variant, mean_v, std_v, len(vals)))
        if any(r[3] > 0 for r in rows_change):
            print_table(f"Lifelong trait change within {reg}", rows_change, markdown=False)

        rows_dist: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for variant in variants:
            vals = (trait_dist_tail.get(variant) or {}).get(reg, [])
            mean_v, std_v = mean_std(vals)
            rows_dist.append((variant, mean_v, std_v, len(vals)))
        if any(r[3] > 0 for r in rows_dist):
            print_table(
                f"Lifelong trait distance from init (tail) {reg}",
                rows_dist,
                markdown=False,
            )


def summarize_trait_reflections(entries: List[Dict[str, Any]]):
    agg = aggregate_trait_reflections(entries)
    if not agg:
        print("\n=== No trait reflection events found ===")
        return
    print("\n=== Trait reflection mean deltas ===")

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "nan"
        try:
            return f"{float(v):+0.3f}"
        except Exception:
            return "nan"

    for variant in _variant_order(agg.keys()):
        data = agg.get(variant, {})
        total_updates = data.get("total_updates", 0)
        per_regime = data.get("per_regime", {}) or {}
        counts = data.get("counts_by_regime", {}) or {}
        overall = data.get("overall", {}) or {}
        print(f"[Variant: {variant}] updates={int(total_updates)}")
        if not per_regime:
            print("  no structured trait updates logged")
            continue
        for reg, deltas in per_regime.items():
            parts = []
            for trait in ["survival", "food", "damage", "move"]:
                v = deltas.get(trait)
                if v is None:
                    continue
                parts.append(f"Δ{trait}={_fmt(v)}")
            count = counts.get(reg, 0)
            if parts:
                suffix = f" over {int(count)} updates" if count else ""
                print(f"  {reg}: {', '.join(parts)}{suffix}")
        if any(val is not None for val in overall.values()):
            parts_overall = [
                f"Δ{trait}={_fmt(overall.get(trait))}"
                for trait in ["survival", "food", "damage", "move"]
                if overall.get(trait) is not None
            ]
            if parts_overall:
                print(f"  overall: {', '.join(parts_overall)}")


def make_lifelong_plots(entries: List[Dict[str, Any]]):
    chapter_returns, regime_order, forgetting, _, _forget_detail = aggregate_lifelong(entries)
    if not chapter_returns:
        print("[WARN] No lifelong data available for plotting.")
        return

    variant_order = _variant_order(chapter_returns.keys())
    for reg in regime_order:
        rows: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for variant in variant_order:
            vals = (chapter_returns.get(variant) or {}).get(reg, [])
            mean_v, std_v = mean_std(vals)
            rows.append((variant, mean_v, std_v, len(vals)))
        _plot_bar(
            rows,
            title=f"Lifelong {reg} mean return",
            ylabel="Mean return",
            outfile=FIG_DIR / f"lifelong_{reg}.png",
            variant_order=variant_order,
        )

    rows_gap: List[Tuple[str, Optional[float], Optional[float], int]] = []
    for variant in variant_order:
        vals = forgetting.get(variant, [])
        mean_v, std_v = mean_std(vals)
        rows_gap.append((variant, mean_v, std_v, len(vals)))
        _plot_bar(
            rows_gap,
            title="Lifelong forgetting gap (R1_return - R1)",
            ylabel="Return gap",
            outfile=FIG_DIR / "lifelong_forgetting_gap.png",
            variant_order=variant_order,
        )

    trait_change, trait_dist_tail, regime_order_tc, variants = aggregate_trait_movement(entries)
    if trait_change or trait_dist_tail:
        reg_change = "R2" if "R2" in regime_order_tc else regime_order_tc[0]
        rows_tc: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for v in variants:
            vals = (trait_change.get(v) or {}).get(reg_change, [])
            mean_v, std_v = mean_std(vals)
            rows_tc.append((v, mean_v, std_v, len(vals)))
        _plot_bar(
            rows_tc,
            title=f"Lifelong trait change within {reg_change}",
            ylabel="Trait delta (||tail-head||_2)",
            outfile=FIG_DIR / f"lifelong_trait_change_{reg_change}.png",
            variant_order=variants,
        )

        reg_dist = "R1_return" if "R1_return" in regime_order_tc else regime_order_tc[-1]
        rows_td: List[Tuple[str, Optional[float], Optional[float], int]] = []
        for v in variants:
            vals = (trait_dist_tail.get(v) or {}).get(reg_dist, [])
            mean_v, std_v = mean_std(vals)
            rows_td.append((v, mean_v, std_v, len(vals)))
        _plot_bar(
            rows_td,
            title=f"Trait distance from init (tail) {reg_dist}",
            ylabel="||theta_tail - theta0||_2",
            outfile=FIG_DIR / f"lifelong_trait_dist_{reg_dist}.png",
            variant_order=variants,
        )


def make_all_plots(entries: List[Dict[str, Any]]):
    rows_phase_c_ns, rows_phase_c_s = compute_phase_c_rows(entries)
    rows_stage4_ns, rows_stage4_s = compute_stage4_rows(entries)
    rows_phase_c_online_ns, rows_phase_c_online_s = compute_phase_c_online_rows(entries)

    _plot_bar(
        rows_phase_c_ns,
        title="Phase C (test only, planner on, use_self=False)",
        ylabel="Mean return (Phase C, test envs only)",
        outfile=FIG_DIR / "phaseC_test_useSelfFalse.png",
    )
    _plot_bar(
        rows_phase_c_s,
        title="Phase C (test only, planner on, use_self=True)",
        ylabel="Mean return (Phase C, test envs only)",
        outfile=FIG_DIR / "phaseC_test_useSelfTrue.png",
    )
    _plot_bar(
        rows_stage4_ns,
        title="Stage 4 eval (mixed train+test, planner on, use_self=False)",
        ylabel="Mean return (Stage 4 eval)",
        outfile=FIG_DIR / "stage4_eval_useSelfFalse.png",
    )
    _plot_bar(
        rows_stage4_s,
        title="Stage 4 eval (mixed train+test, planner on, use_self=True)",
        ylabel="Mean return (Stage 4 eval)",
        outfile=FIG_DIR / "stage4_eval_useSelfTrue.png",
    )
    _plot_bar(
        rows_phase_c_online_ns,
        title="Phase C online adaptation (test only, planner on, use_self=False)",
        ylabel="Mean return (Phase C online)",
        outfile=FIG_DIR / "phaseC_online_useSelfFalse.png",
    )
    _plot_bar(
        rows_phase_c_online_s,
        title="Phase C online adaptation (test only, planner on, use_self=True)",
        ylabel="Mean return (Phase C online)",
        outfile=FIG_DIR / "phaseC_online_useSelfTrue.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results and optionally make plots.")
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate bar plots into figures/ (requires matplotlib).",
    )
    parser.add_argument(
        "--lifelong",
        action="store_true",
        help="Use lifelong sweep file (default: stage4 lifecycle sweep).",
    )
    parser.add_argument(
        "--sweep-path",
        type=str,
        default=None,
        help="Optional explicit path to a sweep JSON file.",
    )
    args = parser.parse_args()

    sweep_path = (
        Path(args.sweep_path)
        if args.sweep_path
        else (DEFAULT_LIFELONG_SWEEP if args.lifelong else DEFAULT_STAGE4_SWEEP)
    )

    raw_entries = load_entries(sweep_path)
    entries: List[Dict[str, Any]] = []
    for idx, e in enumerate(raw_entries):
        if not isinstance(e, dict):
            print(f"[WARN] Skipping entry {idx}: not a dict")
            continue
        sm = e.get("stage_metrics")
        if not isinstance(sm, dict):
            print(f"[WARN] Skipping entry {idx}: missing stage_metrics")
            continue
        entries.append(e)
    print(f"[INFO] Loaded {len(entries)} usable entries from {sweep_path}")
    print_debug_keys(entries)

    summarize_phase_c(entries)
    summarize_phase_c_online(entries)
    summarize_stage4(entries)
    summarize_probe(entries)
    summarize_self_model_improvement(entries)
    has_lifelong = any(
        _lifelong_metric_from_stage(e.get("stage_metrics", {}) or {}) is not None
        or isinstance((e.get("stage_metrics", {}) or {}).get("lifelong_eval"), dict)
        for e in entries
    )
    if has_lifelong:
        summarize_lifelong(entries)
        summarize_lifelong_traits(entries)
        summarize_trait_reflections(entries)
    else:
        summarize_trait_reflections(entries)

    task_meta = _collect_minigrid_metadata(entries)
    if task_meta:
        task_names_list = [str(t.get("name", t.get("task_id", ""))) for t in task_meta if t.get("name") or t.get("task_id")]
        task_ids = task_names_list
        task_names = {str(t.get("name", t.get("task_id", ""))): str(t.get("name", t.get("task_id", ""))) for t in task_meta}
        category_map: Dict[str, List[str]] = {}
        for t in task_meta:
            tid = str(t.get("name", t.get("task_id", "")))
            cat = str(t.get("category", "unknown"))
            category_map.setdefault(cat, []).append(tid)

        # Stage 4 per-task returns (use_self True/False)
        per_task_self, missing_self = _collect_stage4_per_task(entries, "eval_after_stage4_self", task_ids)
        per_task_no_self, missing_no_self = _collect_stage4_per_task(entries, "eval_after_stage4_no_self", task_ids)
        missing_any = missing_self or missing_no_self
        if missing_any:
            print("[WARN] MiniGrid Stage 4 per-task data missing in some runs; skipping those for per-task/category breakdown.")
        if not per_task_self and not per_task_no_self:
            print("[WARN] MiniGrid per-task returns not found; skipping per-task/category Stage 4 breakdown.")
        else:

            # Per-category returns by averaging across tasks per run
            per_cat_self = _collect_stage4_per_category(per_task_self, category_map)
            per_cat_no_self = _collect_stage4_per_category(per_task_no_self, category_map)

            variant_order = _variant_order(set(per_task_self.keys()) | set(per_task_no_self.keys()))
            category_order = list(category_map.keys())
            task_id_to_label = {tid: task_names.get(tid, tid) for tid in task_ids}

            # Categories
            rows_cat_no_self = _rows_with_norm(per_cat_no_self, category_order, variant_order)
            rows_cat_self = _rows_with_norm(per_cat_self, category_order, variant_order)
            _print_table_with_norm(
                "Stage 4 eval by category (planner on, use_self=False)",
                "Category",
                rows_cat_no_self,
                markdown=False,
            )
            _print_table_with_norm(
                "Stage 4 eval by category (planner on, use_self=False) [Markdown]",
                "Category",
                rows_cat_no_self,
                markdown=True,
            )
            _print_table_with_norm(
                "Stage 4 eval by category (planner on, use_self=True)",
                "Category",
                rows_cat_self,
                markdown=False,
            )
            _print_table_with_norm(
                "Stage 4 eval by category (planner on, use_self=True) [Markdown]",
                "Category",
                rows_cat_self,
                markdown=True,
            )

            # Tasks
            def _remap_task_rows(rows):
                remapped = []
                for key, variant, mean_v, std_v, norm, n in rows:
                    label = task_id_to_label.get(key, key)
                    remapped.append((label, variant, mean_v, std_v, norm, n))
                return remapped

            rows_task_no_self = _remap_task_rows(_rows_with_norm(per_task_no_self, task_ids, variant_order))
            rows_task_self = _remap_task_rows(_rows_with_norm(per_task_self, task_ids, variant_order))
            _print_table_with_norm(
                "Stage 4 eval by task (planner on, use_self=False)",
                "Task",
                rows_task_no_self,
                markdown=False,
            )
            _print_table_with_norm(
                "Stage 4 eval by task (planner on, use_self=False) [Markdown]",
                "Task",
                rows_task_no_self,
                markdown=True,
            )
            _print_table_with_norm(
                "Stage 4 eval by task (planner on, use_self=True)",
                "Task",
                rows_task_self,
                markdown=False,
            )
            _print_table_with_norm(
                "Stage 4 eval by task (planner on, use_self=True) [Markdown]",
                "Task",
                rows_task_self,
                markdown=True,
            )

        # Battery description (existing block)
        by_category: Dict[str, List[str]] = {}
        for item in task_meta:
            cat = str(item.get("category", "unknown"))
            name = str(item.get("name", f"task_{item.get('env_id', len(by_category))}"))
            by_category.setdefault(cat, []).append(name)
        print("\n=== MiniGrid task battery (from metadata) ===")
        preferred_order = ["food", "survival", "danger"]
        for cat in preferred_order + [c for c in sorted(by_category.keys()) if c not in preferred_order]:
            names = by_category.get(cat, [])
            if not names:
                continue
            print(f"category: {cat} ({len(names)} tasks)")
            for n in names:
                print(f"  - {n}")
    else:
        print("\n[WARN] MiniGrid category breakdown skipped (missing or malformed minigrid_task_metadata)")

    if args.make_plots:
        make_all_plots(entries)
        if has_lifelong:
            make_lifelong_plots(entries)


if __name__ == "__main__":
    main()
