#!/usr/bin/env python
"""
Validate AGI bench reports for schema/gate/manifest consistency.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REQUIRED_GATES = ("gate0", "gate1", "gate2", "gate3", "gate4")
ALLOWED_GATE_VALUES = ("pass", "fail", "na")
REQUIRED_CAPABILITIES = (
    "generalization_score",
    "sample_efficiency_score",
    "robustness_score",
    "tool_workflow_score",
)
DEFAULT_REQUIRED_SUITES = ("long_horizon", "core", "tools", "language", "social", "lifelong", "safety")
STRICT_REQUIRED_SUITES = DEFAULT_REQUIRED_SUITES + ("safety_ood",)
REQUIRED_MANIFEST_KEYS = ("config_hash", "seed_list", "seed_count", "git_commit", "suite", "environment")
REQUIRED_SAFETY_METRICS = ("constraint_compliance", "catastrophic_fail_rate")
STRICT_SUITE_METRICS = {
    "long_horizon": ("horizon_utilization",),
    "core": ("mean_return", "test_mean_return"),
    "tools": ("pass_rate_unmasked", "mean_steps_to_pass_unmasked", "invalid_action_rate", "recovery_rate"),
    "language": ("pass_rate", "causal_drop"),
    "social": ("success_rate", "transfer_rate"),
    "lifelong": ("forgetting_gap", "forward_transfer", "env_family_coverage"),
    "safety": REQUIRED_SAFETY_METRICS,
    "safety_ood": REQUIRED_SAFETY_METRICS,
}
UNIT_RATE_METRICS = {
    "pass_rate_unmasked",
    "invalid_action_rate",
    "recovery_rate",
    "pass_rate",
    "causal_drop",
    "success_rate",
    "transfer_rate",
    "env_family_coverage",
    "constraint_compliance",
    "catastrophic_fail_rate",
}


def _split_csv(values: List[str]) -> List[str]:
    out: List[str] = []
    for item in values:
        for tok in str(item).split(","):
            tok = tok.strip()
            if tok:
                out.append(tok)
    return out


def _parse_expected_gate(values: List[str]) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    for item in values:
        part = str(item).strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --expect-gate value '{part}', expected key=value")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid --expect-gate value '{part}', expected key=value")
        expected[key] = value
    return expected


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _suite_by_name(report: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    suites = report.get("suites", [])
    if not isinstance(suites, list):
        return None
    for suite in suites:
        if isinstance(suite, dict) and str(suite.get("name")) == str(name):
            return suite
    return None


def _observed_ok_seeds(suite: Dict[str, Any]) -> List[int]:
    seeds: List[int] = []
    run_cache = suite.get("run_cache", [])
    if not isinstance(run_cache, list):
        return seeds
    for record in run_cache:
        if not isinstance(record, dict) or str(record.get("status")) != "ok":
            continue
        seed = record.get("seed")
        if isinstance(seed, int):
            seeds.append(int(seed))
    return sorted(set(seeds))


def _append_strict_metric_errors(errors: List[str], suite: Dict[str, Any], required_metrics: Iterable[str]) -> None:
    name = str(suite.get("name"))
    metrics = suite.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append(f"{name} suite metrics is missing or not an object")
        return
    for key in required_metrics:
        if key not in metrics:
            errors.append(f"{name} suite metrics missing key '{key}'")
            continue
        value = metrics.get(key)
        if not _is_number(value):
            errors.append(f"{name} suite metric '{key}' must be numeric")
            continue
        if key in UNIT_RATE_METRICS:
            fv = float(value)
            if fv < 0.0 or fv > 1.0:
                errors.append(f"{name} suite metric '{key}' must be in [0,1], got {value}")


def _validate_recomputed_gates(report: Dict[str, Any]) -> List[str]:
    try:
        import bench
    except Exception as exc:
        return [f"failed to import bench for gate recomputation: {exc}"]

    expected = copy.deepcopy(report)
    try:
        bench._refresh_overall(expected)  # type: ignore[attr-defined]
    except Exception as exc:
        return [f"failed to recompute gates from report metrics: {exc}"]

    stored_gates = ((report.get("overall") or {}).get("gates") or {})
    recomputed_gates = ((expected.get("overall") or {}).get("gates") or {})
    errors: List[str] = []
    for gate in REQUIRED_GATES:
        if stored_gates.get(gate) != recomputed_gates.get(gate):
            errors.append(
                f"overall.gates.{gate} disagrees with recomputed value: "
                f"stored={stored_gates.get(gate)!r} recomputed={recomputed_gates.get(gate)!r}"
            )
    return errors


def validate_report(
    report: Dict[str, Any],
    *,
    require_schema: str,
    required_suites: List[str],
    expected_gates: Dict[str, str],
    strict_acceptance: bool = False,
) -> List[str]:
    errors: List[str] = []

    schema_version = str(report.get("schema_version"))
    if schema_version != str(require_schema):
        errors.append(f"schema_version expected '{require_schema}' but got '{schema_version}'")

    meta = report.get("meta", {})
    if not isinstance(meta, dict):
        errors.append("meta is missing or not an object")
        meta = {}
    run_manifest = meta.get("run_manifest", {})
    if not isinstance(run_manifest, dict):
        errors.append("meta.run_manifest is missing or not an object")
        run_manifest = {}
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in run_manifest:
            errors.append(f"meta.run_manifest missing key '{key}'")

    overall = report.get("overall", {})
    if not isinstance(overall, dict):
        errors.append("overall is missing or not an object")
        overall = {}
    gates = overall.get("gates", {})
    if not isinstance(gates, dict):
        errors.append("overall.gates is missing or not an object")
        gates = {}
    for gate in REQUIRED_GATES:
        if gate not in gates:
            errors.append(f"overall.gates missing key '{gate}'")
            continue
        value = str(gates.get(gate))
        if value not in ALLOWED_GATE_VALUES:
            errors.append(f"overall.gates.{gate} has invalid value '{value}'")
    capabilities = overall.get("capabilities", {})
    if not isinstance(capabilities, dict):
        errors.append("overall.capabilities is missing or not an object")
        capabilities = {}
    for key in REQUIRED_CAPABILITIES:
        if key not in capabilities:
            errors.append(f"overall.capabilities missing key '{key}'")
    if "confidence" not in overall:
        errors.append("overall.confidence is missing")

    for gate, expected in expected_gates.items():
        actual = gates.get(gate)
        if str(actual) != str(expected):
            errors.append(f"overall.gates.{gate} expected '{expected}' but got '{actual}'")

    suites = report.get("suites", [])
    if not isinstance(suites, list):
        errors.append("suites is missing or not a list")
        suites = []
    suite_names = [str(s.get("name")) for s in suites if isinstance(s, dict)]
    missing_suites = [name for name in required_suites if name not in suite_names]
    if missing_suites:
        errors.append(f"missing required suites: {', '.join(missing_suites)}")
    for idx, suite in enumerate(suites):
        if not isinstance(suite, dict):
            errors.append(f"suites[{idx}] is not an object")
            continue
        suite_name = str(suite.get("name", f"#{idx}"))
        status = suite.get("status")
        if strict_acceptance:
            if status != "ok":
                errors.append(f"{suite_name} suite status expected 'ok' but got '{status}'")
            score = suite.get("score")
            if not _is_number(score):
                errors.append(f"{suite_name} suite score must be numeric in strict acceptance mode")
            elif float(score) < 0.0 or float(score) > 1.0:
                errors.append(f"{suite_name} suite score must be in [0,1], got {score}")
        if "ci" not in suite:
            errors.append(f"suites[{idx}] missing key 'ci'")
        if "metrics" not in suite:
            errors.append(f"suites[{idx}] missing key 'metrics'")
        if "per_env" not in suite:
            errors.append(f"suites[{idx}] missing key 'per_env'")

    if "safety" in required_suites:
        safety_suite = None
        for suite in suites:
            if isinstance(suite, dict) and str(suite.get("name")) == "safety":
                safety_suite = suite
                break
        if safety_suite is None:
            errors.append("missing required suites: safety")
        else:
            metrics = safety_suite.get("metrics", {})
            if not isinstance(metrics, dict):
                errors.append("safety suite metrics is missing or not an object")
                metrics = {}
            for key in REQUIRED_SAFETY_METRICS:
                if key not in metrics:
                    errors.append(f"safety suite metrics missing key '{key}'")
                    continue
                value = metrics.get(key)
                if not isinstance(value, (int, float)):
                    errors.append(f"safety suite metric '{key}' must be numeric")
                    continue
                fv = float(value)
                if fv < 0.0 or fv > 1.0:
                    errors.append(f"safety suite metric '{key}' must be in [0,1], got {value}")

    if strict_acceptance:
        for suite_name, metric_keys in STRICT_SUITE_METRICS.items():
            if suite_name not in required_suites:
                continue
            suite = _suite_by_name(report, suite_name)
            if suite is None:
                continue
            _append_strict_metric_errors(errors, suite, metric_keys)

        meta_seed_list = report.get("meta", {}).get("seed_list", [])
        if isinstance(meta_seed_list, list) and all(isinstance(x, int) for x in meta_seed_list):
            expected_seeds = sorted(set(int(x) for x in meta_seed_list))
            if expected_seeds:
                for suite_name in required_suites:
                    suite = _suite_by_name(report, suite_name)
                    if suite is None:
                        continue
                    observed = _observed_ok_seeds(suite)
                    # Some finalized reports intentionally drop run_cache. When it
                    # exists, enforce that it supports the declared seed coverage.
                    if observed and observed != expected_seeds:
                        errors.append(
                            f"{suite_name} suite run_cache seed coverage mismatch: "
                            f"expected={expected_seeds} observed={observed}"
                        )

        ll_suite = _suite_by_name(report, "lifelong")
        if isinstance(ll_suite, dict):
            ll_metrics = ll_suite.get("metrics", {})
            if isinstance(ll_metrics, dict):
                coverage = ll_metrics.get("env_family_coverage")
                if _is_number(coverage) and float(coverage) < 1.0:
                    errors.append(
                        f"lifelong suite env_family_coverage must be 1.0 in strict acceptance mode, got {coverage}"
                    )

        errors.extend(_validate_recomputed_gates(report))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AGI bench report format and key gate fields.")
    parser.add_argument("--report", type=str, required=True, help="Path to report JSON.")
    parser.add_argument("--require-schema", type=str, default="0.2", help="Required schema version.")
    parser.add_argument(
        "--require-suites",
        type=str,
        nargs="*",
        default=list(DEFAULT_REQUIRED_SUITES),
        help="Required suite names (CSV or repeated args).",
    )
    parser.add_argument(
        "--expect-gate",
        type=str,
        nargs="*",
        default=[],
        help="Expected gate values as key=value pairs, e.g. gate2=pass gate3=fail.",
    )
    parser.add_argument(
        "--strict-acceptance",
        action="store_true",
        help="Enforce acceptance-grade invariants: safety_ood, ok suites, typed metrics, and recomputed gates.",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[ERR] report not found: {report_path}")
        return 2

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERR] failed to read JSON: {exc}")
        return 2

    try:
        expected_gates = _parse_expected_gate(_split_csv(args.expect_gate))
    except ValueError as exc:
        print(f"[ERR] {exc}")
        return 2

    required_suites = _split_csv(args.require_suites)
    if not required_suites:
        required_suites = list(DEFAULT_REQUIRED_SUITES)
    if args.strict_acceptance:
        required_suites = list(dict.fromkeys(required_suites + list(STRICT_REQUIRED_SUITES)))

    errors = validate_report(
        data,
        require_schema=str(args.require_schema),
        required_suites=required_suites,
        expected_gates=expected_gates,
        strict_acceptance=bool(args.strict_acceptance),
    )
    if errors:
        print(f"[ERR] validation failed for {report_path}")
        for err in errors:
            print(f"  - {err}")
        return 1

    gates = ((data.get("overall") or {}).get("gates") or {})
    print(
        f"[OK] report={report_path} schema={data.get('schema_version')} "
        f"gate2={gates.get('gate2')} gate3={gates.get('gate3')} gate4={gates.get('gate4')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
