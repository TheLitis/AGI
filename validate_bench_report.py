#!/usr/bin/env python
"""
Validate AGI bench reports for schema/gate/manifest consistency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_GATES = ("gate0", "gate1", "gate2", "gate3", "gate4")
REQUIRED_CAPABILITIES = (
    "generalization_score",
    "sample_efficiency_score",
    "robustness_score",
    "tool_workflow_score",
)
DEFAULT_REQUIRED_SUITES = ("long_horizon", "core", "tools", "language", "social", "lifelong", "safety")
REQUIRED_MANIFEST_KEYS = ("config_hash", "seed_list", "seed_count", "git_commit", "suite", "environment")


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


def validate_report(
    report: Dict[str, Any],
    *,
    require_schema: str,
    required_suites: List[str],
    expected_gates: Dict[str, str],
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
        if "ci" not in suite:
            errors.append(f"suites[{idx}] missing key 'ci'")
        if "metrics" not in suite:
            errors.append(f"suites[{idx}] missing key 'metrics'")
        if "per_env" not in suite:
            errors.append(f"suites[{idx}] missing key 'per_env'")

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

    errors = validate_report(
        data,
        require_schema=str(args.require_schema),
        required_suites=required_suites,
        expected_gates=expected_gates,
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
