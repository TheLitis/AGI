#!/usr/bin/env python
"""
Promote a validated milestone bench report to canonical report path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_bench_report  # noqa: E402


DEFAULT_CANONICAL = "reports/agi_v1.quick.seed01234.safetygate_v1.json"


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"failed to parse JSON '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object in '{path}'")
    return data


def _validate_report(data: dict, src: Path) -> None:
    errors = validate_bench_report.validate_report(
        data,
        require_schema="0.2",
        required_suites=list(validate_bench_report.DEFAULT_REQUIRED_SUITES),
        expected_gates={},
    )
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(f"validation failed for '{src}':\n  - {joined}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="Path to source milestone report JSON.")
    parser.add_argument(
        "--dst",
        type=str,
        default=DEFAULT_CANONICAL,
        help=f"Destination canonical report path (default: {DEFAULT_CANONICAL}).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate source report with validate_bench_report before promotion.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"[ERR] source report not found: {src}")
        return 2

    try:
        data = _load_json(src)
        if bool(args.validate):
            _validate_report(data, src)
    except Exception as exc:
        print(f"[ERR] {exc}")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] promoted {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
