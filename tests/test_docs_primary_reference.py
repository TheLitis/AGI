import json
import re
from pathlib import Path

import validate_bench_report


REQUIRED_SUITES = ["long_horizon", "core", "tools", "language", "social", "lifelong", "safety"]


def _extract_backtick_path(path: Path, marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if marker in line:
            match = re.search(r"`([^`]+)`", line)
            assert match is not None, f"missing backtick path in line: {line}"
            return match.group(1).strip()
    raise AssertionError(f"marker not found: {marker} in {path}")


def _extract_checklist_reference(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Produce .* reference report:\s*`([^`]+)`", text)
    assert match is not None, "missing 'Produce ... reference report:' marker in CHECKLIST.md"
    return match.group(1).strip()


def test_docs_primary_reference_is_consistent_and_validated():
    repo_root = Path(__file__).resolve().parents[1]
    roadmap_path = repo_root / "ROADMAP.md"
    roadmap_v2_path = repo_root / "ROADMAP_v2.md"
    checklist_path = repo_root / "CHECKLIST.md"

    roadmap_ref = _extract_backtick_path(roadmap_path, "Primary reference:")
    roadmap_v2_ref = _extract_backtick_path(roadmap_v2_path, "Primary reference report:")
    checklist_ref = _extract_checklist_reference(checklist_path)

    assert roadmap_ref == roadmap_v2_ref == checklist_ref

    report_path = repo_root / roadmap_ref
    assert report_path.exists(), f"reference report not found: {report_path}"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    errors = validate_bench_report.validate_report(
        data,
        require_schema="0.2",
        required_suites=REQUIRED_SUITES,
        expected_gates={},
    )
    assert errors == [], "\n".join(errors)
