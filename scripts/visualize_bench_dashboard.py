#!/usr/bin/env python
"""
Build a static HTML dashboard from bench report JSON files.

The script is intentionally dependency-free so it can run in CI and on fresh
workstations. It does not launch experiments and only reads existing artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_report(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Report root is not an object: {path}")
    data["_source_path"] = str(path)
    return data


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _gate_class(value: Any) -> str:
    value_s = str(value).lower()
    if value_s == "pass":
        return "pass"
    if value_s == "fail":
        return "fail"
    return "na"


def _bar(value: Any, *, max_value: float = 1.0) -> str:
    if not isinstance(value, (int, float)) or max_value <= 0:
        return '<div class="bar empty"><span></span></div>'
    pct = max(0.0, min(100.0, 100.0 * float(value) / float(max_value)))
    return f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'


def _iter_suites(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    suites = report.get("suites", [])
    if isinstance(suites, list):
        for suite in suites:
            if isinstance(suite, dict):
                yield suite


def _suite_rows(report: Dict[str, Any]) -> str:
    rows: List[str] = []
    for suite in _iter_suites(report):
        metrics = suite.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        metric_bits = []
        for key in sorted(metrics):
            value = metrics.get(key)
            if isinstance(value, (dict, list)):
                continue
            metric_bits.append(f"<code>{html.escape(str(key))}</code>: {_fmt(value)}")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(suite.get('name', 'unknown')))}</td>"
            f"<td><span class=\"pill {_gate_class(suite.get('status'))}\">{html.escape(str(suite.get('status', 'n/a')))}</span></td>"
            f"<td>{_fmt(suite.get('score'))}{_bar(suite.get('score'))}</td>"
            f"<td>{'<br>'.join(metric_bits) if metric_bits else 'n/a'}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _report_summary(report: Dict[str, Any]) -> str:
    source = html.escape(str(report.get("_source_path", "unknown")))
    overall = report.get("overall", {}) if isinstance(report.get("overall"), dict) else {}
    gates = overall.get("gates", {}) if isinstance(overall.get("gates"), dict) else {}
    caps = overall.get("capabilities", {}) if isinstance(overall.get("capabilities"), dict) else {}
    gate_cards = []
    for gate in ("gate0", "gate1", "gate2", "gate3", "gate4"):
        value = gates.get(gate, "n/a")
        gate_cards.append(
            f'<div class="gate {_gate_class(value)}"><b>{gate}</b><span>{html.escape(str(value))}</span></div>'
        )
    cap_rows = []
    for key in ("generalization_score", "sample_efficiency_score", "robustness_score", "tool_workflow_score"):
        cap_rows.append(f"<tr><td>{html.escape(key)}</td><td>{_fmt(caps.get(key))}{_bar(caps.get(key))}</td></tr>")
    return f"""
<section class="report">
  <h2>{source}</h2>
  <div class="summary">
    <div><div class="label">AGI score</div><div class="big">{_fmt(overall.get('agi_score'))}</div>{_bar(overall.get('agi_score'))}</div>
    <div><div class="label">Confidence</div><div class="big">{_fmt(overall.get('confidence'))}</div>{_bar(overall.get('confidence'))}</div>
    <div class="gates">{''.join(gate_cards)}</div>
  </div>
  <h3>Capabilities</h3>
  <table><tbody>{''.join(cap_rows)}</tbody></table>
  <h3>Suites</h3>
  <table>
    <thead><tr><th>Suite</th><th>Status</th><th>Score</th><th>Metrics</th></tr></thead>
    <tbody>{_suite_rows(report)}</tbody>
  </table>
  {_lifelong_board(report)}
</section>
"""


def _lifelong_chapters(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    for suite in _iter_suites(report):
        if suite.get("name") != "lifelong":
            continue
        run_cache = suite.get("run_cache", [])
        if not isinstance(run_cache, list):
            continue
        for record in run_cache:
            if not isinstance(record, dict):
                continue
            result = record.get("result", {})
            if not isinstance(result, dict):
                continue
            stage_metrics = result.get("stage_metrics", {})
            if not isinstance(stage_metrics, dict):
                continue
            ll = stage_metrics.get("lifelong_eval", {})
            if not isinstance(ll, dict):
                continue
            for chapter in ll.get("lifelong_per_chapter", []) or []:
                if not isinstance(chapter, dict):
                    continue
                row = dict(chapter)
                row["_seed"] = record.get("seed")
                case = record.get("case")
                row["_case"] = case.get("name") if isinstance(case, dict) else case
                chapters.append(row)
    return chapters


def _lifelong_board(report: Dict[str, Any]) -> str:
    chapters = _lifelong_chapters(report)
    if not chapters:
        return ""
    rows: List[str] = []
    for ch in chapters[:120]:
        planner = ch.get("planner_debug", {})
        if not isinstance(planner, dict):
            planner = {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(ch.get('_case', 'n/a')))}</td>"
            f"<td>{_fmt(ch.get('_seed'), digits=0)}</td>"
            f"<td>{html.escape(str(ch.get('regime', 'n/a')))}</td>"
            f"<td>{_fmt(ch.get('mean_return'))}{_bar(ch.get('mean_return'), max_value=100.0)}</td>"
            f"<td>{_fmt(ch.get('mean_damage'))}</td>"
            f"<td>{_fmt(ch.get('trait_change_norm'))}</td>"
            f"<td>{_fmt(planner.get('planner_override_rate'))}</td>"
            f"<td>{html.escape(json.dumps(ch.get('scenario_counts', {}), ensure_ascii=False))}</td>"
            "</tr>"
        )
    overflow = "" if len(chapters) <= 120 else f"<p>Showing first 120 of {len(chapters)} chapters.</p>"
    return f"""
  <h3>Lifelong Chapter Board</h3>
  {overflow}
  <table>
    <thead><tr><th>Case</th><th>Seed</th><th>Regime</th><th>Return</th><th>Damage</th><th>Trait delta</th><th>Planner override</th><th>Scenarios</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
"""


def _render_html(reports: List[Dict[str, Any]]) -> str:
    body = "\n".join(_report_summary(report) for report in reports)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AGI Bench Dashboard</title>
<style>
:root {{ --bg:#0f1417; --panel:#182126; --ink:#e7f1f3; --muted:#8da1a8; --ok:#46d369; --bad:#ff5d5d; --na:#87919a; --line:#2b3a40; --accent:#6ec6ff; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:28px; background:radial-gradient(circle at top left,#18313b,#0f1417 45%); color:var(--ink); font:14px/1.45 'Segoe UI', sans-serif; }}
h1 {{ margin:0 0 18px; font-size:32px; letter-spacing:.02em; }}
h2 {{ margin:0 0 18px; font-size:18px; color:#d7eef7; word-break:break-all; }}
h3 {{ margin:24px 0 10px; color:#d7eef7; }}
.report {{ background:rgba(24,33,38,.92); border:1px solid var(--line); border-radius:18px; padding:20px; margin:0 0 22px; box-shadow:0 16px 50px rgba(0,0,0,.25); }}
.summary {{ display:grid; grid-template-columns: 180px 180px 1fr; gap:18px; align-items:stretch; }}
.label {{ color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:.12em; }}
.big {{ font-size:30px; font-weight:700; margin:4px 0; }}
.gates {{ display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; }}
.gate {{ min-width:96px; padding:10px 12px; border-radius:12px; background:#11191d; border:1px solid var(--line); }}
.gate b {{ display:block; color:var(--muted); }}
.gate span {{ font-size:20px; font-weight:700; }}
.pass span,.pill.pass {{ color:var(--ok); }}
.fail span,.pill.fail {{ color:var(--bad); }}
.na span,.pill.na {{ color:var(--na); }}
.pill {{ display:inline-block; padding:3px 9px; border:1px solid var(--line); border-radius:999px; background:#11191d; }}
table {{ width:100%; border-collapse:collapse; border:1px solid var(--line); overflow:hidden; border-radius:12px; }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; }}
th {{ color:#b8c8ce; background:#11191d; }}
code {{ color:#cbeafe; }}
.bar {{ height:7px; background:#11191d; border-radius:999px; overflow:hidden; margin-top:5px; }}
.bar span {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),var(--ok)); }}
.bar.empty span {{ width:0; }}
@media (max-width: 900px) {{ body {{ padding:14px; }} .summary {{ grid-template-columns:1fr; }} table {{ font-size:12px; }} }}
</style>
</head>
<body>
<h1>AGI Bench Dashboard</h1>
{body}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a static HTML bench dashboard.")
    parser.add_argument("--report", action="append", required=True, help="Report JSON path. Repeat for comparisons.")
    parser.add_argument("--out", default="reports/visualizations/dashboard.html", help="Output HTML path.")
    args = parser.parse_args()

    reports = [_load_report(Path(path)) for path in args.report]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(reports), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
