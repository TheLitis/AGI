"""Render an exported trajectory JSONL file as a static HTML replay."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                episodes.append(payload)
    return episodes


def _render_html(episode: Dict[str, Any], source: Path) -> str:
    steps = episode.get("steps")
    if not isinstance(steps, list):
        steps = []
    safe_title = html.escape(str(source))
    episode_json = json.dumps(episode, ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trajectory Replay</title>
  <style>
    :root {{
      --bg: #10151d;
      --panel: #182231;
      --panel-2: #202d3f;
      --ink: #f4f1e8;
      --muted: #aeb8c4;
      --accent: #f2b84b;
      --grid: #33465f;
      --good: #66d19e;
      --bad: #f07167;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 10% 0%, rgba(242, 184, 75, 0.15), transparent 34rem),
        linear-gradient(135deg, #0b0f15, var(--bg));
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 5vw, 4.5rem);
      letter-spacing: -0.06em;
    }}
    .source, .muted {{ color: var(--muted); }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(260px, 420px);
      gap: 18px;
      margin-top: 24px;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.28);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(var(--cols), minmax(30px, 1fr));
      gap: 6px;
      margin: 16px 0;
    }}
    .cell {{
      min-height: 38px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      background: var(--grid);
      color: var(--ink);
      font: 700 0.95rem/1 ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
    .cell.v0 {{ background: #253244; }}
    .cell.v1 {{ background: #456581; }}
    .cell.v2 {{ background: #668f5a; }}
    .cell.v3 {{ background: #ba7a3d; }}
    .cell.v4 {{ background: #9c4d4d; }}
    .cell.v5 {{ background: #715aa8; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .stat {{
      background: var(--panel-2);
      border-radius: 14px;
      padding: 12px;
    }}
    .stat strong {{
      display: block;
      font: 700 1.25rem/1.2 ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 16px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: #19130a;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    input[type="range"] {{ flex: 1 1 220px; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #0b1017;
      border-radius: 16px;
      padding: 14px;
      color: #d7e0ea;
      max-height: 420px;
      overflow: auto;
    }}
    @media (max-width: 820px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <p class="muted">AGI trajectory inspector</p>
  <h1>Trajectory Replay</h1>
  <div class="source">{safe_title}</div>
  <section class="layout">
    <div class="panel">
      <div id="stepLabel" class="muted"></div>
      <div id="grid" class="grid"></div>
      <div class="controls">
        <button id="prev">Prev</button>
        <button id="play">Play</button>
        <button id="next">Next</button>
        <input id="slider" type="range" min="0" max="{max(0, len(steps) - 1)}" value="0">
      </div>
      <div class="stats" id="stats"></div>
    </div>
    <div class="panel">
      <h2>Step Contract</h2>
      <pre id="info"></pre>
      <h2>Episode</h2>
      <pre id="episode"></pre>
    </div>
  </section>
</main>
<script>
const episode = {episode_json};
const steps = Array.isArray(episode.steps) ? episode.steps : [];
let idx = 0;
let timer = null;
const gridEl = document.getElementById("grid");
const infoEl = document.getElementById("info");
const episodeEl = document.getElementById("episode");
const statsEl = document.getElementById("stats");
const labelEl = document.getElementById("stepLabel");
const sliderEl = document.getElementById("slider");

function patchFor(step) {{
  return step.next_obs_patch || step.obs_patch || [];
}}

function renderGrid(patch) {{
  if (!Array.isArray(patch) || patch.length === 0 || !Array.isArray(patch[0])) {{
    gridEl.style.setProperty("--cols", "1");
    gridEl.innerHTML = '<div class="cell">no patch</div>';
    return;
  }}
  const cols = patch[0].length || 1;
  gridEl.style.setProperty("--cols", String(cols));
  gridEl.innerHTML = patch.flatMap(row => row.map(value => {{
    const n = Number(value);
    const cls = Number.isFinite(n) ? `v${{Math.max(0, Math.min(5, n))}}` : "v0";
    return `<div class="cell ${{cls}}">${{String(value)}}</div>`;
  }})).join("");
}}

function stat(label, value, tone) {{
  return `<div class="stat"><span class="muted">${{label}}</span><strong style="color:${{tone || 'var(--ink)'}}">${{value}}</strong></div>`;
}}

function render() {{
  const step = steps[idx] || {{}};
  renderGrid(patchFor(step));
  labelEl.textContent = `step ${{idx + 1}} / ${{Math.max(steps.length, 1)}}`;
  sliderEl.value = String(idx);
  const doneTone = step.done ? "var(--bad)" : "var(--good)";
  statsEl.innerHTML = [
    stat("action", step.action ?? "n/a"),
    stat("reward", Number(step.reward || 0).toFixed(3)),
    stat("energy", Number(step.next_energy ?? step.energy ?? 0).toFixed(2)),
    stat("done", String(Boolean(step.done)), doneTone),
  ].join("");
  infoEl.textContent = JSON.stringify(step.info || {{}}, null, 2);
  episodeEl.textContent = JSON.stringify({{
    run_id: episode.run_id,
    stage: episode.stage,
    episode_index: episode.episode_index,
    env_name: episode.env_name,
    scenario_name: episode.scenario_name,
    total_return: episode.total_return,
    length: episode.length,
    final_info: episode.final_info,
  }}, null, 2);
}}

document.getElementById("prev").onclick = () => {{ idx = Math.max(0, idx - 1); render(); }};
document.getElementById("next").onclick = () => {{ idx = Math.min(Math.max(steps.length - 1, 0), idx + 1); render(); }};
sliderEl.oninput = () => {{ idx = Number(sliderEl.value || 0); render(); }};
document.getElementById("play").onclick = (event) => {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
    event.target.textContent = "Play";
    return;
  }}
  event.target.textContent = "Pause";
  timer = setInterval(() => {{
    idx = idx >= steps.length - 1 ? 0 : idx + 1;
    render();
  }}, 500);
}};
render();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="Trajectory JSONL produced by --save-trajectories.")
    parser.add_argument("--out", default="reports/visualizations/replay.html", help="Output HTML file.")
    parser.add_argument("--episode-index", type=int, default=0, help="JSONL episode index to render.")
    args = parser.parse_args()

    source = Path(args.trajectory)
    episodes = _load_episodes(source)
    if not episodes:
        raise SystemExit(f"No episodes found in {source}")
    idx = max(0, min(int(args.episode_index), len(episodes) - 1))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(episodes[idx], source), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
