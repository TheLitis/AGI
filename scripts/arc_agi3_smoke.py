"""Run/list ARC-AGI-3 environments through the official SDK adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_agi3_adapter import (
    ARC_AGI3_AVAILABLE,
    list_arc_agi3_games,
    run_arc_agi3_episode,
    write_arc_agi3_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", help="Game id to run. Repeat for multiple games.")
    parser.add_argument("--list-games", action="store_true", help="List available ARC-AGI-3 games and exit.")
    parser.add_argument("--mode", default="normal", choices=["normal", "online", "offline", "competition"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--policy", default="random", choices=["random", "first", "last"])
    parser.add_argument("--render-mode", default=None, choices=[None, "terminal", "terminal-fast", "human"])
    parser.add_argument("--no-recording", action="store_true", help="Disable official ARC JSONL recording.")
    parser.add_argument("--make-retries", type=int, default=3, help="Retries for SDK environment creation/download.")
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--recordings-dir", default="reports/arc_agi3/recordings")
    parser.add_argument("--report", default="reports/arc_agi3/smoke.json")
    args = parser.parse_args()

    if not ARC_AGI3_AVAILABLE:
        raise SystemExit("ARC-AGI-3 SDK missing. Install with: python -m pip install arc-agi")

    if args.list_games:
        for game_id in list_arc_agi3_games(mode=args.mode, environments_dir=args.environments_dir):
            print(game_id)
        return 0

    games = args.game or ["ls20"]
    results = []
    for idx, game_id in enumerate(games):
        result = run_arc_agi3_episode(
            game_id=str(game_id),
            seed=int(args.seed) + int(idx),
            max_steps=int(args.max_steps),
            mode=str(args.mode),
            policy=str(args.policy),
            render_mode=args.render_mode,
            save_recording=not bool(args.no_recording),
            environments_dir=str(args.environments_dir),
            recordings_dir=str(args.recordings_dir),
            make_retries=int(args.make_retries),
        )
        results.append(result)
        print(
            f"{result.game_id}: state={result.final_state} levels={result.levels_completed}/"
            f"{result.win_levels} steps={result.steps} score={result.score} error={result.error}"
        )

    payload = write_arc_agi3_report(results, Path(args.report))
    print(f"[OK] wrote {args.report} summary={payload.get('summary')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
