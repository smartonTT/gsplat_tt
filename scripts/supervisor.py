"""Supervisor outer loop — priority queue + per-iter sequencer.

In v1 this script is invoked turn-by-turn by the supervising Claude:

  python scripts/supervisor.py next-iter           # prints next queue item + worker prompt
  python scripts/supervisor.py validate <iter_dir> # invokes dispatch_validator.sh prep + reminds supervisor to call subagent
  python scripts/supervisor.py decide <iter_dir>   # runs decide_and_log.py --commit
  python scripts/supervisor.py status              # prints summary of current state
  python scripts/supervisor.py queue-status        # prints remaining queue items + stalled sub-tracks

The actual subagent dispatch (worker + validator) happens via the supervising
Claude's Agent tool. supervisor.py is the deterministic accounting layer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "optimization-log"
JSONL = OUT / "iters.jsonl"
STATUS = OUT / "STATUS.md"
QUEUE_PATH = OUT / "QUEUE.json"


DEFAULT_QUEUE = [
    {"iter": 1, "slug": "dst-resident-state", "class": "kernel-algebra", "track": "kernel-layer-a",
     "hypothesis": "Acquire Dst once per output tile; keep R/G/B/T in Dst[0..3] across the Gaussian loop. Storage relocation only, should be bit-identical."},
    {"iter": 2, "slug": "basis-form-tile-local", "class": "kernel-algebra", "track": "kernel-layer-b",
     "hypothesis": "Expand Q = A·x² + B·xy + C·y² + D·x + E·y + F with tile-local centered coords in [-15.5, 15.5]. 6 mul_tiles + acc_to_dest + exp_tile."},
    {"iter": 3, "slug": "layer-a-plus-b", "class": "kernel-algebra", "track": "kernel-combined",
     "hypothesis": "Compose Dst-resident state (Layer A) with basis-form (Layer B)."},
    {"iter": 4, "slug": "tighter-tile-bbox", "class": "binning", "track": "binning",
     "hypothesis": "Use per-Gaussian (μ, Σ) for 3σ ellipse bbox vs conservative axis-aligned."},
    {"iter": 5, "slug": "skip-empty-tiles", "class": "dispatch", "track": "dispatch",
     "hypothesis": "Skip kernel launch for tiles with zero assigned Gaussians."},
    {"iter": 6, "slug": "parallel-sort-tiles", "class": "sort", "track": "sort",
     "hypothesis": "Parallel sort across tiles using available CPU cores."},
    {"iter": 7, "slug": "host-device-overlap", "class": "dispatch", "track": "overlap",
     "hypothesis": "Pipeline depth 2 at host/device boundary; prep frame N+1 while device runs frame N."},
]


def ensure_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        QUEUE_PATH.write_text(json.dumps(DEFAULT_QUEUE, indent=2))
    return json.loads(QUEUE_PATH.read_text())


def completed_iter_nums() -> set[int]:
    if not JSONL.exists():
        return set()
    nums = set()
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row["iter_dir"]
        # iter-NNN-slug
        try:
            n = int(name.split("-")[1])
            nums.add(n)
        except (IndexError, ValueError):
            pass
    return nums


def stalled_tracks() -> set[str]:
    """Track is stalled if it has 3 consecutive backburner entries."""
    if not JSONL.exists():
        return set()
    by_track_recent: dict[str, list[str]] = {}
    queue = ensure_queue()
    track_by_iter = {q["iter"]: q["track"] for q in queue}
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row["iter_dir"]
        try:
            n = int(name.split("-")[1])
        except (IndexError, ValueError):
            continue
        track = track_by_iter.get(n)
        if track is None:
            continue
        by_track_recent.setdefault(track, []).append(row.get("action", ""))

    stalled = set()
    for track, actions in by_track_recent.items():
        # 3 consecutive 'backburner' (REJECT or NEEDS_REVIEW) → stalled
        if len(actions) >= 3 and all(a == "backburner" for a in actions[-3:]):
            stalled.add(track)
    return stalled


def next_iter() -> dict | None:
    queue = ensure_queue()
    done = completed_iter_nums()
    stalled = stalled_tracks()
    for q in queue:
        if q["iter"] in done:
            continue
        if q["track"] in stalled:
            continue
        return q
    return None


def cmd_next_iter():
    item = next_iter()
    if not item:
        print(json.dumps({"status": "queue_exhausted_or_all_stalled"}))
        sys.exit(0)
    worker_prompt = (REPO / "prompts" / "worker.md").read_text()
    print(json.dumps({"item": item, "worker_prompt_template": worker_prompt}, indent=2))


def cmd_validate(iter_dir: str):
    subprocess.check_call([str(REPO / "scripts" / "dispatch_validator.sh"), iter_dir, "--prepare-prompt"])


def cmd_decide(iter_dir: str):
    subprocess.check_call([
        sys.executable, str(REPO / "scripts" / "decide_and_log.py"),
        "--iter-dir", iter_dir,
        "--state-dir", str(OUT),
        "--commit",
    ])
    subprocess.check_call([sys.executable, str(OUT / "build_report.py")])


def cmd_status():
    completed = sorted(completed_iter_nums())
    stalled = sorted(stalled_tracks())
    item = next_iter()
    print(json.dumps({
        "completed_iters": completed,
        "stalled_tracks": stalled,
        "next_iter_item": item,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def cmd_queue_status():
    queue = ensure_queue()
    done = completed_iter_nums()
    stalled = stalled_tracks()
    rows = []
    for q in queue:
        rows.append({
            "iter": q["iter"], "slug": q["slug"], "class": q["class"], "track": q["track"],
            "status": "done" if q["iter"] in done else ("stalled" if q["track"] in stalled else "pending"),
        })
    print(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next-iter")
    p = sub.add_parser("validate"); p.add_argument("iter_dir")
    p = sub.add_parser("decide"); p.add_argument("iter_dir")
    sub.add_parser("status")
    sub.add_parser("queue-status")
    args = ap.parse_args()
    {
        "next-iter": cmd_next_iter,
        "validate": lambda: cmd_validate(args.iter_dir),
        "decide": lambda: cmd_decide(args.iter_dir),
        "status": cmd_status,
        "queue-status": cmd_queue_status,
    }[args.cmd]()


if __name__ == "__main__":
    main()
