"""Summarize a render_30frame timing.jsonl into median + heavy views."""
import json
import statistics
import sys
from pathlib import Path


def main(path: str = "/tmp/amend002-cpu-cpp-mb/timing.jsonl"):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    totals = [r["total_ms"] for r in rows]
    projects = [r.get("project", 0) for r in rows]
    tas = [r.get("tile_assign", 0) for r in rows]
    sorts = [r.get("sort", 0) for r in rows]
    blends = [r.get("blend", 0) for r in rows]
    print(
        f"30 views median: "
        f"total={statistics.median(totals):.1f} "
        f"project={statistics.median(projects):.1f} "
        f"ta={statistics.median(tas):.1f} "
        f"sort={statistics.median(sorts):.1f} "
        f"blend={statistics.median(blends):.1f}"
    )
    print(
        f"30 views sum:    "
        f"total={sum(totals):.1f} "
        f"project={sum(projects):.1f} "
        f"ta={sum(tas):.1f} "
        f"sort={sum(sorts):.1f} "
        f"blend={sum(blends):.1f}"
    )
    heavy = sorted(rows, key=lambda r: r["total_ms"], reverse=True)[:5]
    print("heaviest views:")
    for r in heavy:
        name = r.get("view", "")
        total = r["total_ms"]
        proj = r.get("project", 0)
        blend = r.get("blend", 0)
        ta = r.get("tile_assign", 0)
        print(f"  {name:14s} total={total:6.1f}  project={proj:5.1f}  ta={ta:5.1f}  blend={blend:5.1f}")
    light = sorted(rows, key=lambda r: r["total_ms"])[:5]
    print("lightest views:")
    for r in light:
        name = r.get("view", "")
        total = r["total_ms"]
        proj = r.get("project", 0)
        blend = r.get("blend", 0)
        ta = r.get("tile_assign", 0)
        print(f"  {name:14s} total={total:6.1f}  project={proj:5.1f}  ta={ta:5.1f}  blend={blend:5.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/amend002-cpu-cpp-mb/timing.jsonl")
