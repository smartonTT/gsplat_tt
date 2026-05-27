"""Reconcile metal-port validator verdict + metrics (mirrors decide_and_log.py).

Appends to opt/metal-iters.jsonl instead of opt/iters.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.decide_and_log import decide, git_commit


def atomic_append_jsonl(path: Path, row: dict) -> None:
    existing = path.read_text() if path.exists() else ""
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), prefix=".metal-iters-") as tmp:
        tmp.write(existing)
        if existing and not existing.endswith("\n"):
            tmp.write("\n")
        tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def latest_best(jsonl: Path) -> float:
    if not jsonl.exists() or not jsonl.read_text().strip():
        return float("inf")
    best = float("inf")
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("action") == "commit":
            best = min(best, row.get("sum_total_ms", float("inf")))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--opt-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    metrics = json.loads((args.iter_dir / "metrics.json").read_text())
    validator = json.loads((args.iter_dir / "validator.json").read_text())
    jsonl = args.opt_dir / "metal-iters.jsonl"
    prev_best = metrics.get("prev_best_sum_ms")
    if prev_best is None or prev_best != prev_best:
        prev_best = latest_best(jsonl)

    decision = decide(metrics, validator, prev_best)
    commit_sha = None
    if decision["action"] == "commit" and args.commit and not args.dry_run:
        commit_sha = git_commit(metrics["iter_dir"], metrics)
    decision["commit_sha"] = commit_sha

    (args.iter_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    row = {
        "iter_dir": metrics["iter_dir"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": decision["verdict"],
        "action": decision["action"],
        "sum_total_ms": metrics["sum_total_ms"],
        "per_view_median_ms": metrics.get("per_view_median_ms", {}),
        "per_stage_median_ms": metrics.get("per_stage_median_ms", {}),
        "psnr_per_view": metrics.get("psnr_per_view", {}),
        "tile_structure_per_view": metrics.get("tile_structure_per_view", {}),
        "class": metrics.get("class", "port"),
        "commit_sha": commit_sha,
        "high_promotion_priority": decision["high_promotion_priority"],
        "validator_reasoning": validator.get("reasoning", ""),
    }
    atomic_append_jsonl(jsonl, row)

    build_report = args.opt_dir / "build_report.py"
    if build_report.exists():
        try:
            subprocess.run(["python3", str(build_report)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"warning: build_report.py failed: {e}", flush=True)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
