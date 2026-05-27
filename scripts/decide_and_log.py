"""Reconcile validator verdict + metrics into a commit/backburner decision.

Reads:
  <iter_dir>/metrics.json
  <iter_dir>/validator.json
  <opt_dir>/iters.jsonl       (appended)

Writes:
  <iter_dir>/decision.json
  <opt_dir>/iters.jsonl       (atomically appended one JSONL row)
  <opt_dir>/REPORT.html       (regenerated via build_report.py if present)

Action enum:
  - "commit"                            (KEEP + faster than prev best)
  - "no_commit_valid_but_not_faster"    (KEEP but no speed improvement)
  - "backburner"                        (REJECT or NEEDS_REVIEW)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_append_jsonl(path: Path, row: dict) -> None:
    existing = path.read_text() if path.exists() else ""
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), prefix=".iters-") as tmp:
        tmp.write(existing)
        if existing and not existing.endswith("\n"):
            tmp.write("\n")
        tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def latest_best(jsonl: Path) -> float:
    """Lowest sum_total_ms among committed KEEPs in the log; inf if none."""
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


def decide(metrics: dict, validator: dict, prev_best: float) -> dict:
    verdict = validator["verdict"]
    ms = metrics["sum_total_ms"]
    # 2% tolerance: noise band on a 30-frame sum is real.
    faster = ms <= prev_best * 1.02
    high_priority = False

    if verdict == "KEEP":
        action = "commit" if faster else "no_commit_valid_but_not_faster"
    elif verdict in ("REJECT", "NEEDS_REVIEW"):
        action = "backburner"
        high_priority = (verdict == "NEEDS_REVIEW") and (ms < prev_best)
    else:
        action = "backburner"

    return {
        "verdict": verdict,
        "action": action,
        "high_promotion_priority": high_priority,
        "sum_total_ms": ms,
        "prev_best_sum_ms": prev_best,
    }


def git_commit(iter_name: str, metrics: dict) -> str:
    psnr_per_view = metrics.get("psnr_per_view", {})
    psnr_min = min(psnr_per_view.values()) if psnr_per_view else float("nan")
    msg = (
        f"iter {iter_name}: {metrics['class']} "
        f"sum_ms={metrics['sum_total_ms']:.1f} "
        f"psnr_min={psnr_min:.1f}"
    )
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--opt-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    if args.dry_run and args.commit:
        raise SystemExit("--dry-run and --commit are mutually exclusive")

    metrics = json.loads((args.iter_dir / "metrics.json").read_text())
    validator = json.loads((args.iter_dir / "validator.json").read_text())
    jsonl = args.opt_dir / "iters.jsonl"
    prev_best = metrics.get("prev_best_sum_ms")
    if prev_best is None or prev_best != prev_best:  # also catches NaN
        prev_best = latest_best(jsonl)

    decision = decide(metrics, validator, prev_best)

    commit_sha = None
    if decision["action"] == "commit" and args.commit:
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
        "class": metrics["class"],
        "commit_sha": commit_sha,
        "high_promotion_priority": decision["high_promotion_priority"],
        "validator_reasoning": validator.get("reasoning", ""),
    }
    atomic_append_jsonl(jsonl, row)

    # Regenerate REPORT.html best-effort. A failure here must not fail the decision.
    build_report = args.opt_dir / "build_report.py"
    if build_report.exists():
        try:
            subprocess.run(["python3", str(build_report)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"warning: build_report.py failed: {e}", flush=True)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
