"""Reconcile validator verdict + metrics into a commit/backburner decision.

Reads:
  <iter_dir>/metrics.json
  <iter_dir>/validator.json
  <state_dir>/iters.jsonl       (current log; appended)
  <state_dir>/BACKBURNER.md     (appended for REJECT/NEEDS_REVIEW)
  <state_dir>/STATUS.md         (updated header + recent decisions)

Writes:
  <iter_dir>/decision.json      {verdict, action, high_promotion_priority, commit_sha?}
  <state_dir>/iters.jsonl       (atomically appended one line)
  <state_dir>/BACKBURNER.md     (appended)
  <state_dir>/STATUS.md         (overwritten with updated state)

Actions enum:
  - "commit"                            (KEEP + faster than prev best)
  - "no_commit_valid_but_not_faster"    (KEEP but no speed improvement)
  - "backburner"                        (REJECT or NEEDS_REVIEW)

With --commit, also performs `git add <worker-manifest> && git commit -m ...`.
With --dry-run, decision.json is written but no git commit or destructive log mutation
(the iters.jsonl append still happens — it's the audit trail).
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
    """Append one line atomically via temp+rename of the whole file."""
    existing = path.read_text() if path.exists() else ""
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), prefix=".iters-") as tmp:
        tmp.write(existing)
        if existing and not existing.endswith("\n"):
            tmp.write("\n")
        tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def latest_best(jsonl: Path) -> float:
    """Lowest kernel_ms_median among committed KEEPs in the log; inf if none."""
    if not jsonl.exists() or not jsonl.read_text().strip():
        return float("inf")
    best = float("inf")
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("action") == "commit":
            best = min(best, row.get("kernel_ms_median", float("inf")))
    return best


def decide(metrics: dict, validator: dict, prev_best: float) -> dict:
    verdict = validator["verdict"]
    ms = metrics["kernel_ms_median"]
    faster = ms <= prev_best * 1.02
    high_priority = False

    if verdict == "KEEP":
        action = "commit" if faster else "no_commit_valid_but_not_faster"
    elif verdict in ("REJECT", "NEEDS_REVIEW"):
        action = "backburner"
        high_priority = (verdict == "NEEDS_REVIEW") and (ms < prev_best)
    else:
        action = "backburner"  # malformed → safe fallback

    return {"verdict": verdict, "action": action, "high_promotion_priority": high_priority,
            "kernel_ms_median": ms, "prev_best_kernel_ms": prev_best}


def write_backburner_entry(backburner: Path, iter_name: str, decision: dict, metrics: dict, validator: dict) -> None:
    priority_badge = "⭐ HIGH-PROMOTION-PRIORITY" if decision["high_promotion_priority"] else ""
    psnr = metrics["psnr_per_view"]
    entry = f"""
## {iter_name} — {decision['verdict']} {priority_badge}

- Class: `{metrics['class']}`
- kernel ms: median {metrics['kernel_ms_median']:.2f} / p99 {metrics['kernel_ms_p99']:.2f}
- PSNR per view: hero {psnr['hero']:.1f} / side {psnr['side']:.1f} / top {psnr['top']:.1f}
- Validator reasoning: {validator['reasoning']}
- Thumbnails: ![hero](screenshots/{iter_name}/hero.png) ![diff10](screenshots/{iter_name}/hero_diff10.png)

"""
    with backburner.open("a") as f:
        f.write(entry)


def update_status(status: Path, decision: dict, metrics: dict, jsonl: Path) -> None:
    best = latest_best(jsonl)
    best_str = f"{best:.2f} ms" if best != float("inf") else "n/a"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # Preserve any existing ESCALATIONS section; just update the Current State.
    text = status.read_text()
    head, _, _ = text.partition("## Current State")
    new = head + f"""## Current State

- Last updated: {timestamp}
- Last iter: {metrics['iter_dir']} → {decision['verdict']} / {decision['action']}
- Current best kernel ms (committed): {best_str}
- Last iter kernel ms median: {metrics['kernel_ms_median']:.2f}
- Target: 1.0 ms
"""
    status.write_text(new)


def git_commit(iter_name: str, metrics: dict) -> str:
    psnr_min = min(metrics["psnr_per_view"].values())
    msg = f"iter {iter_name}: {metrics['class']} kernel={metrics['kernel_ms_median']:.2f} ms PSNR_min={psnr_min:.1f}"
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    if args.dry_run and args.commit:
        raise SystemExit("--dry-run and --commit are mutually exclusive")

    metrics = json.loads((args.iter_dir / "metrics.json").read_text())
    validator = json.loads((args.iter_dir / "validator.json").read_text())
    jsonl = args.state_dir / "iters.jsonl"
    # metrics.json carries the authoritative prev_best (set by run_iter.sh at the
    # start of the iter from the log). Re-reading the log here would race against
    # any other decide_and_log running in parallel. Fall back to the log only if
    # metrics.json doesn't have it (shouldn't happen in practice).
    prev_best = metrics.get("prev_best_kernel_ms")
    if prev_best is None or prev_best != prev_best:  # also catch NaN
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
        "kernel_ms_median": metrics["kernel_ms_median"],
        "kernel_ms_p99": metrics["kernel_ms_p99"],
        "per_view_median": metrics["per_view_median"],
        "psnr_per_view": metrics["psnr_per_view"],
        "tile_structure_ratio_per_view": metrics.get("tile_structure_ratio_per_view"),
        "class": metrics["class"],
        "commit_sha": commit_sha,
        "high_promotion_priority": decision["high_promotion_priority"],
    }
    if "stage_medians" in metrics:
        row["stage_medians"] = metrics["stage_medians"]
    atomic_append_jsonl(jsonl, row)

    if decision["action"] == "backburner":
        write_backburner_entry(args.state_dir / "BACKBURNER.md", metrics["iter_dir"], decision, metrics, validator)

    update_status(args.state_dir / "STATUS.md", decision, metrics, jsonl)

    # Always refresh REPORT.html after a decision so the human-facing report
    # never lags iters.jsonl. Best-effort: a build_report failure must not
    # fail the whole decision (we still want the row in the log + STATUS).
    build_report = args.state_dir / "build_report.py"
    if build_report.exists():
        try:
            subprocess.run(["python3", str(build_report)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"warning: build_report.py failed: {e}", flush=True)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
