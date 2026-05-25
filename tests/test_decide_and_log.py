"""Tests for scripts/decide_and_log.py reconciliation logic."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "decide_and_log.py"


def _setup_iter(tmp_path, validator_verdict: str, kernel_ms: float, prev_best: float):
    iter_dir = tmp_path / "iter-099-testslug"
    iter_dir.mkdir()
    (iter_dir / "metrics.json").write_text(json.dumps({
        "iter_dir": "iter-099-testslug",
        "class": "kernel-algebra",
        "kernel_ms_median": kernel_ms,
        "kernel_ms_p99": kernel_ms * 1.2,
        "per_view_median": {"hero": kernel_ms, "side": kernel_ms, "top": kernel_ms},
        "psnr_per_view": {"hero": 110.0, "side": 110.0, "top": 110.0},
        "prev_best_kernel_ms": prev_best,
        "frame_count": 30,
    }))
    (iter_dir / "validator.json").write_text(json.dumps({
        "verdict": validator_verdict,
        "visual_checks": [{"name": f"check_{i}", "result": "pass", "evidence": ""} for i in range(8)],
        "psnr_check": {"floor": 100.0, "actual": {"hero": 110.0, "side": 110.0, "top": 110.0}, "pass": True},
        "per_view_consistency": {"max_psnr_delta_db": 0.1, "max_ms_ratio": 1.0, "pass": True},
        "timing": {"median_ms": kernel_ms, "p99_ms": kernel_ms * 1.2, "vs_prev_best_pct": ((kernel_ms - prev_best) / prev_best) * 100.0, "pass": kernel_ms <= prev_best * 1.02},
        "reasoning": "All checks pass.",
    }))
    return iter_dir


def _setup_state(tmp_path):
    state_dir = tmp_path / "docs" / "optimization-log"
    state_dir.mkdir(parents=True)
    (state_dir / "iters.jsonl").touch()
    (state_dir / "STATUS.md").write_text("# Status\n## ESCALATIONS (read these first)\n\n_None._\n\n## Current State\n- Current best: n/a\n")
    (state_dir / "BACKBURNER.md").write_text("# BACKBURNER\n\n_None yet._\n")
    return state_dir


def run_decide(iter_dir, state_dir, dry_run=True):
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--iter-dir", str(iter_dir),
         "--state-dir", str(state_dir),
         "--dry-run" if dry_run else "--commit"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    return json.loads((iter_dir / "decision.json").read_text())


def test_keep_faster_then_commit_action(tmp_path):
    iter_dir = _setup_iter(tmp_path, "KEEP", kernel_ms=10.0, prev_best=20.0)
    state_dir = _setup_state(tmp_path)
    decision = run_decide(iter_dir, state_dir, dry_run=True)
    assert decision["action"] == "commit"
    assert decision["verdict"] == "KEEP"


def test_keep_not_faster_then_no_commit(tmp_path):
    iter_dir = _setup_iter(tmp_path, "KEEP", kernel_ms=25.0, prev_best=20.0)
    state_dir = _setup_state(tmp_path)
    decision = run_decide(iter_dir, state_dir, dry_run=True)
    assert decision["action"] == "no_commit_valid_but_not_faster"


def test_reject_then_backburner(tmp_path):
    iter_dir = _setup_iter(tmp_path, "REJECT", kernel_ms=5.0, prev_best=20.0)
    state_dir = _setup_state(tmp_path)
    decision = run_decide(iter_dir, state_dir, dry_run=True)
    assert decision["action"] == "backburner"
    assert "BACKBURNER" in (state_dir / "BACKBURNER.md").read_text()


def test_needs_review_then_backburner_high_priority(tmp_path):
    iter_dir = _setup_iter(tmp_path, "NEEDS_REVIEW", kernel_ms=5.0, prev_best=20.0)
    state_dir = _setup_state(tmp_path)
    decision = run_decide(iter_dir, state_dir, dry_run=True)
    assert decision["action"] == "backburner"
    assert decision["high_promotion_priority"] is True  # NEEDS_REVIEW + faster than best


def test_iters_jsonl_append(tmp_path):
    iter_dir = _setup_iter(tmp_path, "KEEP", kernel_ms=10.0, prev_best=20.0)
    state_dir = _setup_state(tmp_path)
    run_decide(iter_dir, state_dir, dry_run=True)
    lines = (state_dir / "iters.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["verdict"] == "KEEP"
    assert row["kernel_ms_median"] == 10.0
