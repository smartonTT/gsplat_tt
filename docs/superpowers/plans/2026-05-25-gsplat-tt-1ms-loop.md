# gsplat_tt 1 ms autonomous-loop infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autonomous optimization loop infrastructure described in `docs/superpowers/specs/2026-05-25-gsplat-tt-1ms-design.md` and freeze the iter-0 reference. Terminal state: supervisor dispatches iter-1 (Layer A — Dst-resident state) without human intervention.

**Architecture:** Three-layer loop — supervisor (Opus 4.7) dispatches worker (Sonnet 4.6) per experiment, then dispatches validator (Sonnet 4.6, fresh context) to judge artifacts cold. All decisions land in `iters.jsonl`/`STATUS.md`/`BACKBURNER.md`/`REPORT.html`. No user gates. Builds heavily on existing scripts (`render_fixed.py`, `image_diff.py`, `build_report.py`).

**Tech Stack:** Python 3.11, bash, jq, tt-metal, numpy, scikit-image, matplotlib, PIL. Mac for orchestration + git, `yyzo-bh-14` for builds/renders via devsync.

---

## File Structure

**New files:**
- `scripts/run_iter.sh` — single-command iter driver (build + render + metrics + report)
- `scripts/compute_metrics.py` — fp64 PSNR + per-view aggregation (replaces single-image `image_diff.py` usage)
- `scripts/dispatch_validator.sh` — bundles artifacts and invokes validator subagent
- `scripts/decide_and_log.py` — §4 reconciliation matrix + git commit + log update
- `scripts/health_check.sh` — supervisor watchdog
- `scripts/promote_to_stable.sh` — only bh-30 toucher; KEEP→stable_viewer promotion
- `scripts/supervisor.py` — outer loop entry point
- `prompts/validator.md` — validator subagent prompt template (embeds §5 rules)
- `prompts/worker.md` — worker subagent prompt template
- `.claude/settings.local.json` — bypassPermissions + reference write-protect hook
- `docs/optimization-log/STATUS.md` — supervisor live state (initialized empty)
- `docs/optimization-log/BACKBURNER.md` — parked items (initialized empty)
- `docs/optimization-log/iters.jsonl` — append-only iter log (initialized empty)
- `docs/optimization-log/iter-000-baseline.md` — iter-0 record
- `benchmarks/reference/stitch_side.png` — frozen reference (new, 1024×1024 implicit)
- `benchmarks/reference/stitch_top.png` — frozen reference (new, 1024×1024 implicit)
- `.opt-v2-last-build-commit` — gitignored sentinel for JIT cache wipe gate
- `tests/test_compute_metrics.py`, `tests/test_decide_and_log.py`, `tests/test_dispatch_validator.sh`, `tests/fixtures/` — unit + integration tests

**Modified files:**
- `scripts/render_fixed.py` — add `--cycles` training-pattern mode
- `docs/optimization-log/build_report.py` — REPORT.md → REPORT.html, new graphs, validator JSON rendering, BACKBURNER section
- `.gitignore` — add `.opt-v2-last-build-commit`

**Carry-over from `smarton/opt-stable`** (one commit):
- `gsplat/viewer.py`, `gsplat/camera_controls.py`, `gsplat/letterbox.py`, `gsplat/nerfview_viewer.py`, `gsplat/viser_patches.py`
- `scripts/derive_camera.py`, `scripts/derive_all_views.py`, `scripts/start_stable_viewer.sh`
- `docs/optimization-log/report.css`
- `gsplat/__main__.py --force-square 1024` flag
- `gsplat/rasterization.py` 1024-res support diff

---

## Phase 1 — Branch setup and infra carry-over

### Task 1: Stash opt-stable WIP and create opt-v2 branch

**Files:**
- New branch: `smarton/opt-v2` off `origin/main`
- Backup: `/Users/smarton/dev/nosync/opt-stable-wip-2026-05-25/`

- [ ] **Step 1: Inspect uncommitted state on opt-stable**

```bash
cd /Users/smarton/dev/gsplat_tt
git status --short
git stash list
```

Expected: modifications present (CLAUDE.md, backend.py, benchmarks/reference/luigi_*.png, others).

- [ ] **Step 2: Back up uncommitted state to nosync (preserve, don't lose)**

```bash
mkdir -p /Users/smarton/dev/nosync/opt-stable-wip-2026-05-25/
git diff > /Users/smarton/dev/nosync/opt-stable-wip-2026-05-25/diff.patch
git status --short > /Users/smarton/dev/nosync/opt-stable-wip-2026-05-25/status.txt
```

- [ ] **Step 3: Stash and verify clean tree**

```bash
git stash push -u -m "opt-stable wip 2026-05-25 pre opt-v2 branch"
git status --short
```

Expected: empty output (clean tree).

- [ ] **Step 4: Fetch origin and create opt-v2 off origin/main**

```bash
git fetch origin
git checkout -b smarton/opt-v2 origin/main
git log --oneline -5
```

Expected: HEAD is on origin/main's tip.

- [ ] **Step 5: Commit — no code change; this task is a state transition**

No commit yet (next task adds the first commit on opt-v2).

---

### Task 2: Cherry-pick infra carry-over from opt-stable

**Files:**
- Cherry-pick source: `smarton/opt-stable`
- Target paths (preserve exact):
  - `gsplat/viewer.py`, `gsplat/camera_controls.py`, `gsplat/letterbox.py`, `gsplat/nerfview_viewer.py`, `gsplat/viser_patches.py`
  - `scripts/derive_camera.py`, `scripts/derive_all_views.py`, `scripts/start_stable_viewer.sh`, `scripts/render_fixed.py`, `scripts/image_diff.py`
  - `docs/optimization-log/report.css`, `docs/optimization-log/build_report.py`
  - `gsplat/__main__.py`, `gsplat/rasterization.py` (1024-res relevant diffs only)

- [ ] **Step 1: Verify these files exist on opt-stable HEAD**

```bash
git show smarton/opt-stable:gsplat/viewer.py | head -5
git show smarton/opt-stable:scripts/render_fixed.py | head -5
git show smarton/opt-stable:docs/optimization-log/build_report.py | head -5
```

Expected: each file's contents visible (no "fatal: path does not exist").

- [ ] **Step 2: Checkout files individually from opt-stable**

```bash
for f in \
  gsplat/viewer.py gsplat/camera_controls.py gsplat/letterbox.py gsplat/nerfview_viewer.py gsplat/viser_patches.py \
  scripts/derive_camera.py scripts/derive_all_views.py scripts/start_stable_viewer.sh scripts/render_fixed.py scripts/image_diff.py \
  docs/optimization-log/report.css docs/optimization-log/build_report.py
do
  git checkout smarton/opt-stable -- "$f"
done
git status --short
```

Expected: those files staged.

- [ ] **Step 3: Selectively bring 1024-res support from opt-stable's `__main__.py` and `rasterization.py`**

```bash
# Diff opt-stable vs main to see only the 1024-res relevant changes
git diff origin/main..smarton/opt-stable -- gsplat/__main__.py gsplat/rasterization.py > /tmp/res-diff.patch
# Review and apply only the --force-square flag wiring + 1024 plumbing
# (engineer reviews diff manually; if it's solely 1024-res, apply with `git apply`)
cat /tmp/res-diff.patch
```

If the diff contains only 1024-res wiring (`--force-square`, 1024 handling), apply:
```bash
git apply /tmp/res-diff.patch
```

If the diff is broader, hand-edit `gsplat/__main__.py` and `gsplat/rasterization.py` to add ONLY the 1024-square flag and its rasterization-pipeline handling — the rest stays at main's behavior.

- [ ] **Step 4: Commit infra carry-over**

```bash
git add gsplat/viewer.py gsplat/camera_controls.py gsplat/letterbox.py gsplat/nerfview_viewer.py gsplat/viser_patches.py \
        scripts/derive_camera.py scripts/derive_all_views.py scripts/start_stable_viewer.sh scripts/render_fixed.py scripts/image_diff.py \
        docs/optimization-log/report.css docs/optimization-log/build_report.py \
        gsplat/__main__.py gsplat/rasterization.py
git commit -m "$(cat <<'EOF'
opt-v2: carry over viewer, fixed-camera render, 1024-res, report tooling from opt-stable

Cherry-pick of working tooling from smarton/opt-stable into a fresh opt-v2
branch off origin/main. Includes the viser-based viewer, render_fixed.py +
image_diff.py for benchmark/validation, build_report.py + report.css for the
optimization log, and the 1024-square rasterization flag.

No kernel changes vs main; alpha_blend_compute.cpp remains at main's
~106 ms baseline. iter-0 baseline render happens in a later commit.
EOF
)"
git log --oneline -3
```

Expected: 1 new commit on opt-v2.

---

### Task 3: Add anti-tampering hook and bypass-permissions in `.claude/settings.local.json`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/.claude/settings.local.json`
- Modify: `/Users/smarton/dev/gsplat_tt/.gitignore` (add `.claude/settings.local.json` if not already covered)

- [ ] **Step 1: Verify `.gitignore` policy for `.claude/settings.local.json`**

```bash
grep -n settings.local /Users/smarton/dev/gsplat_tt/.gitignore || echo "not gitignored"
grep -n "^.claude" /Users/smarton/dev/gsplat_tt/.gitignore || echo "no .claude rule"
```

If neither is present, add `.claude/settings.local.json` to `.gitignore`.

- [ ] **Step 2: Create `.claude/settings.local.json`**

```json
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | grep -q '^.*benchmarks/reference/.*\\.png$' && echo '{\"decision\":\"block\",\"reason\":\"benchmarks/reference/*.png are frozen at iter-0; do not modify\"}' || true"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Pipe-test the hook command before relying on it**

```bash
echo '{"tool_name":"Edit","tool_input":{"file_path":"/Users/smarton/dev/gsplat_tt/benchmarks/reference/stitch_hero.png"}}' \
  | jq -r '.tool_input.file_path' \
  | grep -q '^.*benchmarks/reference/.*\.png$' \
  && echo '{"decision":"block","reason":"benchmarks/reference/*.png are frozen at iter-0; do not modify"}' || true
```

Expected output: `{"decision":"block","reason":"benchmarks/reference/*.png are frozen at iter-0; do not modify"}`

Then test a path that should NOT block:
```bash
echo '{"tool_name":"Edit","tool_input":{"file_path":"/Users/smarton/dev/gsplat_tt/scripts/run_iter.sh"}}' \
  | jq -r '.tool_input.file_path' \
  | grep -q '^.*benchmarks/reference/.*\.png$' \
  && echo '{"decision":"block","reason":"...","reason2":"..."}' || true
```

Expected output: empty (no block).

- [ ] **Step 4: Validate JSON schema**

```bash
jq -e '.hooks.PreToolUse[] | select(.matcher == "Write|Edit") | .hooks[] | select(.type == "command") | .command' \
  /Users/smarton/dev/gsplat_tt/.claude/settings.local.json
```

Expected: exit 0 + prints the command string.

- [ ] **Step 5: Commit (gitignore only, not the settings.local.json)**

```bash
git add .gitignore
git commit -m "opt-v2: gitignore .claude/settings.local.json"
git log --oneline -2
```

---

### Task 4: Wipe stale optimization-log artifacts and initialize loop state files

**Files:**
- Delete: `docs/optimization-log/screenshots/`, all `iter-*.md` in `docs/optimization-log/`, `REPORT.md`, `SUMMARY.md`, `SUPERVISOR-LOOP.md` (if present)
- Create: `docs/optimization-log/STATUS.md`, `docs/optimization-log/BACKBURNER.md`, `docs/optimization-log/iters.jsonl`
- Modify: `.gitignore` to add `.opt-v2-last-build-commit`

- [ ] **Step 1: Inspect current state of optimization-log**

```bash
ls /Users/smarton/dev/gsplat_tt/docs/optimization-log/
```

- [ ] **Step 2: Remove stale artifacts**

```bash
cd /Users/smarton/dev/gsplat_tt/docs/optimization-log
rm -rf screenshots/
rm -f REPORT.md REPORT.html SUMMARY.md SUPERVISOR-LOOP.md
rm -f iter-*.md
ls
```

Expected output: only `build_report.py`, `report.css` remain (plus any other carry-over files).

- [ ] **Step 3: Initialize empty loop-state files**

```bash
cd /Users/smarton/dev/gsplat_tt/docs/optimization-log
cat > STATUS.md <<'EOF'
# Supervisor Status

## ESCALATIONS (read these first)

_None._

## Current State

- Loop: not yet started
- Branch: smarton/opt-v2
- iter-0 baseline: not yet run
- Current best kernel ms: n/a
- Target: 1.0 ms
EOF

cat > BACKBURNER.md <<'EOF'
# BACKBURNER

Parked experiments — REJECT or NEEDS_REVIEW iters that the user may want to promote.

_None yet._
EOF

: > iters.jsonl
```

- [ ] **Step 4: Add `.opt-v2-last-build-commit` to `.gitignore`**

```bash
echo ".opt-v2-last-build-commit" >> /Users/smarton/dev/gsplat_tt/.gitignore
git diff .gitignore
```

- [ ] **Step 5: Commit**

```bash
git add docs/optimization-log/STATUS.md docs/optimization-log/BACKBURNER.md docs/optimization-log/iters.jsonl .gitignore
# stale removals get picked up by git automatically once committed; explicitly stage deletions
git add -u docs/optimization-log/
git commit -m "$(cat <<'EOF'
opt-v2: wipe stale optimization-log artifacts; init STATUS, BACKBURNER, iters.jsonl

Removes REPORT.md / iter-*.md / SUMMARY.md / SUPERVISOR-LOOP.md /
screenshots/ from the opt-stable carry-over so the new autonomous loop
starts with a clean optimization-log. Initializes three empty state files
the loop will append to.
EOF
)"
git log --oneline -3
```

---

### Task 5: Copy the design spec into the opt-v2 branch

**Files:**
- Source: existing spec at `/Users/smarton/dev/gsplat_tt/docs/superpowers/specs/2026-05-25-gsplat-tt-1ms-design.md` (currently uncommitted, written before branch creation)

- [ ] **Step 1: Verify the spec file exists**

```bash
ls -la /Users/smarton/dev/gsplat_tt/docs/superpowers/specs/
```

- [ ] **Step 2: Commit the spec to opt-v2**

```bash
cd /Users/smarton/dev/gsplat_tt
git add docs/superpowers/specs/2026-05-25-gsplat-tt-1ms-design.md
git commit -m "opt-v2: add 1 ms autonomous-loop design spec"
git log --oneline -4
```

---

## Phase 2 — `compute_metrics.py` (fp64 PSNR + per-view aggregation)

`scripts/image_diff.py` already does single-image PSNR + diff×10. The new `scripts/compute_metrics.py` is a thin wrapper that aggregates across 3 views and reports the metrics schema the validator and report consume.

### Task 6: Write failing test for `compute_metrics.py`

**Files:**
- Create: `tests/test_compute_metrics.py`
- Create: `tests/fixtures/iter-test-good/{hero,side,top}.png` (synthetic), `tests/fixtures/iter-test-good/timing.jsonl`
- Create: `tests/fixtures/reference/stitch_{hero,side,top}.png` (synthetic)

- [ ] **Step 1: Create fixture generator**

```bash
mkdir -p /Users/smarton/dev/gsplat_tt/tests/fixtures/iter-test-good
mkdir -p /Users/smarton/dev/gsplat_tt/tests/fixtures/reference
```

Create `tests/fixtures/_make_fixtures.py`:

```python
"""One-shot script to generate synthetic test fixtures."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def make_pair(seed, name, identical=True):
    rng = np.random.default_rng(seed)
    ref = (rng.random((1024, 1024, 3)) * 255).astype(np.uint8)
    Image.fromarray(ref).save(ROOT / "reference" / f"stitch_{name}.png")
    cand = ref.copy() if identical else (ref + rng.integers(0, 2, size=ref.shape, dtype=np.int16)).clip(0, 255).astype(np.uint8)
    Image.fromarray(cand).save(ROOT / "iter-test-good" / f"{name}.png")

for n, s in [("hero", 1), ("side", 2), ("top", 3)]:
    make_pair(s, n, identical=False)  # tiny diff → PSNR very high but not inf

timing = [{"frame": i, "view": v, "kernel_ms": 16.5 + (i % 3) * 0.1}
          for i in range(30) for v in ["hero", "side", "top"][:1]]  # 30 frames hero only for fixture simplicity
# Actually: 30 frames total, alternating views per cycle of 3
timing = []
views = ["hero", "side", "top"]
for cycle in range(10):
    for v in views:
        timing.append({"cycle": cycle, "view": v, "kernel_ms": 16.5 + (cycle % 3) * 0.1})
(ROOT / "iter-test-good" / "timing.jsonl").write_text(
    "\n".join(json.dumps(t) for t in timing)
)
```

Run:
```bash
cd /Users/smarton/dev/gsplat_tt && python tests/fixtures/_make_fixtures.py
ls tests/fixtures/iter-test-good/ tests/fixtures/reference/
```

Expected: `hero.png side.png top.png timing.jsonl` in iter-test-good; `stitch_hero.png stitch_side.png stitch_top.png` in reference.

- [ ] **Step 2: Write `tests/test_compute_metrics.py`**

```python
"""Tests for scripts/compute_metrics.py."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compute_metrics.py"
FIXTURES = REPO / "tests" / "fixtures"
REF_DIR = FIXTURES / "reference"
ITER_DIR = FIXTURES / "iter-test-good"


def run_compute(iter_dir: Path, ref_dir: Path, class_tag: str = "kernel-algebra", prev_best: float = 20.0):
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--iter-dir", str(iter_dir),
         "--reference-dir", str(ref_dir),
         "--class", class_tag,
         "--prev-best-ms", str(prev_best)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    return json.loads((iter_dir / "metrics.json").read_text())


def test_metrics_schema(tmp_path):
    # Copy fixtures so tests don't mutate the source tree
    import shutil
    iter_copy = tmp_path / "iter-test"
    shutil.copytree(ITER_DIR, iter_copy)
    metrics = run_compute(iter_copy, REF_DIR)

    assert "kernel_ms_median" in metrics
    assert "kernel_ms_p99" in metrics
    assert "per_view_median" in metrics
    assert set(metrics["per_view_median"].keys()) == {"hero", "side", "top"}
    assert "psnr_per_view" in metrics
    assert set(metrics["psnr_per_view"].keys()) == {"hero", "side", "top"}
    assert metrics["class"] == "kernel-algebra"
    assert metrics["prev_best_kernel_ms"] == 20.0


def test_psnr_high_for_near_identical(tmp_path):
    import shutil
    iter_copy = tmp_path / "iter-test"
    shutil.copytree(ITER_DIR, iter_copy)
    metrics = run_compute(iter_copy, REF_DIR)
    # fixtures differ by at most 1 LSB → PSNR should be > 80 dB
    for view, psnr in metrics["psnr_per_view"].items():
        assert psnr > 80.0, f"{view} PSNR = {psnr}, expected > 80"


def test_diff10_images_written(tmp_path):
    import shutil
    iter_copy = tmp_path / "iter-test"
    shutil.copytree(ITER_DIR, iter_copy)
    run_compute(iter_copy, REF_DIR)
    for view in ["hero", "side", "top"]:
        assert (iter_copy / f"{view}_diff10.png").exists(), f"{view}_diff10.png missing"
```

- [ ] **Step 3: Run test, verify it fails (script doesn't exist yet)**

```bash
cd /Users/smarton/dev/gsplat_tt && python -m pytest tests/test_compute_metrics.py -v 2>&1 | tail -20
```

Expected: `FAILED` with FileNotFoundError or similar (script not found).

---

### Task 7: Implement `compute_metrics.py`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/compute_metrics.py`

- [ ] **Step 1: Write the script**

```python
"""Aggregate per-view PSNR + timing metrics for one iter.

Reads:  <iter_dir>/{hero,side,top}.png, <iter_dir>/timing.jsonl,
        <reference_dir>/stitch_{hero,side,top}.png
Writes: <iter_dir>/{hero,side,top}_diff10.png, <iter_dir>/metrics.json

PSNR computed in fp64 so >100 dB is representable. timing.jsonl format:
one JSON object per measured frame with at minimum `view` and `kernel_ms`.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


VIEWS = ("hero", "side", "top")


def load_rgb_fp64(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr_fp64(ref: np.ndarray, cand: np.ndarray) -> float:
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def write_diff10(ref: np.ndarray, cand: np.ndarray, out: Path) -> None:
    """10x amplified absolute diff, clipped to [0,1], saved as 8-bit PNG."""
    amp = np.clip(np.abs(ref - cand) * 10.0, 0.0, 1.0)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save(out)


def aggregate_timing(timing_path: Path) -> dict:
    rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    all_ms = [r["kernel_ms"] for r in rows]
    per_view = {v: [r["kernel_ms"] for r in rows if r.get("view") == v] for v in VIEWS}
    return {
        "kernel_ms_median": float(statistics.median(all_ms)),
        "kernel_ms_p99": float(np.percentile(all_ms, 99)),
        "per_view_median": {v: (float(statistics.median(per_view[v])) if per_view[v] else float("nan")) for v in VIEWS},
        "frame_count": len(all_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--class", dest="class_tag", required=True)
    ap.add_argument("--prev-best-ms", required=True, type=float)
    args = ap.parse_args()

    psnr = {}
    for v in VIEWS:
        ref = load_rgb_fp64(args.reference_dir / f"stitch_{v}.png")
        cand = load_rgb_fp64(args.iter_dir / f"{v}.png")
        assert ref.shape == cand.shape, f"shape mismatch for {v}: ref {ref.shape} vs cand {cand.shape}"
        psnr[v] = psnr_fp64(ref, cand)
        write_diff10(ref, cand, args.iter_dir / f"{v}_diff10.png")

    timing = aggregate_timing(args.iter_dir / "timing.jsonl")

    # Optional: per-zone Tracy data if --profile was used.
    tracy_zones = None
    zones_csv = args.iter_dir / "zones.csv"
    if zones_csv.exists():
        import csv
        with zones_csv.open() as f:
            tracy_zones = [
                {"name": r["name"], "avg_ns": float(r["avg_ns"]), "count": int(r["count"])}
                for r in csv.DictReader(f)
                if r.get("name") and r.get("avg_ns")
            ]

    metrics = {
        "iter_dir": str(args.iter_dir.name),
        "class": args.class_tag,
        "prev_best_kernel_ms": args.prev_best_ms,
        "psnr_per_view": psnr,
        **timing,
    }
    if tracy_zones is not None:
        metrics["tracy_zones"] = tracy_zones
    (args.iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests, verify they pass**

```bash
cd /Users/smarton/dev/gsplat_tt && python -m pytest tests/test_compute_metrics.py -v 2>&1 | tail -20
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/compute_metrics.py tests/test_compute_metrics.py tests/fixtures/
git commit -m "opt-v2: compute_metrics.py — fp64 PSNR + per-view aggregation"
```

---

## Phase 3 — `render_fixed.py` training-pattern extension

`render_fixed.py` today renders one view with `--warmup N --frames M`. We add a `--cycles` mode that loops `[hero, side, top]` and writes one combined `timing.jsonl` plus three final PNGs.

### Task 8: Add `--cycles` mode to `render_fixed.py`

**Files:**
- Modify: `scripts/render_fixed.py`

- [ ] **Step 1: Read the existing render_fixed.py to find the per-view rendering function**

```bash
wc -l /Users/smarton/dev/gsplat_tt/scripts/render_fixed.py
grep -n "^def " /Users/smarton/dev/gsplat_tt/scripts/render_fixed.py
```

Identify the function that renders a single view given a camera and pipeline; call it `render_one_view`. (If named differently, substitute below.)

- [ ] **Step 2: Add `--cycles` arg and a new `main_cycles()` entry path**

At the bottom of `render_fixed.py`, before `if __name__ == "__main__":`, append:

```python
def main_cycles(args):
    """Training-pattern measurement: cycle [hero, side, top] × N times.

    Args we accept (parsed in main()):
      --scene <scene>
      --measure-cycles N  (default 10)
      --warmup-cycles W   (default 1)
      --out-dir <dir>     (writes <view>.png and timing.jsonl into this dir)
    """
    import json
    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    views = ("hero", "side", "top")

    # Build pipeline once — scene is view-invariant
    pipeline = build_pipeline(args.scene, args.backend, args.ply, args.size)
    cameras = {v: load_camera(args.scene, v, args.camera_file) for v in views}

    timing_rows = []
    final_imgs = {}
    total_cycles = args.warmup_cycles + args.measure_cycles
    for cycle in range(total_cycles):
        measured = cycle >= args.warmup_cycles
        for v in views:
            img, kernel_ms = render_one_view(pipeline, cameras[v])
            if measured:
                timing_rows.append({"cycle": cycle - args.warmup_cycles, "view": v, "kernel_ms": float(kernel_ms)})
                final_imgs[v] = img  # last measured render per view wins (deterministic)

    for v, img in final_imgs.items():
        from PIL import Image
        Image.fromarray(img).save(out_dir / f"{v}.png")

    (out_dir / "timing.jsonl").write_text("\n".join(json.dumps(r) for r in timing_rows))
    print(json.dumps({"frames_measured": len(timing_rows), "out_dir": str(out_dir)}))
```

Then in `main()` (or wherever args are parsed), add to the argparse setup:

```python
    p.add_argument("--cycles", action="store_true",
                   help="Training-pattern mode: cycle [hero,side,top] × N times")
    p.add_argument("--scene", default="stitch")
    p.add_argument("--measure-cycles", type=int, default=10)
    p.add_argument("--warmup-cycles", type=int, default=1)
    p.add_argument("--out-dir", default=None)
```

And in `main()` body, before the existing single-view logic:

```python
    if args.cycles:
        if not args.out_dir:
            raise SystemExit("--cycles requires --out-dir")
        return main_cycles(args)
```

**Note for engineer:** `build_pipeline` and `render_one_view` and `load_camera` are pseudo-names. Use the actual factory functions in render_fixed.py — they exist as the body of the existing main() flow. Refactor by extracting them if not already separate functions.

- [ ] **Step 3: Smoke test on Mac (cpu backend, tiny size)**

```bash
cd /Users/smarton/dev/gsplat_tt
# Use cpu backend + smaller size for local sanity; tt backend requires bh-14
python scripts/render_fixed.py --cycles --backend cpu --scene stitch --size 256 \
       --warmup-cycles 1 --measure-cycles 2 --out-dir /tmp/cycles_smoke
ls /tmp/cycles_smoke/
cat /tmp/cycles_smoke/timing.jsonl | head
```

Expected: `hero.png side.png top.png timing.jsonl` created; timing.jsonl has 6 lines (2 measured cycles × 3 views).

If `--backend cpu` doesn't exist or the scene doesn't load: skip the smoke test and rely on the bh-14 integration in Phase 10.

- [ ] **Step 4: Commit**

```bash
git add scripts/render_fixed.py
git commit -m "opt-v2: render_fixed.py — add --cycles training-pattern mode"
```

---

## Phase 4 — `dispatch_validator.sh` + validator prompt template

The validator is a Sonnet 4.6 subagent invoked fresh per iter with only the artifacts. It reads `prompts/validator.md` (the rules) and the per-iter files, returns JSON to `<iter_dir>/validator.json`.

### Task 9: Write `prompts/validator.md`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/prompts/validator.md`

- [ ] **Step 1: Create the prompts directory and validator prompt**

```bash
mkdir -p /Users/smarton/dev/gsplat_tt/prompts
```

Write `prompts/validator.md`:

```markdown
# Validator Subagent — gsplat_tt iter judgment

You are the validator for one iteration of the gsplat_tt autonomous optimization loop. You have no memory of prior iters. You see only the artifacts listed below. Decide KEEP / REJECT / NEEDS_REVIEW per the §5 rules in the design spec.

You MUST cite specific check failures. "Looks okay" is not a valid reasoning.

## Inputs (file paths provided in the calling prompt)

- 3 rendered PNGs: `{iter_dir}/{hero,side,top}.png`
- 3 frozen reference PNGs: `{ref_dir}/stitch_{hero,side,top}.png`
- 3 diff×10 images: `{iter_dir}/{hero,side,top}_diff10.png`
- `{iter_dir}/metrics.json` — `{kernel_ms_median, kernel_ms_p99, per_view_median, psnr_per_view, prev_best_kernel_ms, class}`

## Visual checks (primary gate — any ✗ on a structural-artifact check = REJECT, regardless of numbers)

For each rendered PNG and its diff×10 image:

1. **Tile grid seams?** Diff×10 shows horizontal/vertical lines at 32-pixel or 16-pixel spacing → REJECT.
2. **Tile-shaped uniform-fill blocks?** Square 32×32 or 16×16 regions that are constant color where reference has detail → REJECT.
3. **Missing-splat black holes?** Local regions in render darker than reference where there should be coverage (bright spots in diff×10) → REJECT if hole > ~16 pixels in any dimension; smaller speckle is acceptable noise.
4. **Color channel clipping bands?** Saturated R/G/B regions not in reference, often flat colored stripes → REJECT.
5. **Ringing / halos at high-contrast edges?** Structured ringing at object silhouettes (not just edge noise) → REJECT.
6. **NaN/Inf signatures?** Pixels of pure black/white/magenta in spatially-correlated patches → REJECT.
7. **Geometry shift?** Diff×10 shows the silhouette of an object → REJECT (kernel producing wrong coordinates).
8. **Diff×10 structure check:** Diff×10 dominated by uniform speckle (✓ ok) vs. structured patterns — grids, edges, gradients, bands (✗ REJECT)?

## Numeric checks (only after visual gate passes)

Class PSNR floor:

| Class | Floor | Below floor |
|---|---|---|
| `kernel-algebra` | >100 dB any view | NEEDS_REVIEW |
| `precompute` | >100 dB | NEEDS_REVIEW |
| `dispatch` | >100 dB | NEEDS_REVIEW |
| `binning` | >50 dB | KEEP if visuals pass; below 50 → NEEDS_REVIEW |
| `sort` | >50 dB | KEEP if visuals pass; below 50 → NEEDS_REVIEW |
| `host-prep` | >50 dB | KEEP if visuals pass; below 50 → NEEDS_REVIEW |

Per-view consistency:
- Max per-view PSNR delta >20 dB → NEEDS_REVIEW.
- Max per-view kernel-ms ratio >2× → NEEDS_REVIEW.

Timing:
- `kernel_ms_median ≤ prev_best × 1.02` → progress/break-even.
- `kernel_ms_p99 > kernel_ms_median × 3` → NEEDS_REVIEW (suspicious tail).

## Required output

Write ONLY this JSON to stdout, nothing else:

```json
{
  "verdict": "KEEP" | "REJECT" | "NEEDS_REVIEW",
  "visual_checks": [
    {"name": "tile_grid_seams", "result": "pass" | "fail", "evidence": "..."},
    {"name": "tile_uniform_fill", "result": "pass" | "fail", "evidence": "..."},
    {"name": "missing_splat_holes", "result": "pass" | "fail", "evidence": "..."},
    {"name": "color_clipping_bands", "result": "pass" | "fail", "evidence": "..."},
    {"name": "ringing_halos", "result": "pass" | "fail", "evidence": "..."},
    {"name": "nan_inf_signatures", "result": "pass" | "fail", "evidence": "..."},
    {"name": "geometry_shift", "result": "pass" | "fail", "evidence": "..."},
    {"name": "diff10_structure", "result": "pass" | "fail", "evidence": "..."}
  ],
  "psnr_check": {"floor": 100.0, "actual": {"hero": ..., "side": ..., "top": ...}, "pass": true | false},
  "per_view_consistency": {"max_psnr_delta_db": ..., "max_ms_ratio": ..., "pass": true | false},
  "timing": {"median_ms": ..., "p99_ms": ..., "vs_prev_best_pct": ..., "pass": true | false},
  "reasoning": "one paragraph citing specific check failures if any"
}
```

If you cannot read an artifact file: still emit the JSON with `verdict: REJECT` and `reasoning` citing which file failed.
```

- [ ] **Step 2: Commit**

```bash
git add prompts/validator.md
git commit -m "opt-v2: validator subagent prompt template"
```

---

### Task 10: Write `dispatch_validator.sh`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/dispatch_validator.sh`

- [ ] **Step 1: Write the dispatcher**

```bash
#!/usr/bin/env bash
# Usage: scripts/dispatch_validator.sh <iter_dir>
#
# Dispatches the validator subagent (Sonnet 4.6, fresh context) with only the
# artifacts in <iter_dir> and the rules in prompts/validator.md. Writes the
# validator's JSON response to <iter_dir>/validator.json. Re-dispatches once
# if response is malformed; still malformed -> writes MALFORMED_VALIDATOR
# sentinel and a REJECT verdict.
set -euo pipefail

ITER_DIR="${1:?usage: $0 <iter_dir>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT="$REPO/prompts/validator.md"
REF_DIR="$REPO/benchmarks/reference"
OUT="$ITER_DIR/validator.json"

if [[ ! -f "$ITER_DIR/metrics.json" ]]; then
  echo "ERROR: $ITER_DIR/metrics.json missing" >&2
  exit 1
fi

# Build the validator subagent prompt: validator.md + concrete file paths.
build_prompt() {
  cat "$PROMPT"
  echo ""
  echo "## Concrete artifact paths for this iter"
  echo ""
  echo "- iter_dir: $ITER_DIR"
  echo "- ref_dir: $REF_DIR"
  echo "- renders: $ITER_DIR/hero.png $ITER_DIR/side.png $ITER_DIR/top.png"
  echo "- references: $REF_DIR/stitch_hero.png $REF_DIR/stitch_side.png $REF_DIR/stitch_top.png"
  echo "- diff10: $ITER_DIR/hero_diff10.png $ITER_DIR/side_diff10.png $ITER_DIR/top_diff10.png"
  echo "- metrics: $ITER_DIR/metrics.json"
  echo ""
  echo "Read the metrics JSON inline:"
  echo '```json'
  cat "$ITER_DIR/metrics.json"
  echo '```'
  echo ""
  echo "Now produce the required JSON output. Nothing else."
}

invoke_validator() {
  # The supervisor invokes this via the Agent tool with subagent_type=general-purpose
  # or by spawning a `claude -p` headless subprocess. This script is a thin shim
  # that returns the prompt for the supervisor to feed to the agent.
  build_prompt > "$ITER_DIR/.validator_prompt.md"
  echo "VALIDATOR_PROMPT_READY: $ITER_DIR/.validator_prompt.md"
}

validate_json_schema() {
  local file="$1"
  jq -e '.verdict | IN("KEEP", "REJECT", "NEEDS_REVIEW")' "$file" >/dev/null 2>&1 && \
  jq -e '.visual_checks | length == 8' "$file" >/dev/null 2>&1 && \
  jq -e '.psnr_check.pass | type == "boolean"' "$file" >/dev/null 2>&1 && \
  jq -e '.reasoning | length > 10' "$file" >/dev/null 2>&1
}

# The supervisor calls dispatch_validator.sh in two modes:
#   Mode A: --prepare-prompt → just emit the prompt path (supervisor handles agent dispatch)
#   Mode B: --validate-response <response.json> → check schema, return PASS or REPROMPT or REJECT

case "${2:---prepare-prompt}" in
  --prepare-prompt)
    invoke_validator
    ;;
  --validate-response)
    RESPONSE="${3:?--validate-response requires response file path}"
    if validate_json_schema "$RESPONSE"; then
      cp "$RESPONSE" "$OUT"
      echo "VALIDATOR_OK: $OUT"
    else
      echo "VALIDATOR_MALFORMED: $RESPONSE" >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: unknown mode $2" >&2; exit 1;;
esac
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /Users/smarton/dev/gsplat_tt/scripts/dispatch_validator.sh
```

- [ ] **Step 3: Smoke test prompt-preparation mode**

```bash
cd /Users/smarton/dev/gsplat_tt
# Use the fixture iter from Task 6/7
ITER_DIR=$(pwd)/tests/fixtures/iter-test-good
# compute metrics first (needed for the prompt to include real metrics)
python scripts/compute_metrics.py --iter-dir "$ITER_DIR" --reference-dir tests/fixtures/reference --class kernel-algebra --prev-best-ms 20.0 >/dev/null
./scripts/dispatch_validator.sh "$ITER_DIR" --prepare-prompt
head -20 "$ITER_DIR/.validator_prompt.md"
```

Expected: `VALIDATOR_PROMPT_READY: ...` then preview of validator.md content.

- [ ] **Step 4: Commit**

```bash
git add scripts/dispatch_validator.sh
git commit -m "opt-v2: dispatch_validator.sh — validator prompt prep + schema check"
```

---

## Phase 5 — `decide_and_log.py`

The reconciliation matrix from §4 of the spec. Reads validator.json + metrics.json, decides KEEP/REJECT/NEEDS_REVIEW action, optionally commits, updates iters.jsonl + BACKBURNER.md + STATUS.md.

### Task 11: Write failing tests for `decide_and_log.py`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/tests/test_decide_and_log.py`

- [ ] **Step 1: Write the test file**

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /Users/smarton/dev/gsplat_tt && python -m pytest tests/test_decide_and_log.py -v 2>&1 | tail -20
```

Expected: 5 FAILED with FileNotFoundError.

---

### Task 12: Implement `decide_and_log.py`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/decide_and_log.py`

- [ ] **Step 1: Write the script**

```python
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
        "class": metrics["class"],
        "commit_sha": commit_sha,
        "high_promotion_priority": decision["high_promotion_priority"],
    }
    atomic_append_jsonl(jsonl, row)

    if decision["action"] == "backburner":
        write_backburner_entry(args.state_dir / "BACKBURNER.md", metrics["iter_dir"], decision, metrics, validator)

    update_status(args.state_dir / "STATUS.md", decision, metrics, jsonl)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests, verify they pass**

```bash
cd /Users/smarton/dev/gsplat_tt && python -m pytest tests/test_decide_and_log.py -v 2>&1 | tail -20
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/decide_and_log.py tests/test_decide_and_log.py
git commit -m "opt-v2: decide_and_log.py — §4 reconciliation + iters.jsonl + STATUS/BACKBURNER updates"
```

---

## Phase 6 — `build_report.py` refactor (REPORT.html + new graphs)

The existing `build_report.py` outputs `REPORT.md` from a hardcoded data table. We refactor it to:
1. Read `iters.jsonl` as source of truth.
2. Emit `REPORT.html` (not .md).
3. Generate 4 graphs: kernel-ms (median+p99), kernel-ms-per-view, psnr-min-with-floors, class-progress.
4. Render per-iter cards with validator JSON checklist.
5. Render BACKBURNER section.

### Task 13: Rewrite `build_report.py` driven by `iters.jsonl`

**Files:**
- Modify (substantial rewrite): `/Users/smarton/dev/gsplat_tt/docs/optimization-log/build_report.py`

- [ ] **Step 1: Read current build_report.py to understand its structure**

```bash
wc -l /Users/smarton/dev/gsplat_tt/docs/optimization-log/build_report.py
head -100 /Users/smarton/dev/gsplat_tt/docs/optimization-log/build_report.py
```

- [ ] **Step 2: Rewrite `build_report.py`**

```python
"""Builds REPORT.html and graph PNGs from iters.jsonl + per-iter screenshot dirs.

Run from the repo root or anywhere:
    python docs/optimization-log/build_report.py

Idempotent; safe to re-run after every iter.
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "optimization-log"
GRAPHS = OUT / "graphs"
SHOTS = OUT / "screenshots"
GRAPHS.mkdir(parents=True, exist_ok=True)

VERDICT_COLOR = {
    "KEEP": "#2ca02c",
    "NEEDS_REVIEW": "#ff7f0e",
    "REJECT": "#d62728",
}
ACTION_COLOR = {
    "commit": "#2ca02c",
    "no_commit_valid_but_not_faster": "#7f7f7f",
    "backburner": "#d62728",
}


def load_iters() -> list[dict]:
    p = OUT / "iters.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def graph_kernel_ms(iters: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(iters)))
    medians = [r.get("kernel_ms_median", float("nan")) for r in iters]
    p99s = [r.get("kernel_ms_p99", float("nan")) for r in iters]
    colors = [ACTION_COLOR.get(r.get("action"), "#999999") for r in iters]
    ax.plot(xs, medians, "-", color="#666666", linewidth=1, label="median")
    ax.plot(xs, p99s, "--", color="#bbbbbb", linewidth=1, label="p99")
    ax.scatter(xs, medians, c=colors, s=40, zorder=3)
    if iters:
        baseline = max((r["kernel_ms_median"] for r in iters if r["iter_dir"].startswith("iter-000")), default=None)
        if baseline:
            ax.axhline(baseline, color="#cccccc", linestyle=":", label=f"iter-0 baseline ({baseline:.1f} ms)")
    ax.axhline(1.0, color="#000000", linestyle=":", linewidth=1, label="target 1.0 ms")
    ax.set_xlabel("iter index")
    ax.set_ylabel("kernel ms")
    ax.set_yscale("log")
    ax.set_title("Kernel ms (median + p99), colored by action")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-kernel-ms.png", dpi=110)
    plt.close(fig)


def graph_kernel_ms_per_view(iters: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(iters)))
    for view, color in [("hero", "#1f77b4"), ("side", "#2ca02c"), ("top", "#d62728")]:
        ys = [r.get("per_view_median", {}).get(view, float("nan")) for r in iters]
        ax.plot(xs, ys, "-o", color=color, markersize=4, label=view)
    ax.axhline(1.0, color="#000000", linestyle=":", linewidth=1, label="target")
    ax.set_xlabel("iter index")
    ax.set_ylabel("kernel ms (per-view median)")
    ax.set_yscale("log")
    ax.set_title("Per-view kernel ms median")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-kernel-ms-per-view.png", dpi=110)
    plt.close(fig)


def graph_psnr(iters: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(iters)))
    mins = [min(r.get("psnr_per_view", {"x": 0}).values()) for r in iters]
    colors = [ACTION_COLOR.get(r.get("action"), "#999999") for r in iters]
    ax.scatter(xs, mins, c=colors, s=40)
    ax.axhline(100.0, color="#000000", linestyle="--", linewidth=1, label="kernel-algebra floor (100 dB)")
    ax.axhline(50.0, color="#888888", linestyle="--", linewidth=1, label="binning/sort floor (50 dB)")
    ax.set_xlabel("iter index")
    ax.set_ylabel("min PSNR across 3 views (dB)")
    ax.set_title("PSNR min per iter with class floors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-psnr.png", dpi=110)
    plt.close(fig)


def graph_class_progress(iters: list[dict]) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in iters:
        counts[r.get("class", "unknown")][r.get("action", "unknown")] += 1
    classes = list(counts.keys())
    actions = ["commit", "no_commit_valid_but_not_faster", "backburner"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(classes))
    for action in actions:
        ys = np.array([counts[c].get(action, 0) for c in classes])
        ax.bar(classes, ys, bottom=bottom, label=action, color=ACTION_COLOR.get(action, "#999999"))
        bottom += ys
    ax.set_xlabel("optimization class")
    ax.set_ylabel("iter count")
    ax.set_title("Iters per class × action")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "graph-class-progress.png", dpi=110)
    plt.close(fig)


def render_card(r: dict) -> str:
    iter_name = r["iter_dir"]
    shot_dir = f"screenshots/{iter_name}"
    psnr = r.get("psnr_per_view", {})
    per_view = r.get("per_view_median", {})
    psnr_min = min(psnr.values()) if psnr else float("nan")
    badge_color = ACTION_COLOR.get(r.get("action"), "#999999")
    badge = r.get("action", "unknown")
    verdict_badge = r.get("verdict", "?")

    # Try to load validator JSON for the checklist render
    val_path = OUT / "screenshots" / iter_name / "validator.json"
    if val_path.exists():
        val = json.loads(val_path.read_text())
        checks_html = "".join(
            f"<li>{'✓' if c['result'] == 'pass' else '✗'} {escape(c['name'])}: {escape(c.get('evidence',''))[:120]}</li>"
            for c in val.get("visual_checks", [])
        )
        reasoning = escape(val.get("reasoning", ""))
    else:
        checks_html = "<li>(no validator.json)</li>"
        reasoning = ""

    return f"""
<section class="iter-card">
  <header>
    <h3>{escape(iter_name)}
      <span class="badge" style="background:{badge_color}">{escape(badge)}</span>
      <span class="badge verdict">{escape(verdict_badge)}</span>
      <span class="class-tag">{escape(r.get('class', '?'))}</span>
      {f'<a href="../../{escape(r["iter_dir"])}.md">md</a>' if r.get('commit_sha') else ''}
      <span class="ts">{escape(r.get('timestamp', ''))}</span>
    </h3>
  </header>
  <div class="metrics">
    <div class="ms">kernel: <b>{r.get('kernel_ms_median', float('nan')):.2f} ms</b> median · {r.get('kernel_ms_p99', float('nan')):.2f} ms p99</div>
    <div class="per-view">per-view: hero {per_view.get('hero', float('nan')):.2f} / side {per_view.get('side', float('nan')):.2f} / top {per_view.get('top', float('nan')):.2f}</div>
    <div class="psnr">PSNR: hero {psnr.get('hero', float('nan')):.1f} / side {psnr.get('side', float('nan')):.1f} / top {psnr.get('top', float('nan')):.1f} (min <b>{psnr_min:.1f}</b>)</div>
  </div>
  <div class="thumbs">
    <figure><img src="{shot_dir}/hero.png" alt="hero"><figcaption>hero</figcaption></figure>
    <figure><img src="{shot_dir}/side.png" alt="side"><figcaption>side</figcaption></figure>
    <figure><img src="{shot_dir}/top.png" alt="top"><figcaption>top</figcaption></figure>
    <figure><img src="{shot_dir}/hero_diff10.png" alt="hero diff10"><figcaption>hero × 10</figcaption></figure>
    <figure><img src="{shot_dir}/side_diff10.png" alt="side diff10"><figcaption>side × 10</figcaption></figure>
    <figure><img src="{shot_dir}/top_diff10.png" alt="top diff10"><figcaption>top × 10</figcaption></figure>
  </div>
  <details>
    <summary>Validator checks ({verdict_badge})</summary>
    <ul>{checks_html}</ul>
    <p class="reasoning">{reasoning}</p>
  </details>
</section>
"""


def render_html(iters: list[dict]) -> str:
    iter_count = len(iters)
    committed = [r for r in iters if r.get("action") == "commit"]
    best = min((r["kernel_ms_median"] for r in committed), default=float("inf"))
    best_str = f"{best:.2f} ms" if best != float("inf") else "n/a"
    cards = "\n".join(render_card(r) for r in reversed(iters))

    backburner_md_path = OUT / "BACKBURNER.md"
    backburner_html = ""
    if backburner_md_path.exists():
        # Trivial md → html pass: pre-format
        backburner_html = "<pre>" + escape(backburner_md_path.read_text()) + "</pre>"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>gsplat_tt opt-v2 — autonomous loop</title>
  <link rel="stylesheet" href="report.css">
</head>
<body>
  <header class="top">
    <h1>gsplat_tt opt-v2 — autonomous loop</h1>
    <div class="topbar">
      iters: {iter_count} · current best: {best_str} · target: 1.0 ms
    </div>
    <p><a href="STATUS.md">STATUS</a> · <a href="BACKBURNER.md">BACKBURNER</a> · <a href="iters.jsonl">iters.jsonl</a></p>
  </header>
  <section class="graphs">
    <img src="graphs/graph-kernel-ms.png" alt="kernel ms">
    <img src="graphs/graph-kernel-ms-per-view.png" alt="per-view kernel ms">
    <img src="graphs/graph-psnr.png" alt="psnr">
    <img src="graphs/graph-class-progress.png" alt="class progress">
  </section>
  <section class="iters">
    <h2>Per-iter cards (newest first)</h2>
    {cards}
  </section>
  <section class="backburner">
    <h2>BACKBURNER</h2>
    {backburner_html}
  </section>
</body>
</html>
"""


def main() -> None:
    iters = load_iters()
    if iters:
        graph_kernel_ms(iters)
        graph_kernel_ms_per_view(iters)
        graph_psnr(iters)
        graph_class_progress(iters)
    (OUT / "REPORT.html").write_text(render_html(iters))
    print(f"built REPORT.html for {len(iters)} iters")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test with empty iters.jsonl**

```bash
cd /Users/smarton/dev/gsplat_tt
python docs/optimization-log/build_report.py
ls docs/optimization-log/REPORT.html
head -20 docs/optimization-log/REPORT.html
```

Expected: `REPORT.html` produced; graphs not generated (empty iters).

- [ ] **Step 4: Commit**

```bash
git add docs/optimization-log/build_report.py
git commit -m "opt-v2: rewrite build_report.py — REPORT.html driven by iters.jsonl + 4 graphs"
```

---

## Phase 7 — `run_iter.sh` end-to-end iter driver

### Task 14: Write `run_iter.sh`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/run_iter.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Single-command worker entry point for one iter.
#
# Usage: scripts/run_iter.sh <iter_num> <slug> <class> [--profile]
#   <iter_num>  e.g. 1, 2, 47
#   <slug>      kebab-case label, e.g. "dst-resident-state"
#   <class>     one of: kernel-algebra | precompute | dispatch | binning | sort | host-prep
#   --profile   (optional) build with Tracy enabled and capture per-zone timings.
#               Adds ~5-10% kernel overhead, so OFF by default. Use when an iter
#               is surprising or we plateau and need per-zone attribution.
#
# Steps:
#   1. clean tree check
#   2. devsync gate
#   3. JIT cache wipe if needed
#   4. remote build (Tracy-enabled if --profile)
#   5. remote render (training-pattern cycles; Tracy capture if --profile)
#   6. scp results back (+ .tracy + zones.csv if --profile)
#   7. compute_metrics locally (merges zones.csv if present)
#   8. (supervisor invokes dispatch_validator separately)
#   9. (supervisor invokes decide_and_log separately)
#  10. build_report
#
# Returns exit 0 + path to metrics.json on stdout. Non-zero exit = the experiment
# failed before producing artifacts; supervisor handles per §4 decision matrix.
set -euo pipefail

ITER_NUM="${1:?usage: $0 <iter_num> <slug> <class> [--profile]}"
SLUG="${2:?slug required}"
CLASS="${3:?class required}"
PROFILE=0
if [[ "${4:-}" == "--profile" ]]; then PROFILE=1; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ITER_NAME="$(printf "iter-%03d-%s" "$ITER_NUM" "$SLUG")"
ITER_DIR="$REPO/docs/optimization-log/screenshots/$ITER_NAME"
SENTINEL="$REPO/.opt-v2-last-build-commit"
BOX_USER="smarton"
BOX_HOST="yyzo-bh-14"
REMOTE_REPO="/proj_sw/user_dev/smarton/gsplat_tt"

mkdir -p "$ITER_DIR"

# Step 1: clean tree check
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "ERROR: working tree not clean; supervisor must resolve before next iter" >&2
  git -C "$REPO" status --short >&2
  exit 10
fi

# Step 2: devsync gate
if ! devsync is-finished "$BOX_HOST"; then
  echo "ERROR: devsync to $BOX_HOST not finished" >&2
  exit 11
fi

# Step 3: JIT cache wipe if kernel cpp/hpp or CT-args header changed since last build
LAST_BUILD="$(cat "$SENTINEL" 2>/dev/null || echo "")"
CURR="$(git -C "$REPO" rev-parse HEAD)"
NEEDS_WIPE=0
if [[ -z "$LAST_BUILD" ]]; then
  NEEDS_WIPE=1  # first build on opt-v2
else
  CHANGED="$(git -C "$REPO" diff --name-only "$LAST_BUILD" "$CURR" -- \
    'backends/tt/tt-metal/programming_examples/gaussian_splatting/*.cpp' \
    'backends/tt/tt-metal/programming_examples/gaussian_splatting/*.hpp' || true)"
  if [[ -n "$CHANGED" ]]; then
    NEEDS_WIPE=1
  fi
fi
if [[ "$NEEDS_WIPE" == "1" ]]; then
  echo "[run_iter] wiping JIT cache on $BOX_HOST"
  ssh "$BOX_USER@$BOX_HOST" 'rm -rf /localdev/smarton/.cache/tt-metal-cache/' >>"$ITER_DIR/build.log" 2>&1
fi

# Step 4: remote build
echo "[run_iter] building on $BOX_HOST (profile=$PROFILE)"
if [[ "$PROFILE" == "1" ]]; then
  # Tracy-enabled build into a separate build dir to avoid thrashing the fast binary.
  # If tt-metal's Tracy flag name differs, adjust here (commonly ENABLE_TRACY or
  # TT_METAL_ENABLE_TRACING). Build dir kept separate so non-profile iters stay fast.
  BUILD_CMD="cd $REMOTE_REPO && sudo cmake -S backends/tt/tt-metal -B backends/tt/tt-metal/build-tracy -DENABLE_TRACY=ON -DCMAKE_BUILD_TYPE=Release >/dev/null && sudo ninja -C backends/tt/tt-metal/build-tracy metal_example_gaussian_splatting"
else
  BUILD_CMD="cd $REMOTE_REPO && sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting"
fi
timeout 240 ssh "$BOX_USER@$BOX_HOST" "$BUILD_CMD" \
  >>"$ITER_DIR/build.log" 2>&1 || { touch "$ITER_DIR/BUILD_FAIL"; exit 2; }
echo "$CURR" > "$SENTINEL"

# Step 5: remote render (training-pattern cycles)
echo "[run_iter] rendering on $BOX_HOST"
REMOTE_OUT="/tmp/$ITER_NAME"
ssh "$BOX_USER@$BOX_HOST" "mkdir -p $REMOTE_OUT"
if [[ "$PROFILE" == "1" ]]; then
  # Start tracy-capture in the background, then render, then stop capture.
  # tracy-capture listens on TCP 8086 by default; -o writes the .tracy file.
  # Use a remote helper script to keep the inline ssh string sane.
  ssh "$BOX_USER@$BOX_HOST" "
    set -e
    cd $REMOTE_REPO
    export TT_METAL_DEVICE_PROFILER=1
    tracy-capture -o $REMOTE_OUT/iter.tracy -a 127.0.0.1 >/tmp/tracy-cap.log 2>&1 &
    CAP=\$!
    sleep 1
    MESH_DEVICE=P100 python scripts/render_fixed.py --cycles --backend tt --scene stitch --size 1024 --warmup-cycles 1 --measure-cycles 10 --out-dir $REMOTE_OUT --binary backends/tt/tt-metal/build-tracy/programming_examples/metal_example_gaussian_splatting
    # Give tracy-capture a moment to flush, then stop it.
    sleep 2
    kill -TERM \$CAP 2>/dev/null || true
    wait \$CAP 2>/dev/null || true
    tracy-csvexport $REMOTE_OUT/iter.tracy > $REMOTE_OUT/zones.csv
  " >>"$ITER_DIR/run.log" 2>&1 || { touch "$ITER_DIR/DEVICE_FAIL"; exit 3; }
else
  ssh "$BOX_USER@$BOX_HOST" \
    "cd $REMOTE_REPO && MESH_DEVICE=P100 python scripts/render_fixed.py --cycles --backend tt --scene stitch --size 1024 --warmup-cycles 1 --measure-cycles 10 --out-dir $REMOTE_OUT" \
    >>"$ITER_DIR/run.log" 2>&1 || { touch "$ITER_DIR/DEVICE_FAIL"; exit 3; }
fi

# Step 6: scp results back
scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/{hero,side,top}.png" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1
scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/timing.jsonl" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1
if [[ "$PROFILE" == "1" ]]; then
  scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/iter.tracy" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1 || true
  scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/zones.csv" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1 || true
fi

# Step 7: compute metrics locally
PREV_BEST="$(jq -r '[.[]?] | map(select(.action=="commit")) | min_by(.kernel_ms_median).kernel_ms_median // "Infinity"' \
              < <(cat "$REPO/docs/optimization-log/iters.jsonl" | jq -s '.') 2>/dev/null || echo "Infinity")"
python "$REPO/scripts/compute_metrics.py" \
  --iter-dir "$ITER_DIR" \
  --reference-dir "$REPO/benchmarks/reference" \
  --class "$CLASS" \
  --prev-best-ms "$PREV_BEST" >>"$ITER_DIR/run.log" 2>&1

# Step 10: rebuild report (validator + decide steps happen via separate supervisor calls)
python "$REPO/docs/optimization-log/build_report.py" >>"$ITER_DIR/run.log" 2>&1

echo "$ITER_DIR/metrics.json"
```

- [ ] **Step 2: Make it executable + smoke test (dry-run mode unavailable yet — test parses args)**

```bash
chmod +x /Users/smarton/dev/gsplat_tt/scripts/run_iter.sh
# Just verify the help/usage path works
/Users/smarton/dev/gsplat_tt/scripts/run_iter.sh 2>&1 | head -3 || true
```

Expected: `usage: ...` on stderr.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_iter.sh
git commit -m "opt-v2: run_iter.sh — end-to-end iter driver"
```

---

## Phase 8 — `health_check.sh` + `promote_to_stable.sh`

### Task 15: Write `health_check.sh`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/health_check.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Returns one of: OK | STALLED | DEVICE_HUNG | BUILD_STUCK
# Stdout is the verdict; stderr is human-readable details.
set -uo pipefail

BOX_USER="${BOX_USER:-smarton}"
BOX_HOST="${BOX_HOST:-yyzo-bh-14}"

# Check 1: tt-smi
if ! ssh -o ConnectTimeout=5 "$BOX_USER@$BOX_HOST" 'tt-smi -s' >/dev/null 2>&1; then
  echo "tt-smi failed on $BOX_HOST" >&2
  echo "DEVICE_HUNG"
  exit 0
fi

# Check 1.5: tt-triage (if available) — catches hangs/wedges that tt-smi misses.
# Gracefully skipped if the tool isn't installed. Look for any non-empty `errors`
# array in the JSON output as a wedge signal.
TRIAGE_JSON=$(ssh "$BOX_USER@$BOX_HOST" 'command -v tt-triage >/dev/null && tt-triage --json 2>/dev/null || echo ""' 2>/dev/null || echo "")
if [[ -n "$TRIAGE_JSON" ]]; then
  ERR_COUNT=$(echo "$TRIAGE_JSON" | jq -r '(.errors // []) | length' 2>/dev/null || echo "0")
  if [[ "$ERR_COUNT" != "0" ]]; then
    echo "tt-triage flagged $ERR_COUNT issue(s):" >&2
    echo "$TRIAGE_JSON" | jq -r '.errors[]?' >&2 || true
    echo "DEVICE_HUNG"
    exit 0
  fi
fi

# Check 2: viewer on 8080 responsive (tunnel must be up from Mac)
HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8080 || echo "000")
if [[ "$HTTP" != "200" && "$HTTP" != "000" ]]; then
  # 000 = no tunnel; that's a transient mac-side issue not a box issue
  echo "viewer http $HTTP" >&2
fi

# Check 3: watcher waypoint freshness
WP_AGE=$(ssh "$BOX_USER@$BOX_HOST" '
  if [[ -f /tmp/watcher_waypoints.log ]]; then
    LAST=$(stat -c %Y /tmp/watcher_waypoints.log 2>/dev/null || stat -f %m /tmp/watcher_waypoints.log)
    NOW=$(date +%s)
    echo $((NOW - LAST))
  else
    echo 0
  fi' 2>/dev/null || echo 999)
if (( WP_AGE > 30 )); then
  echo "watcher waypoints stale (${WP_AGE}s)" >&2
  echo "STALLED"
  exit 0
fi

# Check 4: any active gsplat process advancing
ACTIVE=$(ssh "$BOX_USER@$BOX_HOST" 'pgrep -af metal_example_gaussian_splatting | head -1' 2>/dev/null || echo "")
if [[ -n "$ACTIVE" ]]; then
  # process running; consider it OK (run_iter.sh tracks its own progress)
  :
fi

echo "OK"
```

- [ ] **Step 2: Make executable + smoke test**

```bash
chmod +x /Users/smarton/dev/gsplat_tt/scripts/health_check.sh
/Users/smarton/dev/gsplat_tt/scripts/health_check.sh
```

Expected: `OK` or one of the other verdicts depending on current bh-14 state.

- [ ] **Step 3: Commit**

```bash
git add scripts/health_check.sh
git commit -m "opt-v2: health_check.sh — supervisor watchdog"
```

---

### Task 16: Write `promote_to_stable.sh`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/promote_to_stable.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# ONLY path that touches bh-30. Promotes a KEEP commit to bh-30's stable_viewer.
# Per feedback-bh30-stable-only: do NOT use bh-30 for anything else.
#
# Usage: scripts/promote_to_stable.sh <commit-sha>
set -euo pipefail

SHA="${1:?usage: $0 <commit-sha>}"
BOX_USER="${BOX_USER:-smarton}"
BH14="yyzo-bh-14"
BH30="aus-misc-bh-30"  # adjust hostname if different — bh-30 is in Austin per memory
REMOTE_BIN_BH14="/proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"
STABLE_DIR_BH30="/proj_sw/user_dev/smarton/gsplat_tt_stable"

# Verify the commit exists locally and matches HEAD or recent log
git rev-parse "$SHA" >/dev/null || { echo "unknown sha $SHA" >&2; exit 1; }

# Build is assumed already done on bh-14 at this sha; verify:
ssh "$BOX_USER@$BH14" "test -f $REMOTE_BIN_BH14" || { echo "no binary on $BH14" >&2; exit 2; }

# Copy from bh-14 to bh-30 via Mac (avoid box-to-box ssh perms)
TMP="/tmp/metal_example_gaussian_splatting_$SHA"
scp "$BOX_USER@$BH14:$REMOTE_BIN_BH14" "$TMP"
scp "$TMP" "$BOX_USER@$BH30:$STABLE_DIR_BH30/metal_example_gaussian_splatting_$SHA"

# SIGTERM → 10s → restart on bh-30
ssh "$BOX_USER@$BH30" "
  pkill -TERM -f metal_example_gaussian_splatting || true
  sleep 10
  ln -sf $STABLE_DIR_BH30/metal_example_gaussian_splatting_$SHA $STABLE_DIR_BH30/metal_example_gaussian_splatting
  bash $STABLE_DIR_BH30/start_stable_viewer.sh
"

echo "promoted $SHA to $BH30"
```

- [ ] **Step 2: Make executable + commit (do NOT smoke-test against bh-30 unless explicit)**

```bash
chmod +x /Users/smarton/dev/gsplat_tt/scripts/promote_to_stable.sh
git add scripts/promote_to_stable.sh
git commit -m "opt-v2: promote_to_stable.sh — KEEP→bh-30 stable_viewer (manual or supervisor-promoted)"
```

---

## Phase 9 — Supervisor scaffold + worker prompt

### Task 17: Write `prompts/worker.md`

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/prompts/worker.md`

- [ ] **Step 1: Write the worker prompt template**

```markdown
# Worker Subagent — gsplat_tt iter execution

You are the worker for one iteration. The supervisor has chosen one hypothesis. Your job is to edit the kernel/host code as described, devsync, build, render, return artifacts.

**You do NOT commit.** The supervisor's decide_and_log.py commits if the validator returns KEEP and the iter is faster than the current best.

**You do NOT touch bh-30** under any circumstances. All builds + renders happen on yyzo-bh-14.

## Inputs (supervisor will fill in per dispatch)

- iter_num: {iter_num}
- slug: {slug}
- class: {class}                   (kernel-algebra | precompute | dispatch | binning | sort | host-prep)
- hypothesis: {hypothesis_paragraph}
- expected_ms_range: {expected_ms_range}
- files_to_edit: {list of paths}
- rollback_plan: {paragraph}

## Workflow

1. Read `docs/superpowers/specs/2026-05-25-gsplat-tt-1ms-design.md` §2 (kernel architecture) and §3 (host architecture) to understand the constraints.
2. Make the code edits per the hypothesis. Stay within the files_to_edit list.
3. Verify devsync has mirrored your edits: `devsync is-finished yyzo-bh-14`.
4. Run `scripts/run_iter.sh {iter_num} {slug} {class}`. This builds + renders + computes metrics.
5. Return the path to `metrics.json` plus a manifest of files you edited.

## Output (final message back to supervisor)

```json
{
  "iter_name": "iter-{NNN}-{slug}",
  "metrics_json_path": "...",
  "edited_files": ["path/a.cpp", "path/b.hpp", ...],
  "build_log": "...",
  "summary": "one paragraph describing what you tried and what happened"
}
```

## Failure modes

- BUILD_FAIL sentinel: report the build error from `<iter_dir>/build.log` verbatim.
- DEVICE_FAIL sentinel: report the device error from `<iter_dir>/run.log` and ask the supervisor to decide whether to retry.
- Working tree not clean (exit 10): something's wrong — abort and report.

## Anti-drift rules

- Do not edit `benchmarks/reference/*.png` (write-protected by a hook anyway).
- Do not commit. The supervisor commits.
- Do not edit files outside `files_to_edit` unless the hypothesis explicitly requires it.
- Do not run on bh-30. Use only bh-14.
- If you need to wipe the JIT cache, `run_iter.sh` will do it automatically when it detects a kernel cpp/hpp change.
```

- [ ] **Step 2: Commit**

```bash
git add prompts/worker.md
git commit -m "opt-v2: worker subagent prompt template"
```

---

### Task 18: Write `supervisor.py` (skeleton)

**Files:**
- Create: `/Users/smarton/dev/gsplat_tt/scripts/supervisor.py`

This is the outer loop. The first version is a skeleton that:
1. Defines the priority queue.
2. For each item: builds the worker prompt, prints it (manual dispatch in v1), waits for the engineer to feed artifacts back, then runs dispatch_validator + decide_and_log.

This skeleton intentionally doesn't fully automate subagent dispatch — the supervisor *Claude* invokes worker/validator subagents via its own Agent tool, calling supervisor.py only for the priority queue, dispatch_validator.sh hand-off, and decide_and_log.py invocation. The .py file holds the queue + sequencing logic.

- [ ] **Step 1: Write the skeleton**

```python
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
```

- [ ] **Step 2: Smoke test**

```bash
cd /Users/smarton/dev/gsplat_tt
python scripts/supervisor.py status
python scripts/supervisor.py queue-status
python scripts/supervisor.py next-iter | head -20
```

Expected: status shows empty completed, no stalled, next iter = iter-1 dst-resident-state.

- [ ] **Step 3: Commit**

```bash
git add scripts/supervisor.py
git commit -m "opt-v2: supervisor.py — priority queue + per-iter sequencer (skeleton)"
```

---

## Phase 10 — iter-0 baseline + frozen references

### Task 19: Run iter-0 to capture frozen references

**Files:**
- Create: `benchmarks/reference/stitch_side.png`, `benchmarks/reference/stitch_top.png` (1024×1024 implicit; agent operates at 1024 only)
- Create: `docs/optimization-log/screenshots/iter-000-baseline/{hero,side,top}.png`
- Create: `docs/optimization-log/screenshots/iter-000-baseline/timing.jsonl`
- Create: `docs/optimization-log/screenshots/iter-000-baseline/metrics.json` (computed via compute_metrics, but with reference == self → PSNR=inf)
- Create: `docs/optimization-log/iter-000-baseline.md`

- [ ] **Step 1: Verify the build is current on bh-14 main-kernel state**

```bash
cd /Users/smarton/dev/gsplat_tt
git rev-parse HEAD
# Verify no kernel modifications vs main
git diff origin/main -- backends/tt/tt-metal/programming_examples/gaussian_splatting/ | head -20
```

Expected: empty diff (we're on opt-v2 from main with only the infra carry-over).

- [ ] **Step 2: Render the three references on bh-14**

```bash
devsync is-finished yyzo-bh-14
ssh smarton@yyzo-bh-14 "cd /proj_sw/user_dev/smarton/gsplat_tt && sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting" 2>&1 | tail -10
ssh smarton@yyzo-bh-14 "
  cd /proj_sw/user_dev/smarton/gsplat_tt
  MESH_DEVICE=P100 python scripts/render_fixed.py --cycles --backend tt --scene stitch --size 1024 \
    --warmup-cycles 1 --measure-cycles 10 --out-dir /tmp/iter-000-baseline
"
```

- [ ] **Step 3: Copy renders back, save as both references and iter-0 artifacts**

```bash
mkdir -p docs/optimization-log/screenshots/iter-000-baseline
scp smarton@yyzo-bh-14:/tmp/iter-000-baseline/{hero,side,top}.png docs/optimization-log/screenshots/iter-000-baseline/
scp smarton@yyzo-bh-14:/tmp/iter-000-baseline/timing.jsonl docs/optimization-log/screenshots/iter-000-baseline/

# Promote to frozen references
cp docs/optimization-log/screenshots/iter-000-baseline/hero.png benchmarks/reference/stitch_hero.png
cp docs/optimization-log/screenshots/iter-000-baseline/side.png benchmarks/reference/stitch_side.png
cp docs/optimization-log/screenshots/iter-000-baseline/top.png  benchmarks/reference/stitch_top.png
```

- [ ] **Step 4: Compute iter-0 metrics (PSNR vs self = inf)**

```bash
python scripts/compute_metrics.py \
  --iter-dir docs/optimization-log/screenshots/iter-000-baseline \
  --reference-dir benchmarks/reference \
  --class baseline \
  --prev-best-ms 1e9
cat docs/optimization-log/screenshots/iter-000-baseline/metrics.json
```

Expected: PSNR values are `Infinity` (or a very large number — fp64 has limits but skimage handles MSE=0 → inf).

- [ ] **Step 5: Write `iter-000-baseline.md`**

```bash
cat > docs/optimization-log/iter-000-baseline.md <<'EOF'
# iter-000 — BASELINE

**Date:** 2026-05-25
**Branch:** smarton/opt-v2
**Commit:** (this commit)
**Class:** baseline (not part of optimization queue)
**Hypothesis:** N/A — capture frozen reference for the rest of the run.

## Setup

- Hardware: yyzo-bh-14, MESH_DEVICE=P100 (single chip P300)
- Scene: stitch_doll.ply
- Resolution: 1024×1024
- Kernel: unmodified main-branch alpha_blend_compute.cpp
- Measurement: 1 warmup cycle + 10 measured cycles of [hero, side, top]

## Results

(filled in from metrics.json after run)

## What this commit produces

- `benchmarks/reference/stitch_{hero,side,top}.png` — the frozen reference for all subsequent iters (1024×1024 implicit)
- `docs/optimization-log/screenshots/iter-000-baseline/` — captured renders + timing

## Frozen reference rule

`benchmarks/reference/*.png` is write-protected by a PreToolUse hook in
`.claude/settings.local.json`. No agent (supervisor, worker, validator) modifies
these files. The user explicitly promotes a new reference only if they choose to,
outside the autonomous loop.
EOF
```

Then edit it to inline the actual numbers from metrics.json:

```bash
python -c "
import json
m = json.loads(open('docs/optimization-log/screenshots/iter-000-baseline/metrics.json').read())
print(f\"kernel ms median: {m['kernel_ms_median']:.2f}\")
print(f\"kernel ms p99: {m['kernel_ms_p99']:.2f}\")
print(f\"per-view: hero {m['per_view_median']['hero']:.2f} / side {m['per_view_median']['side']:.2f} / top {m['per_view_median']['top']:.2f}\")
"
```

Insert the printed lines under the `## Results` heading in `iter-000-baseline.md`.

- [ ] **Step 6: Append iter-0 row to iters.jsonl**

```bash
python -c "
import json
from datetime import datetime, timezone
m = json.loads(open('docs/optimization-log/screenshots/iter-000-baseline/metrics.json').read())
row = {
  'iter_dir': 'iter-000-baseline',
  'timestamp': datetime.now(timezone.utc).isoformat(),
  'verdict': 'KEEP',
  'action': 'commit',
  'kernel_ms_median': m['kernel_ms_median'],
  'kernel_ms_p99': m['kernel_ms_p99'],
  'per_view_median': m['per_view_median'],
  'psnr_per_view': m['psnr_per_view'],
  'class': 'baseline',
  'commit_sha': None,
  'high_promotion_priority': False,
}
import os, tempfile
with tempfile.NamedTemporaryFile('w', delete=False, dir='docs/optimization-log/', prefix='.iters-') as t:
  t.write(json.dumps(row) + '\n')
  os.replace(t.name, 'docs/optimization-log/iters.jsonl')
"
cat docs/optimization-log/iters.jsonl
```

- [ ] **Step 7: Rebuild report**

```bash
python docs/optimization-log/build_report.py
ls docs/optimization-log/REPORT.html docs/optimization-log/graphs/
```

Expected: REPORT.html updated; 4 graph PNGs in graphs/.

- [ ] **Step 8: Commit iter-0**

```bash
git add benchmarks/reference/stitch_hero.png benchmarks/reference/stitch_side.png benchmarks/reference/stitch_top.png \
        docs/optimization-log/screenshots/iter-000-baseline/ \
        docs/optimization-log/iter-000-baseline.md \
        docs/optimization-log/iters.jsonl \
        docs/optimization-log/REPORT.html \
        docs/optimization-log/graphs/

git commit -m "$(cat <<'EOF'
iter 0 BASELINE: main-kernel render at 1024², frozen reference

Captures the frozen reference for the rest of the opt-v2 run.

Setup:
- yyzo-bh-14, MESH_DEVICE=P100 (single chip P300)
- stitch_doll.ply at 1024×1024
- Unmodified main-branch alpha_blend_compute.cpp
- Training-pattern measurement: 1 warmup cycle + 10 measured cycles of [hero, side, top]

benchmarks/reference/*.png is now write-protected via .claude/settings.local.json
PreToolUse hook. Subsequent iters compare against these images for PSNR + diff×10.
EOF
)"
git log --oneline -8
```

---

## Phase 11 — Loop integration test + kickoff

### Task 20: End-to-end dry-run with a no-op iter-1

Sanity check: run the full pipeline against an iter that's identical to iter-0. PSNR should be ∞; verdict KEEP; action no_commit_valid_but_not_faster (because ms ≈ prev_best, no improvement).

**Files:**
- Create temporarily: `docs/optimization-log/screenshots/iter-001-noop/` (will be cleaned up if validator says KEEP-not-faster, or kept if anything surprises)

- [ ] **Step 1: Copy iter-0 artifacts as a stand-in for iter-1 outputs**

```bash
cd /Users/smarton/dev/gsplat_tt
mkdir -p docs/optimization-log/screenshots/iter-001-noop
cp docs/optimization-log/screenshots/iter-000-baseline/{hero,side,top}.png docs/optimization-log/screenshots/iter-001-noop/
cp docs/optimization-log/screenshots/iter-000-baseline/timing.jsonl docs/optimization-log/screenshots/iter-001-noop/
```

- [ ] **Step 2: Run compute_metrics**

```bash
python scripts/compute_metrics.py \
  --iter-dir docs/optimization-log/screenshots/iter-001-noop \
  --reference-dir benchmarks/reference \
  --class kernel-algebra \
  --prev-best-ms "$(jq -s 'map(select(.action=="commit")) | min_by(.kernel_ms_median).kernel_ms_median' docs/optimization-log/iters.jsonl)"
cat docs/optimization-log/screenshots/iter-001-noop/metrics.json
```

Expected: PSNR=inf, kernel_ms ≈ baseline.

- [ ] **Step 3: Hand-write a passing validator.json for the dry-run** (skips actual subagent dispatch for this test)

```bash
cat > docs/optimization-log/screenshots/iter-001-noop/validator.json <<'EOF'
{
  "verdict": "KEEP",
  "visual_checks": [
    {"name": "tile_grid_seams", "result": "pass", "evidence": "diff10 uniform"},
    {"name": "tile_uniform_fill", "result": "pass", "evidence": ""},
    {"name": "missing_splat_holes", "result": "pass", "evidence": ""},
    {"name": "color_clipping_bands", "result": "pass", "evidence": ""},
    {"name": "ringing_halos", "result": "pass", "evidence": ""},
    {"name": "nan_inf_signatures", "result": "pass", "evidence": ""},
    {"name": "geometry_shift", "result": "pass", "evidence": ""},
    {"name": "diff10_structure", "result": "pass", "evidence": "uniform speckle"}
  ],
  "psnr_check": {"floor": 100.0, "actual": {"hero": 9999.0, "side": 9999.0, "top": 9999.0}, "pass": true},
  "per_view_consistency": {"max_psnr_delta_db": 0.0, "max_ms_ratio": 1.0, "pass": true},
  "timing": {"median_ms": 106.0, "p99_ms": 110.0, "vs_prev_best_pct": 0.0, "pass": true},
  "reasoning": "Dry-run no-op iter; renders are byte-identical copies of the frozen reference."
}
EOF
```

- [ ] **Step 4: Run decide_and_log in dry-run mode**

```bash
python scripts/decide_and_log.py \
  --iter-dir docs/optimization-log/screenshots/iter-001-noop \
  --state-dir docs/optimization-log \
  --dry-run
cat docs/optimization-log/screenshots/iter-001-noop/decision.json
```

Expected: `action: no_commit_valid_but_not_faster` (because median ms ≈ prev best, not faster by 2%).

- [ ] **Step 5: Verify iters.jsonl appended cleanly and report regenerated**

```bash
wc -l docs/optimization-log/iters.jsonl  # should be 2 lines: iter-0 + iter-001-noop
python docs/optimization-log/build_report.py
ls docs/optimization-log/REPORT.html docs/optimization-log/graphs/
```

- [ ] **Step 6: Clean up the noop artifacts (do NOT commit them)**

```bash
rm -rf docs/optimization-log/screenshots/iter-001-noop/
# Remove the noop row from iters.jsonl (keep only iter-0)
head -1 docs/optimization-log/iters.jsonl > /tmp/iters.jsonl
mv /tmp/iters.jsonl docs/optimization-log/iters.jsonl
# Regenerate report
python docs/optimization-log/build_report.py
git status --short
```

Expected: clean working tree (the noop dry-run produced no committed changes).

- [ ] **Step 7: No commit. This task validates the pipeline; subsequent real iters drive the actual commits.**

---

### Task 21: Kick off iter-1 (Layer A — Dst-resident state)

This is the supervisor's first real dispatch. The kernel optimization itself is the worker's job — this plan stops at "the supervisor knows what to do next."

**Files:** none changed directly by this task; the worker subagent will edit kernel files in its own context.

- [ ] **Step 1: Print the iter-1 dispatch payload**

```bash
cd /Users/smarton/dev/gsplat_tt
python scripts/supervisor.py next-iter
```

Expected: JSON with `item.iter == 1`, `item.slug == "dst-resident-state"`, `item.class == "kernel-algebra"`, plus the worker prompt template.

- [ ] **Step 2: Confirm devsync is up to date and bh-14 is reachable**

```bash
devsync is-finished yyzo-bh-14
./scripts/health_check.sh
```

Expected: devsync OK; health_check returns `OK`.

- [ ] **Step 3: Document the loop kickoff in STATUS.md**

Edit `docs/optimization-log/STATUS.md` to set `Loop: running` and note iter-1 is dispatched:

```bash
python -c "
from pathlib import Path
from datetime import datetime, timezone
p = Path('docs/optimization-log/STATUS.md')
text = p.read_text()
ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
new = text.replace('Loop: not yet started', 'Loop: running (dispatched iter-1 dst-resident-state at ' + ts + ')')
new = new.replace('iter-0 baseline: not yet run', 'iter-0 baseline: complete')
p.write_text(new)
print(new)
"
```

- [ ] **Step 4: Commit the STATUS.md transition**

```bash
git add docs/optimization-log/STATUS.md
git commit -m "opt-v2: kick off autonomous loop — iter-1 dispatched (Layer A Dst-resident state)"
git log --oneline -10
```

- [ ] **Step 5: Hand off to the supervisor session**

The supervisor (this Claude, in subsequent sessions or in this one if instructed) now:
1. Reads the iter-1 item from `python scripts/supervisor.py next-iter`.
2. Dispatches a worker subagent (Sonnet 4.6) with the worker prompt + iter-1 hypothesis.
3. Worker edits `backends/tt/tt-metal/programming_examples/gaussian_splatting/alpha_blend_compute.cpp` per Section 2 Layer A of the spec, runs `scripts/run_iter.sh 1 dst-resident-state kernel-algebra`, returns metrics path.
4. Supervisor invokes `python scripts/supervisor.py validate <iter_dir>` then dispatches the validator subagent (fresh Sonnet 4.6 context) with the prepared prompt.
5. Validator writes its response; supervisor invokes `python scripts/supervisor.py decide <iter_dir>`.
6. Supervisor schedules a wakeup via `ScheduleWakeup` for health-check, then proceeds to iter-2.

The loop is live.

---

## Self-review notes

**Spec coverage check:**
- §0 verbatim instructions → captured in plan header
- §1 repo setup + iter-0 → Tasks 1-5 + 19
- §2 kernel architecture (Layer A/B/AB) → iter queue items in Task 18 (supervisor.py DEFAULT_QUEUE)
- §3 host architecture (training-pattern, binning/sort, view-invariant) → render_fixed.py extension Task 8 + queue items Tasks 4-7
- §4 supervisor/worker/validator loop → Tasks 17, 18, plus the run-time roles
- §5 validator rejection criteria → prompts/validator.md Task 9
- §6 REPORT.html structure → build_report.py rewrite Task 13
- §7 validation harness → all script tasks (run_iter.sh, dispatch_validator.sh, decide_and_log.py, health_check.sh, promote_to_stable.sh, compute_metrics.py)
- §8 stopping criteria + ESCALATIONS → STATUS.md format in Task 4 + decide_and_log.py update_status logic

**Placeholder scan:** No TBDs. Each step has concrete commands or code.

**Type consistency:**
- `iter_dir` paths use `docs/optimization-log/screenshots/iter-NNN-slug/` everywhere ✓
- `iters.jsonl` row schema is the same in decide_and_log.py and build_report.py ✓
- Validator JSON schema in prompts/validator.md matches dispatch_validator.sh schema check ✓
- Action enum: `commit | no_commit_valid_but_not_faster | backburner` consistent ✓

**Known limitations:**
- Task 2's selective cherry-pick of `__main__.py` / `rasterization.py` requires engineer judgment — the diff may include unrelated opt-stable changes.
- Task 8's `render_fixed.py` extension uses pseudo-names (`build_pipeline`, `render_one_view`, `load_camera`) — engineer reconciles with actual function names in the existing script.
- `promote_to_stable.sh` assumes bh-30's hostname is `aus-misc-bh-30`; if different, the engineer updates the `BH30` variable.

---

## Total task count: 21
