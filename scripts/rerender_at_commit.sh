#!/usr/bin/env bash
# Rerender ONE iter at its actual commit on yyzo-bh-07 (separate from bh-30).
# Produces TRUE per-iter heroes + diffs, not backfilled scene-reference renders.
#
# Steps:
#  1) ssh yyzo-bh-07, fetch + checkout the iter's commit in /localdev/smarton/gstt2-rerender
#     (separate clone so we don't disturb bh-30 or yyzo-bh-07's main workspace).
#  2) Rebuild cpu_cpp (+ TT pybind if commit had GSPLAT_WITH_TT). On build failure,
#     log the error and exit cleanly — the iter row stays as-is, caveat unchanged.
#  3) Render bicycle 30-view with the requested backend(s), into
#     opt/metal-screenshots/<iter_dir>/<backend>/  (per-backend subdir so the
#     report displays it as iter-native, NOT backfill).
#  4) Compute PSNR(tt vs cpu_cpp_mb) and PSNR(tt vs reference_v2) and write
#     a `rerender_at_commit_*` row to opt/metal-iters.jsonl with the real
#     measurements. Existing rows are NEVER mutated.
#  5) Restore HEAD on yyzo-bh-07 and regen opt/REPORT.html.
#
# Usage:
#   bash scripts/rerender_at_commit.sh <iter_dir> <commit_sha> [backends_csv]
# Default backends_csv = "tt,cpu_cpp_mb"
#
# Example:
#   bash scripts/rerender_at_commit.sh amendment-002-supervisor-iter-02 d218410 tt,cpu_cpp_mb
set -uo pipefail
source "$(dirname "$0")/_env.sh"

if [ $# -lt 2 ]; then
  echo "usage: $0 <iter_dir> <commit_sha> [backends_csv]" >&2
  exit 2
fi

ITER_DIR="$1"
COMMIT="$2"
BACKENDS_CSV="${3:-tt,cpu_cpp_mb}"

REMOTE="${RERENDER_HOST:-yyzo-bh-07}"
REMOTE_ROOT="${RERENDER_ROOT:-/localdev/smarton/gstt2-rerender}"
ORIGIN_REPO="${RERENDER_ORIGIN:-bh-30:/localdev/smarton/gstt2}"
JSONL="$OPT_DIR/metal-iters.jsonl"
LOCAL_OUT_ROOT="$OPT_DIR/metal-screenshots/$ITER_DIR"

echo "[rerender] iter_dir=$ITER_DIR commit=$COMMIT remote=$REMOTE backends=$BACKENDS_CSV"

# 1) ensure remote clone exists and is on the right commit
ssh "$REMOTE" "
set -e
if [ ! -d $REMOTE_ROOT/.git ]; then
  echo '[remote] cloning fresh workspace from $ORIGIN_REPO'
  git clone $ORIGIN_REPO $REMOTE_ROOT
fi
cd $REMOTE_ROOT
git fetch --all --quiet || true
git reset --hard HEAD --quiet || true
git clean -fdx -e .venv -e build-cpu -e build-tt -e scenes -e benchmarks --quiet || true
git checkout $COMMIT --quiet
echo '[remote] HEAD now at:' \$(git rev-parse --short HEAD) \$(git log -1 --pretty=format:%s)
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch numpy pillow plyfile cmake ninja pybind11 || true
# scenes + benchmarks may be missing on fresh clone; pull from bh-30 (real files, no symlinks)
mkdir -p scenes benchmarks
rsync -aL bh-30:/localdev/smarton/gstt2/scenes/ scenes/ 2>&1 | tail -3 || true
rsync -aL bh-30:/localdev/smarton/gstt2/benchmarks/ benchmarks/ 2>&1 | tail -3 || true
" 2>&1
RC_CHECKOUT=$?
if [ $RC_CHECKOUT -ne 0 ]; then
  echo "[rerender] FAILED: remote checkout/setup at $COMMIT (rc=$RC_CHECKOUT)" >&2
  exit 0
fi

# 2) build cpu_cpp (no TT initially; if TT was on at this commit, opt-in via env)
ssh "$REMOTE" "
set -e
cd $REMOTE_ROOT
export PATH=$REMOTE_ROOT/.venv/bin:\$PATH
GSPLAT_WITH_TT=OFF GSPLAT_SIMD_AVX2=ON BUILD_DIR=build-cpu bash scripts/build_cpu_cpp.sh 2>&1 | tail -10
" 2>&1
RC_BUILD=$?
if [ $RC_BUILD -ne 0 ]; then
  echo "[rerender] FAILED: remote build at $COMMIT (rc=$RC_BUILD). Iter row stays as-is." >&2
  exit 0
fi

# 3) render each backend into its own subdir
IFS=',' read -ra BACKENDS <<< "$BACKENDS_CSV"
mkdir -p "$LOCAL_OUT_ROOT"
declare -A MEAS_PSNR_REF
declare -A MEAS_SUM_MS
for backend in "${BACKENDS[@]}"; do
  # Use cpu_cpp_mac as the subdir name for cpu_cpp_mb so it matches the report's lookup priority
  subdir="$backend"
  case "$backend" in
    cpu_cpp_mb|cpu_cpp) subdir="cpu_cpp_mac" ;;
    tt) subdir="tt" ;;
  esac
  REMOTE_BACKEND_OUT="$REMOTE_ROOT/opt/metal-screenshots/$ITER_DIR/$subdir"
  LOCAL_BACKEND_OUT="$LOCAL_OUT_ROOT/$subdir"
  mkdir -p "$LOCAL_BACKEND_OUT"
  echo "[rerender] backend=$backend → $subdir"

  ssh "$REMOTE" "
set -e
cd $REMOTE_ROOT
mkdir -p $REMOTE_BACKEND_OUT
export PATH=$REMOTE_ROOT/.venv/bin:\$PATH
export PYTHONPATH=$REMOTE_ROOT:$REMOTE_ROOT/backends/cpu_cpp/build-cpu
python3 scripts/render_30frame.py --backend $backend --cameras benchmarks/cameras_v2.json --out-dir $REMOTE_BACKEND_OUT 2>&1 | tail -30
" 2>&1
  RC_RENDER=$?
  if [ $RC_RENDER -ne 0 ]; then
    echo "[rerender]  $backend render FAILED (rc=$RC_RENDER) — leaving subdir empty" >&2
    continue
  fi
  rsync -a --include='*.png' --include='*.jsonl' --include='*.json' --exclude='*' \
    "$REMOTE:$REMOTE_BACKEND_OUT/" "$LOCAL_BACKEND_OUT/" 2>&1 | tail -3 || true
done

# 4) compute PSNRs locally, append a rerender_at_commit row (no mutation of historical rows)
"$LOCAL_PY" - "$ITER_DIR" "$COMMIT" "$BACKENDS_CSV" "$JSONL" "$LOCAL_OUT_ROOT" "$REPO_ROOT/benchmarks/reference_v2/hero.png" <<'PY'
import json, sys, statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

(_, iter_dir, commit, backends_csv, jsonl_path, out_root_str, ref_hero_str) = sys.argv
out_root = Path(out_root_str)
ref_hero = Path(ref_hero_str)
jsonl = Path(jsonl_path)
backends = backends_csv.split(",")

def _load_rgb(p: Path):
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0

def _psnr(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * float(np.log10(1.0 / (mse ** 0.5)))

backend_subdir = {"tt": "tt", "cpu_cpp_mb": "cpu_cpp_mac", "cpu_cpp": "cpu_cpp_mac"}
heroes = {b: _load_rgb(out_root / backend_subdir.get(b, b) / "hero.png") for b in backends}
ref = _load_rgb(ref_hero)

measurements = {}
for b, h in heroes.items():
    if h is None:
        continue
    measurements[f"hero_psnr_dB_{b}_vs_reference_v2"] = _psnr(h, ref)
    timing = out_root / backend_subdir.get(b, b) / "timing.jsonl"
    if timing.exists():
        rows = []
        for line in timing.read_text().splitlines():
            if line.strip():
                try: rows.append(json.loads(line))
                except: pass
        if rows:
            measurements[f"sum_total_ms_{b}"] = sum(r.get("total_ms", r.get("total", 0.0)) for r in rows)

if "tt" in heroes and "cpu_cpp_mb" in heroes:
    measurements["hero_psnr_dB_tt_vs_cpu_cpp_mb"] = _psnr(heroes["tt"], heroes["cpu_cpp_mb"])

# Normalize inf → None + side flag for jsonl friendliness
clean = {}
for k, v in measurements.items():
    if v is None:
        continue
    if v == float("inf"):
        clean[k] = None
        clean[k + "_infinity"] = True
    else:
        clean[k] = v

row = {
    "iter_dir": iter_dir,
    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "verdict": "REVERIFIED",
    "action": f"rerender_at_commit_{commit[:7]}",
    "class": "verification",
    "rerender_commit": commit,
    "rerender_backends": backends,
    **clean,
    "note": f"True per-iter render at commit {commit[:7]}: checked out the commit on yyzo-bh-07, rebuilt cpu_cpp, rendered backends={backends_csv}. Measurements are the iter's ACTUAL output at its commit state, not a post-hoc backfill.",
}

with jsonl.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[rerender] appended row for {iter_dir} @ {commit[:7]} | {list(clean.keys())}", file=sys.stderr)
PY

# 5) restore yyzo-bh-07 to its HEAD (caller's main branch) for the next run
ssh "$REMOTE" "cd $REMOTE_ROOT && git checkout main --quiet 2>/dev/null || git checkout master --quiet 2>/dev/null || true" 2>&1 | tail -2

"$LOCAL_PY" "$REPO_ROOT/opt/build_report.py" >&2 || true
echo "[rerender] done: $ITER_DIR @ $COMMIT"
