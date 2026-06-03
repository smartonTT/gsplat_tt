#!/usr/bin/env bash
# Render hero.png + timing.jsonl for every iter row in opt/metal-iters.jsonl
# that doesn't already have a usable hero, on a SEPARATE blackhole so the
# bh-30 supervisor isn't disturbed. Rsyncs back to Mac.
#
# Usage: REMOTE=yyzo-bh-07 bash scripts/backfill_missing_shots.sh
set -uo pipefail
source "$(dirname "$0")/_env.sh"

REMOTE="${REMOTE:-yyzo-bh-07}"
REMOTE_ROOT="/localdev/smarton/gstt2-backfill"
LOG="$OPT_DIR/backfill-missing-shots.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

log "=== backfill_missing_shots on $REMOTE ==="

ensure_remote_workspace() {
  log "ensuring remote workspace at $REMOTE:$REMOTE_ROOT"
  ssh "$REMOTE" "mkdir -p $REMOTE_ROOT && [ -d $REMOTE_ROOT/.venv ] || python3 -m venv $REMOTE_ROOT/.venv" 2>&1 | tee -a "$LOG"
  log "syncing repo to $REMOTE:$REMOTE_ROOT (excluding _gsplat_cpu*.so so a Mac-side .dylib never overwrites a Linux .so)"
  rsync -a --delete \
    --exclude '.venv/' --exclude '.venv-*/' --exclude 'build-*/' \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/' \
    --exclude 'opt/screenshots/' --exclude 'opt/metal-screenshots/' \
    --exclude 'scenes/' \
    --exclude 'backends/cpu_cpp/_gsplat_cpu*.so' \
    --exclude 'backends/cpu_cpp/_gsplat_cpu*.dylib' \
    "$REPO_ROOT/" "$REMOTE:$REMOTE_ROOT/" 2>&1 | tail -3 | tee -a "$LOG"
  log "installing python deps on $REMOTE"
  ssh "$REMOTE" "cd $REMOTE_ROOT && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q torch numpy pillow plyfile cmake ninja pybind11" 2>&1 | tail -5 | tee -a "$LOG"
  log "copying scenes + benchmarks from bh-30 (DEREFERENCING symlinks so bicycle.ply is a real file, not a dangling pointer to gsplat_tt)"
  ssh "$REMOTE" "mkdir -p $REMOTE_ROOT/scenes $REMOTE_ROOT/benchmarks && rsync -aL bh-30:/localdev/smarton/gstt2/scenes/ $REMOTE_ROOT/scenes/ && rsync -aL bh-30:/localdev/smarton/gstt2/benchmarks/ $REMOTE_ROOT/benchmarks/" 2>&1 | tail -3 | tee -a "$LOG"
}

build_cpu_cpp_only() {
  log "building cpu_cpp (no TT) on $REMOTE — use scripts/build_cpu_cpp.sh which configures src/CMakeLists.txt correctly"
  ssh "$REMOTE" "cd $REMOTE_ROOT && export PATH=$REMOTE_ROOT/.venv/bin:\$PATH && GSPLAT_WITH_TT=OFF GSPLAT_SIMD_AVX2=ON BUILD_DIR=build-cpu bash scripts/build_cpu_cpp.sh 2>&1 | tail -10" 2>&1 | tee -a "$LOG"
}

list_missing_iters() {
  "$LOCAL_PY" - <<'PY'
import json, os
from pathlib import Path
OPT = Path("/Users/smarton/dev/gstt2/opt")
# Known-broken iters whose existing hero doesn't represent a valid bicycle render
# (e.g., rerender-post-cherrypick was a documented cull regression).
FORCE_REDO = {
    "rerender-post-cherrypick",
}
seen = set()
needed = []
for line in (OPT / "metal-iters.jsonl").read_text().splitlines():
    line = line.strip()
    if not line: continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    d = r.get("iter_dir") or ""
    if not d or d in seen: continue
    seen.add(d)
    if d in FORCE_REDO:
        needed.append(d)
        continue
    base = OPT / "metal-screenshots" / d
    found = (base / "hero.png").exists() or any((base / sub / "hero.png").exists() for sub in ("tt","cpu_cpp_mac","cpu","default"))
    if not found:
        needed.append(d)
print("\n".join(needed))
PY
}

render_one() {
  local iter_dir="$1"
  log "rendering hero for $iter_dir on $REMOTE (backend=cpu_cpp)"
  local remote_out="$REMOTE_ROOT/opt/metal-screenshots/$iter_dir"
  ssh "$REMOTE" "mkdir -p $remote_out && cd $REMOTE_ROOT && export PATH=$REMOTE_ROOT/.venv/bin:\$PATH && export PYTHONPATH=$REMOTE_ROOT:$REMOTE_ROOT/backends/cpu_cpp/build-cpu && .venv/bin/python3 scripts/render_30frame.py --backend cpu_cpp --cameras benchmarks/cameras_v2.json --out-dir $remote_out 2>&1 | tail -10" 2>&1 | tail -12 | tee -a "$LOG"
  local local_out="$OPT_DIR/metal-screenshots/$iter_dir"
  mkdir -p "$local_out"
  rsync -a --include='*.png' --include='*.jsonl' --include='*.json' --exclude='*' "$REMOTE:$remote_out/" "$local_out/" 2>&1 | tail -3 | tee -a "$LOG"
}

# Main
: > "$LOG"
ensure_remote_workspace
build_cpu_cpp_only

mapfile -t MISSING < <(list_missing_iters)
log "${#MISSING[@]} iters need heroes: ${MISSING[*]}"
for d in "${MISSING[@]}"; do
  [ -z "$d" ] && continue
  render_one "$d"
done

log "regenerating local REPORT.html"
"$LOCAL_PY" "$REPO_ROOT/opt/build_report.py" 2>&1 | tail -3 | tee -a "$LOG"

log "DONE. Wrote heroes for ${#MISSING[@]} iters."
