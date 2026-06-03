#!/usr/bin/env bash
# Push amendment-002 sources to bh-30 (scoped, fast).
set -euo pipefail
source "$(dirname "$0")/_env.sh"

REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
DST="/localdev/smarton/gstt2"

echo "[sync_to_bh30] $REPO_ROOT -> $REMOTE:$DST" >&2

rsync -av --delete \
  --exclude '.venv/' --exclude '.venv-*/' --exclude 'build-*/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git/' \
  "$REPO_ROOT/src/" "$REMOTE:$DST/src/"

rsync -av "$REPO_ROOT/cmake/" "$REMOTE:$DST/cmake/"

rsync -av \
  "$REPO_ROOT/backends/cpu_cpp/pybind_module.cpp" \
  "$REMOTE:$DST/backends/cpu_cpp/"

rsync -av \
  "$REPO_ROOT/backends/tt/backend.py" \
  "$REPO_ROOT/backends/__init__.py" \
  "$REMOTE:$DST/backends/tt/"

rsync -av \
  "$REPO_ROOT/backends/__init__.py" \
  "$REMOTE:$DST/backends/"

rsync -av \
  "$REPO_ROOT/scripts/fix_tt_metal_cmake_exports.sh" \
  "$REPO_ROOT/scripts/build_cpu_cpp.sh" \
  "$REPO_ROOT/scripts/sync_to_bh30.sh" \
  "$REPO_ROOT/scripts/amendment002_diagnose_blend.sh" \
  "$REPO_ROOT/scripts/render_iter_preview.sh" \
  "$REPO_ROOT/scripts/a003_verify.py" \
  "$REMOTE:$DST/scripts/"

rsync -av \
  "$REPO_ROOT/opt/metal-iters.jsonl" \
  "$REMOTE:$DST/opt/" 2>/dev/null || true

echo "[sync_to_bh30] ok" >&2
