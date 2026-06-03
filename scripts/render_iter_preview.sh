#!/usr/bin/env bash
# Render hero.png (and reference diff10) for an amendment-002 iter so it shows
# in opt/REPORT.html ledger. Skips work if hero already exists.
# Usage: bash scripts/render_iter_preview.sh <iter_dir> [backend]
set -uo pipefail
source "$(dirname "$0")/_env.sh"

ITER_DIR="${1:?usage: render_iter_preview.sh <iter_dir> [backend]}"
BACKEND="${2:-tt}"
OUT="$OPT_DIR/metal-screenshots/$ITER_DIR"
mkdir -p "$OUT"

if [ -f "$OUT/hero.png" ] && [ -f "$OUT/timing.jsonl" ]; then
  echo "[render_iter_preview] artifacts exist: $OUT"
else
  REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
  DST="/localdev/smarton/gstt2"
  REMOTE_OUT="$DST/opt/metal-screenshots/$ITER_DIR"
  ssh "$REMOTE" "cd $DST && export TT_METAL_HOME=/localdev/smarton/tt-metal && export PYTHONPATH=$DST:$DST/backends/cpu_cpp/build-tt && mkdir -p $REMOTE_OUT && python3 scripts/render_30frame.py --backend $BACKEND --cameras benchmarks/cameras_v2.json --out-dir $REMOTE_OUT 2>&1 | tail -40" || { echo "[render_iter_preview] render failed"; exit 0; }
  rsync -av --include='*.png' --include='*.jsonl' --include='*.json' --exclude='*' "$REMOTE:$REMOTE_OUT/" "$OUT/" 2>&1 | tail -5 || true
fi

if [ -f "$OUT/hero.png" ] && [ -f "$REPO_ROOT/benchmarks/reference_v2/hero.png" ]; then
  "$LOCAL_PY" -c "
import numpy as np
from PIL import Image
ref = np.asarray(Image.open('$REPO_ROOT/benchmarks/reference_v2/hero.png').convert('RGB'), dtype=np.float64) / 255.0
c = np.asarray(Image.open('$OUT/hero.png').convert('RGB'), dtype=np.float64) / 255.0
if ref.shape == c.shape:
    amp = np.clip(np.abs(ref - c) * 10.0, 0, 1)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save('$OUT/hero_diff10.png')
    print('[render_iter_preview] diff10 ok')
" 2>&1 | tail -3
fi
echo "[render_iter_preview] done: $OUT"
