#!/usr/bin/env bash
# Render hero.png + hero_diff10.png for every opt/ttw/iters.jsonl row missing shots.
# One devrun / one device session: JIT warms once, then loops a003_verify (1 view).
#
# Usage (from gstt2 repo root on Mac):
#   bash scripts/backfill_ttw_shots.sh
#   ONLY_MISSING=1 bash scripts/backfill_ttw_shots.sh   # default: skip dirs with hero+diff
#
# After completion, syncs opt/metal-screenshots/ttw-* back and regenerates REPORT.html.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ONLY_MISSING="${ONLY_MISSING:-1}"
DEVRUN="${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}"
PY="$(command -v python3 || command -v python)"
[[ -n "$PY" ]] || { echo "python3 required" >&2; exit 1; }
[[ -x "$DEVRUN" ]] || { echo "missing devrun: $DEVRUN" >&2; exit 1; }

missing="$("$PY" - "$ROOT" "$ONLY_MISSING" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
only = sys.argv[2] == "1"
jsonl = root / "opt/ttw/iters.jsonl"
dirs = []
for line in jsonl.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    n = r.get("iter")
    d = str(r.get("iter_dir") or "").strip() or (f"ttw-{int(n):03d}" if n is not None else "")
    if not d:
        continue
    base = root / "opt/metal-screenshots" / d
    hero = base / "hero.png"
    diff = base / "hero_diff10.png"
    if only and hero.exists() and (diff.exists() or (base / "diff10x.png").exists()):
        continue
    dirs.append(d)
seen = set()
out = []
for d in dirs:
    if d not in seen:
        seen.add(d)
        out.append(d)
print(" ".join(out))
PY
)"

if [[ -z "$missing" ]]; then
  echo "[backfill] all ttw rows already have hero + diff"
  "$PY" opt/build_report.py
  "$PY" opt/validate_report.py
  exit 0
fi

echo "[backfill] will render: $missing"

if [[ "${REBUILD:-1}" == 1 ]]; then
  echo "[backfill] incremental build on device (REBUILD=0 to skip)"
  BUILD_CMD='export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100; source .venv/bin/activate 2>/dev/null; cmake --build build-tt -j 16'
  "$DEVRUN" --tag backfill-build --timeout 1800 -- "$BUILD_CMD" || exit 1
fi

VERIFY_ENV='export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 GSPLAT_TT_BLEND_MODE=2 GSPLAT_TT_MB_KERNEL=1 GSPLAT_TT_DEVICE_PROJECT=1 GSPLAT_TT_RESIDENT_PROJECT=1 GSPLAT_TT_RESIDENT_GATHER=1 GSPLAT_TT_DEVICE_TILE_ASSIGN=1 GSPLAT_TT_RESIDENT_TA_IN=1 GSPLAT_TT_DEVICE_SORT=1 GSPLAT_TT_RESIDENT_PAIRS=1 GSPLAT_TT_RESIDENT_BLEND=1 GSPLAT_TT_SORT_DEVICE_PUBLISH=1 GSPLAT_TT_TA_DEVICE_SCAN=1 GSPLAT_TT_PROJ_DEVICE_SCAN=1 GSPLAT_TT_SFPU_CULL=1 GSPLAT_TT_TILE_BUCKET=1 GSPLAT_TT_BUCKET_FIT=8192 GSPLAT_TT_FUSED_TILE=0 GSPLAT_TT_MB_TIMING=1'

REMOTE_CMD="$VERIFY_ENV; source .venv/bin/activate 2>/dev/null; set -e"
for d in $missing; do
  REMOTE_CMD="$REMOTE_CMD; echo \"[backfill] === $d ===\"; .venv/bin/python3 scripts/a003_verify.py --views 1 --iter-dir \"$d\" --out \"opt/ttw/backfill-${d}.json\" || exit 1"
done

# Do not wrap in bash -lc — devrun already runs bash -lc on the device.
"$DEVRUN" --tag backfill-ttw-shots --timeout 7200 -- "$REMOTE_CMD"
vrc=$?
if [[ $vrc -ne 0 ]]; then
  echo "[backfill] devrun failed rc=$vrc" >&2
  exit "$vrc"
fi

echo "[backfill] regenerating REPORT.html"
"$PY" opt/build_report.py
"$PY" opt/validate_report.py
echo "[backfill] done"
