#!/usr/bin/env bash
# Atomically record one iter: render on bh-30, measure ms/frame + hero PSNR
# vs benchmarks/reference_v2/hero.png, capture timing.jsonl, then append ONE
# complete row to opt/metal-iters.jsonl with hero_psnr_dB, sum_total_ms,
# per_stage_median_ms, and the supplied verdict/action/note. Finally regen
# opt/REPORT.html.
#
# Usage:
#   bash scripts/log_iter.sh <iter_dir> <verdict> <action> <class> <note> [backend] [env_kv...]
#
# Example (TT device project shift):
#   bash scripts/log_iter.sh \
#     amendment-002-supervisor-iter-04-tt005-final \
#     CODE_SHIPPED_BOTH_GATES_FAIL stage1_tt005_meanscam port \
#     "device kernel shipped; PSNR/perf gates fail per measurements" \
#     tt \
#     GSPLAT_TT_DEVICE_PROJECT=1
#
# Default backend = "tt". Extra args of form KEY=VAL are exported to the remote
# render env (e.g. GSPLAT_TT_DEVICE_PROJECT=1).
set -uo pipefail
source "$(dirname "$0")/_env.sh"

if [ $# -lt 5 ]; then
  echo "usage: $0 <iter_dir> <verdict> <action> <class> <note> [backend] [KEY=VAL...]" >&2
  exit 2
fi

ITER_DIR="$1"; VERDICT="$2"; ACTION="$3"; CLASS="$4"; NOTE="$5"
BACKEND="${6:-tt}"
shift 6 || true
ENV_KV=("$@")

REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
DST="/localdev/smarton/gstt2"
REMOTE_OUT="$DST/opt/metal-screenshots/$ITER_DIR"
LOCAL_OUT="$OPT_DIR/metal-screenshots/$ITER_DIR"
JSONL="$OPT_DIR/metal-iters.jsonl"

mkdir -p "$LOCAL_OUT"

env_export=""
for kv in "${ENV_KV[@]}"; do env_export+="export $kv; "; done

echo "[log_iter] iter_dir=$ITER_DIR backend=$BACKEND env=${ENV_KV[*]:-(none)}" >&2

set +e
ssh "$REMOTE" "
set -e
mkdir -p $REMOTE_OUT
cd $DST
${env_export}
export TT_METAL_HOME=/localdev/smarton/tt-metal
# tt-metal needs TT_METAL_RUNTIME_ROOT to find its runtime resources unless
# cwd is the tt-metal repo root or the package is pip-installed. Without
# this, MetalContext::instance() fails and TtBackend.project silently falls
# back to the CPU path, which is what skewed the iter-04 measurement.
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-/localdev/smarton/tt-metal}"
export PYTHONPATH=$DST:$DST/backends/cpu_cpp/build-tt
source .venv/bin/activate 2>/dev/null || true
python3 scripts/render_30frame.py --backend $BACKEND --cameras benchmarks/cameras_v2.json --out-dir $REMOTE_OUT 2>&1 | tail -40
" 2>&1
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "[log_iter] WARNING: remote render exited with rc=$RC; logging FAILED iter row" >&2
fi

rsync -a --include='*.png' --include='*.jsonl' --include='*.json' --exclude='*' \
  "$REMOTE:$REMOTE_OUT/" "$LOCAL_OUT/" 2>&1 | tail -3 || true

"$LOCAL_PY" - "$ITER_DIR" "$VERDICT" "$ACTION" "$CLASS" "$NOTE" "$BACKEND" "$RC" "$JSONL" "$LOCAL_OUT" "$REPO_ROOT/benchmarks/reference_v2/hero.png" <<'PY'
import json, sys, statistics
from datetime import datetime, timezone
from pathlib import Path

(_, iter_dir, verdict, action, cls, note, backend, rc_str, jsonl_path, out_dir_str, ref_hero_str) = sys.argv
out_dir = Path(out_dir_str)
ref_hero = Path(ref_hero_str)
jsonl = Path(jsonl_path)
rc = int(rc_str)

timing_path = out_dir / "timing.jsonl"
sum_total_ms = None
per_stage_median_ms = {}
if timing_path.exists():
    rows = []
    for line in timing_path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if rows:
        sum_total_ms = sum(r.get("total_ms", r.get("total", 0.0)) for r in rows)
        for src, dst in zip(("project","tile_assign","sort","blend"),
                             ("project_ms","tile_assign_ms","sort_ms","blend_ms")):
            vals = [float(r[src]) for r in rows if src in r and isinstance(r[src], (int, float))]
            if vals:
                per_stage_median_ms[dst] = statistics.median(vals)

hero_psnr_dB = None
hero = out_dir / "hero.png"
if hero.exists() and ref_hero.exists():
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(ref_hero).convert("RGB"), dtype=np.float64) / 255.0
        b = np.asarray(Image.open(hero).convert("RGB"), dtype=np.float64) / 255.0
        if a.shape == b.shape:
            mse = float(np.mean((a - b) ** 2))
            hero_psnr_dB = float("inf") if mse == 0.0 else 20.0 * float(np.log10(1.0 / (mse ** 0.5)))
    except Exception as e:
        print(f"[log_iter] PSNR compute failed: {e}", file=sys.stderr)

row = {
    "iter_dir": iter_dir,
    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "verdict": verdict,
    "action": action,
    "class": cls,
    "backend": backend,
    "render_rc": rc,
    "hero_psnr_dB": hero_psnr_dB if hero_psnr_dB != float("inf") else None,
    "hero_psnr_dB_infinity": hero_psnr_dB == float("inf"),
    "sum_total_ms": sum_total_ms,
    "per_stage_median_ms": per_stage_median_ms or None,
    "note": note,
}
# Drop empty optional fields so the jsonl stays tidy
row = {k: v for k, v in row.items() if v is not None}

with jsonl.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[log_iter] appended row to {jsonl} | hero_psnr={hero_psnr_dB} sum_ms={sum_total_ms} stages={list(per_stage_median_ms.keys())}", file=sys.stderr)
PY

"$LOCAL_PY" "$REPO_ROOT/opt/build_report.py" >&2 || true
echo "[log_iter] done"
