#!/bin/bash
# Device screenshot of one iteration: sync <rev> to the device host, build, render the
# bicycle hero view (plus the 30-view md5 sweep) on the device, fetch the hero and write
# the 10x diff and PSNRs. One ttp lock p100 around the whole sync + build + render.
#   opt/ttw/screenshot.sh <iter> <rev> [ENV=V ...]
# Env: H (yyzo-bh-04, the measurement box since 2026-10-06), T (remote tree, /localdev/smarton/gstt2-shot), NAME (ttw-<iter>,
# the output dir under opt/metal-screenshots/), FILE (hero.png), NO_SYNC=1 (reuse tree).
# Output: opt/metal-screenshots/<NAME>/<FILE>, <FILE stem>_diff10.png (|hero - ref| * 10,
# ref = benchmarks/reference_v2/hero.png), and a SHOT line with the sweep md5 (8 chars of
# md5 of the sorted per-view md5 list; 906e0435 = the bicycle default since iter 205, PFWC_RECIP_NEWTON on; 46a725ab before), the
# PSNR vs reference_v2 (the same reference as the diff: device_screenshot.psnr_vs_ref) and
# the golden match vs tests/fixtures/hero/hero_golden_8bit.png (golden_match, a badge only).
# Still LOOK at the hero and the diff (tile seams, blocky/empty tiles) before attaching it.
set -u
cd "$(git rev-parse --show-toplevel)"
it=${1:?iter}; rev=${2:?rev}; shift 2
H=${H:-yyzo-bh-04}; T=${T:-/localdev/smarton/gstt2-shot}
NAME=${NAME:-ttw-$it}; FILE=${FILE:-hero.png}
tag=$NAME-${FILE%.png}
O=opt/metal-screenshots/$NAME; mkdir -p "$O" tmp/shot
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
sha=$(git rev-parse --short "$rev") || exit 2
if [ "${1:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/opt/ttw/screenshot.sh" "$it" "$rev" --locked "$@"
fi
shift
set -o pipefail
if [ "${NO_SYNC:-0}" != 1 ]; then
  opt/sync_remote.sh "$H" "$T" "$rev" > tmp/shot/sync-$tag.log 2>&1 || { tail -20 tmp/shot/sync-$tag.log; exit 3; }
  tail -2 tmp/shot/sync-$tag.log
fi
scp -q -o BatchMode=yes opt/ttw/screenshot_remote.sh "$H:$T/tmp/screenshot_remote.sh" || exit 3
$DEVRUN --host "$H" --no-verify --timeout ${SHOT_TIMEOUT:-420} --tag shot-$tag -- \
  "bash $T/tmp/screenshot_remote.sh $T $tag $*" 2>&1 | tee tmp/shot/run-$tag.out | grep -E "^(===|run rc|VIEWS|ALL_|  +[0-9]+ )"
rc=${PIPESTATUS[0]}
scp -q -o BatchMode=yes "$H:$T/tmp/shot/hero-$tag.png" "$O/$FILE" || { echo "SHOT_FAIL no hero (rc=$rc)"; exit 4; }
scp -q -o BatchMode=yes "$H:$T/tmp/shot/md5-$tag.txt" "$H:$T/tmp/shot/run-$tag.log" tmp/shot/
python3 - "$O/$FILE" "$O/${FILE%.png}_diff10.png" tmp/shot/md5-$tag.txt "$sha" "$*" <<'PY'
import hashlib, sys
import numpy as np
from PIL import Image
hero, diff, md5f, sha, env = sys.argv[1:6]
def rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return "inf" if m == 0 else round(float(10 * np.log10(255.0 ** 2 / m)), 2)
h = rgb(hero)
gold = rgb("tests/fixtures/hero/hero_golden_8bit.png")
ref = rgb("benchmarks/reference_v2/hero.png")
Image.fromarray(np.clip(np.abs(h - ref) * 10, 0, 255).astype(np.uint8)).save(diff)
sweep = hashlib.md5(open(md5f, "rb").read()).hexdigest()[:8]
print(f"SHOT hero={hero} diff={diff} commit={sha} env='{env or 'defaults'}' size={h.shape[1]}x{h.shape[0]} "
      f"sweep_md5={sweep} ref=benchmarks/reference_v2/hero.png psnr_vs_ref={psnr(h, ref)} "
      f"golden_match={str(bool(np.array_equal(h, gold))).lower()} max_lsb_vs_golden={int(np.abs(h - gold).max())}")
PY
