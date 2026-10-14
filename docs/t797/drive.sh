#!/bin/bash
# t797 (continues #794): 3DGS Jacobian clamp build, tests, device A/B, hero PSNR. ONE `ttp lock p100`:
#   ttp detach t797 -- docs/t797/drive.sh <rev>       (Mac, worktree root; <rev> committed)
# ssh preflight @p100, opt/sync_remote.sh (one-way, repo-scoped, this worktree) + render build, remote.sh build
# (gsplat_cpu_tests '[project]', _gsplat_cpu), remote.sh pytest, then base (GSPLAT_TT_JCLAMP=0) vs cand (GSPLAT_TT_JCLAMP=1):
# orbit hero with the CPU ref (golden + hero_vs_cpu), _DSC8679, warm-up, ROUNDS alternated rounds of orbit b2b
# + train-NN b2b, JIT listings for opt/ab_jit_check.py, device screenshots (opt/ttw/screenshot.sh, NAME=t797),
# pull tmp/t797, PSNR of the _DSC8679 renders vs the JPG on the Mac (images stay uncommitted).
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never bh-30,
# StrictHostKeyChecking=yes on every ssh/scp. Each ssh step < 5 min except pytest (CPU only, timeout 40 min).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} ROUNDS=${ROUNDS:-4}
D=docs/t797; O=${O:-$D/out}
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- env O="$O" ROUNDS="$ROUNDS" "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O" tmp/t797-bin
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=20 "$@"\n' $c > tmp/t797-bin/$c
  chmod +x tmp/t797-bin/$c
done
export PATH=$PWD/tmp/t797-bin:$PATH
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
sha=$(git rev-parse "$rev^{commit}")
echo "=== t797 $rev=$sha to $H:$T rounds=$ROUNDS $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
if [ $rc = 3 ]; then echo "sync rc=3 (stale-build heuristic), resync"; opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?; fi
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 ssh "$H" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
timeout 60 ssh "$H" "rm -rf $T/tmp/t797 /localdev/smarton/.cache/ttmc-t797-*; mkdir -p $T/tmp/t797"
scp -q "$D/remote.sh" "$H:$T/tmp/t797-remote.sh.tmp" && ssh "$H" "mv $T/tmp/t797-remote.sh.tmp $T/tmp/t797-remote.sh" || exit 3
rc=0
step() {  # name timeout args...
  local n=$1 to=$2; shift 2
  echo "=== step $n $(TZ=America/Los_Angeles date +%T)"
  timeout "$to" ssh "$H" "T=$T bash $T/tmp/t797-remote.sh $*" > "$O/$n.out" 2>&1; local s=$?
  grep -vE '^\s*$' "$O/$n.out" | cut -c1-300 | tail -40
  [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && { echo "=== hang/timeout in $n"; exit 124; }
  return 0
}
step build 900 build
step pytest 2700 pytest
step hero 600 run 0 hbase hcand
step dsc 400 run 0 dbase dcand
step warm 400 run 0 obase ocand
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) = 1 ]; then step r$r 500 run $r obase ocand nbase ncand; else step r$r 500 run $r ocand obase ncand nbase; fi
done
timeout 300 ssh "$H" "for a in base cand; do find /localdev/smarton/.cache/ttmc-t797-\$a -maxdepth 8 -type d -path '*/kernels/*' > $T/tmp/t797/jit-\$a.txt; done"
for a in cand base; do
  e=GSPLAT_TT_JCLAMP=1; [ $a = base ] && e=GSPLAT_TT_JCLAMP=0
  H=$H T=$T NO_SYNC=1 NAME=t797 FILE=hero-$a.png SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 797 "$rev" --locked $e > "$O/shot-$a.out" 2>&1
  s=$?; grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot-$a.out"; echo "shot $a rc=$s"; [ $s = 0 ] || rc=7
done
rm -rf "$O/remote"
opt/sync_remote.sh --pull tmp/t797 "$O/remote" "$H" "$T"; echo "pull rc=$?"
timeout 120 ssh "$H" "rm -rf $T/tmp/t797 /localdev/smarton/.cache/ttmc-t797-*; df -h /localdev/smarton | tail -1"; echo "remote cleanup rc=$?"
python3 opt/ab_jit_check.py --arm base="$O/remote/jit-base.txt" --arm cand="$O/remote/jit-cand.txt" --kernel project_pfwc_compute > "$O/jit-check.out" 2>&1
s=$?; cat "$O/jit-check.out"; echo "ab_jit_check rc=$s"; [ $s = 0 ] || rc=6
python3 - "$O/remote" <<'PY'
import sys, numpy as np
from PIL import Image
o = sys.argv[1]
gt = np.asarray(Image.open("scenes/mipnerf360/bicycle/images_4/_DSC8679.JPG").convert("RGB"), np.float64)
for a in ("base", "cand"):
    try:
        h = np.asarray(Image.open(f"{o}/hero-r0-d{a}.png").convert("RGB"), np.float64)
    except FileNotFoundError:
        print(f"DSC8679 {a} missing"); continue
    if h.shape != gt.shape:
        print(f"DSC8679 {a} shape {h.shape} != JPG {gt.shape}"); continue
    m = np.mean((h - gt) ** 2)
    print(f"DSC8679 {a} {h.shape[1]}x{h.shape[0]} psnr_vs_jpg={10 * np.log10(255.0 ** 2 / m):.3f} dB")
PY
for f in "$O"/remote/run-r[1-9]*.log; do echo "$(basename "$f") $(grep -hoE 'b2b_ms_frame_median=[0-9.]+' "$f" | tail -1)"; done
echo "=== done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
