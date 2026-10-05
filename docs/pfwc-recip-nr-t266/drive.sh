#!/bin/bash
# t266: sync + build, then RECIP_NEWTON off/on arms. Mac side, one ttp lock p100 per device step.
#   drive.sh [rev] [steps]   steps: any of sync r1 dump r2 r3
#   r1: off/nr at the default floor and at 1/1024; r2, r3: default floor, order rotated.
#   dump: hero projection dumps (off, nr) for docs/floor-ab-t260/compare_proj.py.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t266
P=docs/pfwc-recip-nr-t266
O=$P/out; mkdir -p $O
NR=GSPLAT_TT_PFWC_RECIP_NEWTON=1
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
STEPS=" ${2:-sync r1 dump r2 r3} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t266/run-*.log" "$H:$T/tmp/t266/md5-*.txt" "$H:$T/tmp/t266/hero-*.png" "$H:$T/tmp/t266/proj_dev-*.npz" $O/ 2>/dev/null; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
arm() {  # round armspec
  lk $DEVRUN --host $H --no-verify --timeout 420 --tag t266-r$1-${2%%:*} -- "bash $T/$P/remote_time.sh $1 $2"
  echo "TIME_r$1_${2%%:*}_RC=$?"
}
if has r1; then for a in off:255 nr:255:$NR nr1024:1024:$NR off1024:1024; do arm 1 $a; done; fetch; fi
if has dump; then for a in dump-nr:$NR dump-off; do arm 1 $a; done; fetch; fi
if has r2; then for a in nr:255:$NR off:255; do arm 2 $a; done; fi
if has r3; then for a in off:255 nr:255:$NR; do arm 3 $a; done; fi
fetch
echo CHAIN_DONE
