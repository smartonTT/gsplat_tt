#!/bin/bash
# t291: sync + K2 TRISC split Tracy capture + untraced A/B. Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps="sync prof"]   env: PROF_ENV (extra ENV=V for prof), B_ENV (arm B envs, comma list)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t291
P=docs/k2-trisc-t291
O=$P/out; mkdir -p $O tmp/t291
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
STEPS=" ${2:-sync prof} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has prof; then
  n=${PROF_TAG:-t291-k2p}
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag $n -- "bash $T/$P/remote_tracy.sh $n GSPLAT_TT_K2_TRISC=1 ${PROF_ENV:-}"
  echo "PROF_RC=$?"
  for f in zones.txt gaps.txt k2_parts.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/$f $O/$n-$f 2>/dev/null; done
fi
if has ab; then
  B=B:GSPLAT_TT_K2_TRISC=1${B_ENV:+,$B_ENV}
  for r in 1 2 3; do
    case $r in 2) arms="A $B";; *) arms="$B A";; esac
    lk $DEVRUN --host $H --no-verify --timeout 560 --tag t291-ab$r -- "bash $T/$P/remote_ab.sh $r $arms" > $O/ab${AB_TAG:-}-r$r.log 2>&1
    echo "AB${r}_RC=$?"; grep -E "^===|SUMMARY|SWEEP_MD5|rc=" $O/ab${AB_TAG:-}-r$r.log
    [ "$(grep -c "SWEEP_MD5=46a725ab" $O/ab${AB_TAG:-}-r$r.log)" = 2 ] || { echo "not both md5-clean in r$r, stop"; break; }
  done
fi
echo CHAIN_DONE
