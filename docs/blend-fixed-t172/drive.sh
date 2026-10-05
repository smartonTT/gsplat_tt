#!/bin/bash
# t172 driver (Mac side). Each device step is its own `ttp lock p100` hold.
#   drive.sh <rev> [steps=sync,prof]      steps: sync, prof (step 1), time:<round>:<arms> (A/B)
# prof: GSPLAT_TT_BLEND_PROF=2 Tracy capture of views 0:2 + GSPLAT_TT_MB_TILECYC=1 DPRINT of views 0:2.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t172; O=docs/blend-fixed-t172/out; mkdir -p $O
REV=${1:-HEAD}; STEPS=${2:-sync,prof}
lk() {
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo LOCK_BUSY; echo CHAIN_DONE; exit 75; fi
  return $rc
}
IFS=, read -ra ST <<< "$STEPS"
for s in "${ST[@]}"; do
  case $s in
    sync) lk opt/sync_remote.sh $H $T "$REV"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; } ;;
    prof) lk $DEVRUN --host $H --no-verify --timeout 540 --tag t172-tracy -- "bash $T/docs/blend-fixed-t172/remote_prof.sh tracy"
          echo "TRACY_RC=$?"
          lk $DEVRUN --host $H --no-verify --timeout 560 --tag t172-tc -- "bash $T/docs/blend-fixed-t172/remote_prof.sh tc"
          echo "TC_RC=$?"
          scp -q -o BatchMode=yes "$H:$T/tmp/t172/tc.dprint" "$H:$T/tmp/t172/*.log" $O/
          scp -q -o BatchMode=yes "$H:$T/opt/profiler/t172-prof/chunks/0-2/profile_log_device.csv" $O/prof-dev.csv ;;
    time:*) r=${s#time:}; rnd=${r%%:*}; arms=${r#*:}
          lk $DEVRUN --host $H --no-verify --timeout 560 --tag t172-time$rnd -- \
            "bash $T/docs/blend-fixed-t172/remote_time.sh $rnd ${arms//+/ }"
          echo "TIME${rnd}_RC=$?"
          scp -q -o BatchMode=yes "$H:$T/tmp/t172/run-r$rnd-*.log" "$H:$T/tmp/t172/md5-r$rnd-*.txt" $O/ ;;
  esac
done
echo CHAIN_DONE
