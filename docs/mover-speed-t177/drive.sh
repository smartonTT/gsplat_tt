#!/bin/bash
# t177 driver (Mac side), ONE `ttp lock p100` hold for the whole chain (the remote
# tree is shared, so no other task may re-sync it between steps):
#   ttp lock p100 -- bash docs/mover-speed-t177/drive.sh <rev> <table> [rounds=3] [tracy=1]
# sync + build <rev>; if tracy=1, a 30-view EMIT_PROF Tracy capture with
# GSPLAT_TT_OL_MOVER_SPEED_FILE=<table> (tag t177-<table>) and emit_cores.py on it;
# then <rounds> interleaved untraced 30-view rounds of
#   tip = the built-in kMoverSpeedP150 (the smarton/tt-project-opt default),
#   <table> = the same build with the table file.
# remote_job.sh compares every run's 30 views with md5-r82new.txt (the 46a725ab set).
set -o pipefail
REV=${1:?rev}; TB=${2:?table name, e.g. v1}; NR=${3:-3}; TRACY=${4:-1}
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t166
O=docs/mover-speed-t177/out; mkdir -p $O
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
F=$D/docs/mover-speed-t177/tables/$TB.txt
J="T162_TREE=$D bash $D/docs/precull-default-t162/remote_job.sh"
r() { ssh -o BatchMode=yes $H "$@"; }
step() {
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; echo ALLDONE; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; echo ALLDONE; exit 1; }
r "cat $D/SHA" | grep -q "$(git rev-parse $REV)" || { echo "SHA mismatch"; echo ALLDONE; exit 1; }
if [ "$TRACY" = 1 ]; then
  tag=t177-$TB
  echo "##### tracy $tag $(date +%H:%M:%S)"
  $DEVRUN --host $H --no-verify --timeout 540 --tag $tag -- \
    "env GSPLAT_TT_OL_MOVER_SPEED_FILE=$F bash $D/docs/precull-fastemit-t166/remote_tracy.sh $tag"
  echo "##### tracy rc=$?"
  r "cd $D && grep -m1 'mover speed table' opt/profiler/$tag/capture.log; python3 opt/profiler/emit_cores.py opt/profiler/$tag/dev30.csv 30 --weights" > $O/emit_cores-$tag.txt
  grep -A4 "emit window" $O/emit_cores-$tag.txt; grep "predicted" $O/emit_cores-$tag.txt
fi
step smoke "VIEWS=0:2 TMO=600 $J 0 $TB:GSPLAT_TT_OL_MOVER_SPEED_FILE=$F"
r "grep -m1 'mover speed table' $D/tmp/t162-run-t162r0-$TB.log"
for k in $(seq 1 $NR); do
  if [ $((k % 2)) = 1 ]; then o="base $TB:GSPLAT_TT_OL_MOVER_SPEED_FILE=$F"; else o="$TB:GSPLAT_TT_OL_MOVER_SPEED_FILE=$F base"; fi
  step r$k "$J $k $o"
done
r "cd /localdev/smarton/t162_scripts && md5sum md5-t162r[1-$NR]-base.txt md5-t162r[1-$NR]-$TB.txt"
echo ALLDONE
