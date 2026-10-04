#!/bin/bash
# t177 combined gate (Mac side), ONE `ttp lock p100` hold for the whole chain:
#   ttp lock p100 -- bash docs/mover-speed-t177/drive2.sh <rev> [rounds=3] [tracy=1] [first=1]
# (first > 1 resumes at that round and skips the smoke run, to split the gate over two locks)
# <rev> carries the v2 table built in and the t164 issue_brec fold (default on).
# sync + build; unit test (also against the old #174 header, which must fail the
# key check); optional EMIT_PROF Tracy of the candidate; then <rounds> rotated
# untraced 30-view rounds of three arms on the same build:
#   tip  = fold off + the #174 table (tables/v0.txt): smarton/tt-project-opt behaviour
#   v2   = fold off, built-in v2 table
#   cand = fold on, built-in v2 table (the default of <rev>)
# remote_job.sh compares every run's 30 views with md5-r82new.txt (the 46a725ab set).
set -o pipefail
REV=${1:?rev}; NR=${2:-3}; TRACY=${3:-1}; K0=${4:-1}
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t166
O=docs/mover-speed-t177/out; mkdir -p $O
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
F0=$D/docs/mover-speed-t177/tables/v0.txt
J="T162_TREE=$D bash $D/docs/precull-default-t162/remote_job.sh"
A_TIP="tip:GSPLAT_TT_OL_EMIT_FOLD=0,GSPLAT_TT_OL_MOVER_SPEED_FILE=$F0"
A_V2="v2:GSPLAT_TT_OL_EMIT_FOLD=0"
A_CAND="cand:GSPLAT_TT_OL_EMIT_FOLD=1"
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
echo "##### unit $(date +%H:%M:%S)"
r "cd $D && c++ -O2 -std=c++17 -Irender/kernels/dataflow -Irender/host -Isrc tests/unit/test_sort_onelaunch_v2.cpp -o tmp/t177_ut && tmp/t177_ut | tail -2; echo unit_rc=\$?
   mkdir -p tmp/t177_old && awk -v f=docs/mover-speed-t177/tables/v0.txt '/^inline constexpr MoverSpeed kMoverSpeedP150/ {print; while ((getline l < f) > 0) print l; s=1; next} s && /^};/ {s=0} !s' render/host/sort_mover_speed.h > tmp/t177_old/sort_mover_speed.h
   [ -s tmp/t177_old/sort_mover_speed.h ] && c++ -O2 -std=c++17 -Itmp/t177_old -Irender/kernels/dataflow -Irender/host -Isrc tests/unit/test_sort_onelaunch_v2.cpp -o tmp/t177_ut_old && { tmp/t177_ut_old | grep -c 'FAIL.*translated worker'; echo old_rc=\$?; }"
if [ "$TRACY" = 1 ]; then
  tag=t177-cand
  echo "##### tracy $tag $(date +%H:%M:%S)"
  $DEVRUN --host $H --no-verify --timeout 540 --tag $tag -- \
    "env GSPLAT_TT_OL_EMIT_FOLD=1 bash $D/docs/precull-fastemit-t166/remote_tracy.sh $tag"
  echo "##### tracy rc=$?"
  r "cd $D && python3 opt/profiler/emit_cores.py opt/profiler/$tag/dev30.csv 30 --weights" > $O/emit_cores-$tag.txt
  grep -A4 "emit window" $O/emit_cores-$tag.txt; grep "predicted" $O/emit_cores-$tag.txt
fi
[ "$K0" = 1 ] && step smoke "VIEWS=0:2 TMO=600 $J 0 $A_CAND"
for k in $(seq $K0 $NR); do
  case $((k % 3)) in
    1) o="$A_TIP $A_V2 $A_CAND" ;;
    2) o="$A_CAND $A_TIP $A_V2" ;;
    0) o="$A_V2 $A_CAND $A_TIP" ;;
  esac
  step r$k "$J $k $o"
done
r "cd /localdev/smarton/t162_scripts && md5sum md5-t162r[1-$NR]-tip.txt md5-t162r[1-$NR]-v2.txt md5-t162r[1-$NR]-cand.txt"
echo ALLDONE
