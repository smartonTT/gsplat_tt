#!/bin/bash
# t177: drive4 smoked tip (v0 file), v2 and v2 with K2 fold off on e4562c0: all three
# hung (rc=124), so the hang is not the v2 table. Suspects: the board, or the t164
# kernel edit (fold off still changed sort_bin_onelaunch.cpp). <rev> drops the t164
# fold (kernel back to smarton/tt-project-opt), so it differs from the tip only in the
# host speed table. In ONE `ttp lock p100` hold:
#   ttp lock p100 -- bash docs/mover-speed-t177/drive5.sh <rev> <tiprev>
# 2-view smoke of the real tip <tiprev> (own dir) and of <rev> (v0 file, v2 built-in),
# each arm runs even if one hangs; if all pass, the drive3 gate (3 x 30 views).
REV=${1:?rev}; TIP=${2:?tiprev}
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t166; DT=/localdev/smarton/gstt2-t177tip
F0=$D/docs/mover-speed-t177/tables/v0.txt
r() { ssh -o BatchMode=yes $H "$@"; }
declare -A rc
echo "##### sync tip $(date +%H:%M:%S)"
opt/sync_remote.sh $H $DT $TIP || { echo SYNC_FAIL; echo ALLDONE; exit 1; }
echo "##### smoke-realtip $(date +%H:%M:%S)"
r "T162_TREE=$DT VIEWS=0:2 TMO=300 bash $DT/docs/precull-default-t162/remote_job.sh 0"; rc[realtip]=$?
echo "##### smoke-realtip rc=${rc[realtip]}"
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; echo ALLDONE; exit 1; }
J="T162_TREE=$D VIEWS=0:2 TMO=300 bash $D/docs/precull-default-t162/remote_job.sh 0"
for a in "tip:GSPLAT_TT_OL_MOVER_SPEED_FILE=$F0" "v2:GSPLAT_TT_OL_MOVER_SPEED=1"; do
  n=${a%%:*}; echo "##### smoke-$n $(date +%H:%M:%S)"
  r "$J $a"; rc[$n]=$?; echo "##### smoke-$n rc=${rc[$n]}"
done
if [ "${rc[realtip]}" = 0 ] && [ "${rc[tip]}" = 0 ] && [ "${rc[v2]}" = 0 ]; then
  echo "##### gate"
  bash docs/mover-speed-t177/drive3.sh "$REV" 3 0 1
else
  echo "NO GATE (realtip=${rc[realtip]} tip=${rc[tip]} v2=${rc[v2]})"; echo ALLDONE
fi
