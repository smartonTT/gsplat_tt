#!/bin/bash
# t177: the drive3 smoke (v2 table, fold off) hung (rc=124) right after #181's Tracy run
# hung on the same board. Isolate before gating, in ONE `ttp lock p100` hold:
#   ttp lock p100 -- bash docs/mover-speed-t177/drive4.sh <rev>
# 2-view smoke of tip (#174 table via file), v2 (built-in) and v2 with the K2 count
# fold off (the v2 table also reaches the K2 split). Each arm runs even if one hangs.
# If tip and v2 both pass, runs the drive3 gate (3 rounds x 30 views, no Tracy).
REV=${1:?rev}
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t166
F0=$D/docs/mover-speed-t177/tables/v0.txt
J="T162_TREE=$D VIEWS=0:2 TMO=300 bash $D/docs/precull-default-t162/remote_job.sh 0"
r() { ssh -o BatchMode=yes $H "$@"; }
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; echo ALLDONE; exit 1; }
r "cat $D/SHA" | grep -q "$(git rev-parse $REV)" || { echo "SHA mismatch"; echo ALLDONE; exit 1; }
declare -A rc
for a in "tip:GSPLAT_TT_OL_MOVER_SPEED_FILE=$F0" "v2:GSPLAT_TT_OL_EMIT_FOLD=0" "v2k2off:GSPLAT_TT_K2_FOLD=0"; do
  n=${a%%:*}; echo "##### smoke-$n $(date +%H:%M:%S)"
  r "$J $a"; rc[$n]=$?; echo "##### smoke-$n rc=${rc[$n]}"
done
if [ "${rc[tip]}" = 0 ] && [ "${rc[v2]}" = 0 ]; then
  echo "##### gate"
  bash docs/mover-speed-t177/drive3.sh "$REV" 3 0 1
else
  echo "NO GATE (tip=${rc[tip]} v2=${rc[v2]} v2k2off=${rc[v2k2off]})"; echo ALLDONE
fi
