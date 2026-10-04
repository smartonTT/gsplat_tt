#!/bin/bash
# t166 (Mac side): per-row BRISC split sweep under PRECULL=2 + fast emit, one
# `ttp lock p100` hold: sync + build <rev>, then 1 untraced round x 30 views of
#   p1f (PRECULL=1, md5 must equal md5-r82new.txt), p2f (PRECULL=2), and
#   PRECULL=2 + GSPLAT_TT_OL_SPLIT_ROWS=<spec> for each spec in $SPECS,
# then md5 of every split arm vs p2f (must be identical).
#   ttp lock p100 -- bash docs/precull-fastemit-t166/sweep.sh <rev> [round=20]
set -o pipefail
REV=${1:?rev}; RND=${2:-20}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t166
J="T162_TREE=$D bash $D/docs/precull-default-t162/remote_job.sh"
SPECS=${SPECS:-"430/460 400/440/490 450/475"}
r() { ssh -o BatchMode=yes $H "$@"; }
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; echo SWEEP_DONE; exit 1; }
r "cat $D/SHA" | grep -q "$(git rev-parse $REV)" || { echo "SHA mismatch"; echo SWEEP_DONE; exit 1; }
arms="p1f:GSPLAT_TT_PRECULL=1 p2f:GSPLAT_TT_PRECULL=2"
i=0; for s in $SPECS; do i=$((i+1)); arms="$arms s$i:GSPLAT_TT_PRECULL=2,GSPLAT_TT_OL_SPLIT_ROWS=$s"; echo "s$i = $s"; done
r "$J $RND $arms"; echo "##### round rc=$?"
i=0; for s in $SPECS; do i=$((i+1));
  r "cd /localdev/smarton/t162_scripts && cmp -s md5-t162r$RND-p2f.txt md5-t162r$RND-s$i.txt" && echo "s$i ($s) MD5_EQ_P2F" || echo "s$i ($s) MD5_DIFF"
done
echo SWEEP_DONE
