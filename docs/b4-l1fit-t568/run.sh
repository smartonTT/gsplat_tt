#!/bin/bash
# t568: fuse check (30-view md5 + 1 b2b round) then the A/B (hero + 3 alternating b2b rounds) of
# GSPLAT_TT_SORT_IN_BLEND=1 vs base with the solo sort's mover-1 CBs dropped, through t555's drive.sh
# (its own ttp lock p100, bh-33 measurement reservation only, StrictHostKeyChecking=yes shims).
#   ttp detach t568 -- docs/b4-l1fit-t568/run.sh <rev>
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; D=docs/b4-l1fit-t568
PLAN=check ARM=fuse O=$D/out-check docs/b4-step3b-t555/drive.sh "$rev"; c=$?
echo "=== t568 check rc=$c"
[ $c = 0 ] || exit $c
PLAN=ab ARM=fuse O=$D/out-ab docs/b4-step3b-t555/drive.sh "$rev"; a=$?
echo "=== t568 ab rc=$a"
exit $a
