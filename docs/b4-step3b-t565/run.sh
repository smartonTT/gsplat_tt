#!/bin/bash
# t565: device check (30-view md5 + 1 b2b round) then the A/B (hero + 3 alternating b2b rounds)
# of GSPLAT_TT_SORT_IN_BLEND=1 (ARM=fuse) vs base, through t555's drive.sh (its own ttp lock p100,
# bh-33 measurement reservation only, StrictHostKeyChecking=yes shims).
#   ttp detach t565 -- docs/b4-step3b-t565/run.sh <rev> [arm: fuse | fuse34 = fuse at GSPLAT_TT_KCFG_EXTRA_KB=34]
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; ARM=${2:-fuse}; D=docs/b4-step3b-t565
if [ "$ARM" = kcfg48 ]; then  # t565: isolate the fuse-check hang: solo sort at KCFG +48 KB alone
  PLAN=kcfg O=$D/out-kcfg48 docs/b4-step3b-t555/drive.sh "$rev"; c=$?; echo "=== t565 kcfg48 rc=$c"; exit $c
fi
PLAN=check ARM=$ARM O=$D/out-check-$ARM docs/b4-step3b-t555/drive.sh "$rev"; c=$?
echo "=== t565 check rc=$c"
[ $c = 0 ] || exit $c
PLAN=ab ARM=$ARM O=$D/out-ab-$ARM docs/b4-step3b-t555/drive.sh "$rev"; a=$?
echo "=== t565 ab rc=$a"
exit $a
