#!/bin/bash
# t166 diagnosis (Mac): sync + build <rev>, then 3 emit-part Tracy captures:
# p2f (PRECULL=2 + fast emit), p1f (PRECULL=1 + fast), p2s (PRECULL=2, OL_EMIT_FAST=0).
#   drive_prof.sh <rev> [arms="p2f p1f p2s"]   (one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t166
O=docs/precull-fastemit-t166/out; mkdir -p $O
REV=${1:?rev}
ttp lock p100 -- opt/sync_remote.sh $H $T "$REV"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
for a in ${2:-p2f p1f p2s}; do
  case $a in
    p2f) E="GSPLAT_TT_PRECULL=2" ;;
    p1f) E="GSPLAT_TT_PRECULL=1" ;;
    p2s) E="GSPLAT_TT_PRECULL=2 GSPLAT_TT_OL_EMIT_FAST=0" ;;
    p1s) E="GSPLAT_TT_PRECULL=1 GSPLAT_TT_OL_EMIT_FAST=0" ;;
  esac
  tag=t166-$a${TAGSFX:-}
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 540 --tag $tag -- \
    "env $E bash $T/docs/precull-fastemit-t166/remote_tracy.sh $tag"
  echo "TRACY_${a}_RC=$?"
  mkdir -p $O/$tag
  for f in zones.txt emit_parts.txt; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$tag/$f $O/$tag/$f; done
done
echo CHAIN_DONE
