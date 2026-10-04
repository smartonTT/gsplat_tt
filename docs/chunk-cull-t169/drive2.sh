#!/bin/bash
# t169 follow-up probe: Morton reorder alone (O2 kernels, no tile list) vs off,
# to split the noskip arm's +0.87 ms project into reorder cost vs -Os reader.
#   ttp lock p100 -- bash docs/chunk-cull-t169/drive2.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t169
J="T169_TREE=$D bash $D/docs/chunk-cull-t169/remote_job.sh"
r() { ssh -o BatchMode=yes $H "$@"; }
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
r "$J 6 off:GSPLAT_TT_CHUNK_CULL=0 reord:GSPLAT_TT_CHUNK_CULL=0,GSPLAT_TT_CHUNK_REORDER=1" || exit 10
r "$J 7 reord:GSPLAT_TT_CHUNK_CULL=0,GSPLAT_TT_CHUNK_REORDER=1 off:GSPLAT_TT_CHUNK_CULL=0" || exit 11
echo ALLDONE
