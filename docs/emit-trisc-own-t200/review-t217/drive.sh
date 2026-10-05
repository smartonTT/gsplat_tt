#!/bin/bash
# t217: sync to own remote tree, then 960px (tiles_x=30) runs with OL_EMIT_TOWN=1, =0, =1, =0.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t217; P=docs/emit-trisc-own-t200/review-t217
ttp lock p100 -- bash -c "
  opt/sync_remote.sh $H $T ${1:-HEAD} || exit 1
  for t in on1:1 off1:0 on2:1 off2:0; do
    $DEVRUN --host $H --no-verify --timeout 400 --tag t217-\${t%%:*} -- \"bash $T/$P/remote_run.sh \${t%%:*} GSPLAT_TT_OL_EMIT_TOWN=\${t##*:}\" || exit 5
  done
"
rc=$?
for a in on1 off1 on2 off2; do scp -q -o BatchMode=yes "$H:$T/tmp/t217/run-$a.log" $P/out/run-$a.log 2>/dev/null; done
echo "CHAIN_RC=$rc"; exit $rc
