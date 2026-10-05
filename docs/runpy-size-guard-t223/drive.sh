#!/bin/bash
# t223: sync, then 960 px 4-view runs with and without --no-ref, and a 1024 px default 2-view run.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t223; P=docs/runpy-size-guard-t223
C=docs/emit-trisc-own-t200/review-t217/cameras_960.json
ttp lock p100 -- bash -c "
  opt/sync_remote.sh $H $T ${1:-HEAD} || exit 1
  $DEVRUN --host $H --no-verify --timeout 580 --tag t223-n960 -- 'bash $T/$P/remote_run.sh n960 --no-ref --cameras $C --view-range 0:4'; r1=\$?
  $DEVRUN --host $H --no-verify --timeout 580 --tag t223-r960 -- 'bash $T/$P/remote_run.sh r960 --cameras $C --view-range 0:4'; r2=\$?
  $DEVRUN --host $H --no-verify --timeout 580 --tag t223-r1024 -- 'bash $T/$P/remote_run.sh r1024 --view-range 0:2'; r3=\$?
  echo RCS \$r1 \$r2 \$r3; [ \$r1\$r2\$r3 = 000 ]
"
rc=$?
for a in n960 r960 r1024; do scp -q -o BatchMode=yes "$H:$T/tmp/t223/run-$a.log" $P/out/run-$a.log 2>/dev/null; done
echo "CHAIN_RC=$rc"; exit $rc
