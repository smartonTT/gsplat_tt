#!/bin/bash
# t358: p100a check on the measurement box (yyzo-bh-04): sync + build + A/B under one p100 lock.
#   ttp lock p100 -- docs/p150-mover-t358/drive_p100a.sh <rev> <rounds>
set -u
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-04; T=/localdev/smarton/gstt2-t358; D=docs/p150-mover-t358; O=$T/tmp/out358
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 $H)
echo "=== sync $(date -u +%FT%TZ)"
opt/sync_remote.sh $H $T "${1:-HEAD}" > /tmp/t358-sync-p100a.log 2>&1 || { tail -20 /tmp/t358-sync-p100a.log; exit 3; }
tail -1 /tmp/t358-sync-p100a.log
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench_ab.sh $H:$T/tmp/bench_ab.sh || exit 3
"${SSH[@]}" "rm -rf $O; T=$T TTMH=/localdev/smarton/tt-metal CACHE=/localdev/smarton/.cache/ttmc-gstt2-t358 O=$O MESH=P100 bash $T/tmp/bench_ab.sh ${2:-3}"
rc=$?
mkdir -p $D/out-p100a
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$O/*" $D/out-p100a/
echo "=== drive end rc=$rc $(date -u +%FT%TZ)"
exit $rc
