#!/bin/bash
# t373 (copied from t365): bh-30 (p150b) under the existing viewer reservation (no IRD reserve/extend/release).
#   drive_bh30.sh sync    files changed since the tree's SHA, extracted with fresh mtimes into
#                         $T (#358's tree, reused), then build render/build-tt + render/bench/build
#                         under nice -n 19 ionice -c3 with half the cores (viewer keeps running).
#   [ARMS="base pin"] [OUT=out-p150] drive_bh30.sh bench <rounds>
#                         stop the viewer, bench_bh30.sh detached on the box, ALWAYS restart the
#                         viewer right away, log UTC stop/start, check localhost:8091, fetch outputs.
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; T=$P/tree358; D=docs/pinned-out; VDIR=/localdev/smarton/viewer
O=$P/out373; LOUT=${OUT:-out-p150}; TTMH=$VDIR/tt-metal
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" true || { echo "ssh bh-30 failed: viewer untouched"; exit 3; }
if [ "${1:-}" = sync ]; then
  base=$("${SSH[@]}" "cat $T/SHA") || exit 3
  echo "=== sync $base -> $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
  git diff --name-only --diff-filter=d "$base" HEAD | tar -cf - -T - | "${SSH[@]}" "cd $T && tar -xmf - && echo $(git rev-parse HEAD) > SHA" || exit 3
  "${SSH[@]}" "cd $T && export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole && J=\$(( \$(nproc) / 2 )) && mkdir -p tmp &&
    CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) &&
    { nice -n 19 ionice -c3 cmake --build render/build-tt -j \$J > tmp/build373.log 2>&1 || { tail -30 tmp/build373.log; exit 1; }; } &&
    { cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX > tmp/mbcfg.log 2>&1 &&
      nice -n 19 ionice -c3 cmake --build render/bench/build -j \$J > tmp/mbbuild.log 2>&1 || { tail -30 tmp/mbcfg.log tmp/mbbuild.log; exit 1; }; } &&
    tail -1 tmp/build373.log && ls -la render/render_clean*.so render/bench/build/risc_microbench | cut -c25-"
  rc=$?; echo "=== sync rc=$rc $(date -u +%FT%TZ)"; exit $rc
fi
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench_bh30.sh bh-30:$P/bench373.sh || exit 3
"${SSH[@]}" "rm -rf $O"
restart() {
  echo "=== viewer start $(date -u +%FT%TZ)"
  for i in 1 2 3; do VIEWER_HOST=bh-30 opt/viewer/viewer.sh start && break; sleep 10; done
  for _ in $(seq 60); do "${SSH[@]}" "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2"
  echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=bh-30 opt/viewer/viewer.sh stop
trap restart EXIT
"${SSH[@]}" "ARMS='${ARMS:-base pin}' MB=${MB:-} TRACY=${TRACY:-} T=$T TTMH=$TTMH CACHE=$P/cache O=$O setsid nohup bash $P/bench373.sh ${2:-3} > $P/bench373.log 2>&1 < /dev/null &"
for _ in $(seq 400); do "${SSH[@]}" "test -e $O/bench.rc" && break; sleep 5; done
"${SSH[@]}" "cat $P/bench373.log"
restart; trap - EXIT
mkdir -p $D/$LOUT
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "bh-30:$O/*" $D/$LOUT/ 2>/dev/null
rm -rf $D/$LOUT/prof-*
echo "=== drive end $(date -u +%FT%TZ)"
