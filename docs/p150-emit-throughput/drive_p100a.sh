#!/bin/bash
# t365: p100a check on the measurement box (yyzo-bh-04): changed-file sync into #358's tree
# (fresh mtimes), build, then bench365.sh (MB=1 microbench + rounds) under ONE p100 lock.
#   ttp lock p100 -- docs/p150-emit-throughput/drive_p100a.sh <rounds>   [ARMS="P1"] [MB=1] [OUT=out-p100a]
set -u
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-04; T=/localdev/smarton/gstt2-t358; D=docs/p150-emit-throughput; O=$T/tmp/out365; TTMH=/localdev/smarton/tt-metal
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 $H)
base=$("${SSH[@]}" "cat $T/SHA") || exit 3
echo "=== sync $base -> $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
git diff --name-only --diff-filter=d "$base" HEAD | tar -cf - -T - 2>/dev/null | "${SSH[@]}" "cd $T && tar -xmf - 2>/dev/null; echo $(git rev-parse HEAD) > SHA" || exit 3
"${SSH[@]}" "cd $T && export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole && mkdir -p tmp &&
  CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) &&
  { cmake --build render/build-tt -j 10 > tmp/build365.log 2>&1 || { tail -30 tmp/build365.log; exit 1; }; } &&
  { cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX > tmp/mbcfg.log 2>&1 &&
    cmake --build render/bench/build -j 10 > tmp/mbbuild.log 2>&1 || { tail -30 tmp/mbcfg.log tmp/mbbuild.log; exit 1; }; } &&
  tail -1 tmp/build365.log" || exit 1
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $D/bench365.sh $H:$T/tmp/bench365.sh || exit 3
"${SSH[@]}" "rm -rf $O; ARMS='${ARMS:-P1}' MB=${MB:-1} T=$T TTMH=$TTMH CACHE=/localdev/smarton/.cache/ttmc-gstt2-t358 O=$O MESH=P100 bash $T/tmp/bench365.sh ${1:-1}"
rc=$?
mkdir -p $D/${OUT:-out-p100a}
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$O/*" $D/${OUT:-out-p100a}/ 2>/dev/null
echo "=== drive end rc=$rc $(date -u +%FT%TZ)"; exit $rc
