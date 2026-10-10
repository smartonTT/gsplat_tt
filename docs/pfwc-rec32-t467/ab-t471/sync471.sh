#!/bin/bash
# t471 (after docs/eth-launch-t410/sync_bh30.sh): own tree p150bench/tree471 on bh-30, viewer keeps running.
# First call copies tree392 (31bf48c4; venv/scenes are symlinks into the viewer dir, used read-only)
# without build dirs and tmp, then syncs the files that differ from its SHA to HEAD and builds
# render/build-tt under nice -n 19 ionice -c3 with half the cores (user, 2026-10-07).
# Never touches /localdev/smarton/viewer, /localdev/smarton/gstt2, tree392 or /home.
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; T=$P/tree471; TTMH=/localdev/smarton/viewer/tt-metal
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" "test -d $T || { mkdir -p $T && cd $P/tree392 && tar --exclude=./tmp --exclude=./render/build-tt \
  --exclude=./render/bench/build --exclude=./generated --exclude='./render/render_clean*.so' -cf - . | tar -xf - -C $T && mkdir -p $T/tmp && echo copied; }" || exit 3
base=$("${SSH[@]}" "cat $T/SHA") || exit 3
echo "=== sync $base -> $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
git diff --name-only --diff-filter=d "$base" HEAD | tar -cf - -T - | "${SSH[@]}" "cd $T && tar -xmf - && echo $(git rev-parse HEAD) > SHA" || exit 3
"${SSH[@]}" "cd $T && export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole && J=\$(( \$(nproc) / 2 )) &&
  { test -f render/build-tt/build.ninja || nice -n 19 ionice -c3 cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg471.log 2>&1 || { tail -30 tmp/cfg471.log; exit 1; }; } &&
  { nice -n 19 ionice -c3 cmake --build render/build-tt -j \$J > tmp/build471.log 2>&1 || { tail -30 tmp/build471.log; exit 1; }; } &&
  tail -1 tmp/build471.log && ls -la render/render_clean*.so | cut -c25- && cat SHA"
rc=$?; echo "=== sync rc=$rc $(date -u +%FT%TZ)"; exit $rc
