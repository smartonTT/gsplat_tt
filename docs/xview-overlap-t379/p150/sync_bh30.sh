#!/bin/bash
# t394 (from t387 sync_bh30.sh): own tree on bh-30 for the t379 cross-view overlap p150 check
# (viewer keeps running during this). First call copies #382's bench tree (p150bench/tree382,
# iter-210; venv/scenes are symlinks into the viewer dir, used read-only) to p150bench/tree394
# without build dirs and tmp, syncs the files that differ from its SHA to HEAD (fresh mtimes)
# and builds render/build-tt under nice -n 19 ionice -c3 with half the cores.
# Never touches /localdev/smarton/viewer (except the tt-metal read-only), /localdev/smarton/gstt2 or /home.
#   docs/xview-overlap-t379/p150/sync_bh30.sh     (task holds resource 'viewer' exclusive)
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; T=$P/tree394; TTMH=/localdev/smarton/viewer/tt-metal
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" "test -d $T || { mkdir -p $T && cd $P/tree382 && tar --exclude=./tmp --exclude=./render/build-tt \
  --exclude=./render/bench/build --exclude=./generated -cf - . | tar -xf - -C $T && mkdir -p $T/tmp && echo copied; }" || exit 3
base=$("${SSH[@]}" "cat $T/SHA") || exit 3
echo "=== sync $base -> $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
git diff --name-only --diff-filter=d "$base" HEAD | tar -cf - -T - | "${SSH[@]}" "cd $T && tar -xmf - && echo $(git rev-parse HEAD) > SHA" || exit 3
"${SSH[@]}" "cd $T && export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole && J=\$(( \$(nproc) / 2 )) &&
  { test -f render/build-tt/build.ninja || nice -n 19 ionice -c3 cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg394.log 2>&1 || { tail -30 tmp/cfg394.log; exit 1; }; } &&
  { nice -n 19 ionice -c3 cmake --build render/build-tt -j \$J > tmp/build394.log 2>&1 || { tail -30 tmp/build394.log; exit 1; }; } &&
  tail -1 tmp/build394.log && ls -la render/render_clean*.so | cut -c25- && cat SHA"
rc=$?; echo "=== sync rc=$rc $(date -u +%FT%TZ)"; exit $rc
