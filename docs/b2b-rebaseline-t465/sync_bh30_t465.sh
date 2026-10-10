#!/bin/bash
# t465: make p150bench/<tree> at <rev> on bh-30 with the viewer running: copy tree392 (31bf48c4; venv/scenes
# are symlinks into the viewer dir, read-only) without tmp and build dirs, sync the files that differ from its
# SHA to <rev>, build render/build-tt under nice -n 19 ionice -c3 with half the cores (user, 2026-10-07).
# Never touches /localdev/smarton/viewer, /localdev/smarton/gstt2 or /home.
#   sync_bh30_t465.sh <tree name> <rev>
set -u
cd "$(git rev-parse --show-toplevel)"
P=/localdev/smarton/p150bench; T=$P/$1; REV=$(git rev-parse "$2") || exit 2; TTMH=/localdev/smarton/viewer/tt-metal
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 bh-30)
"${SSH[@]}" "test -d $T || { mkdir -p $T && cd $P/tree392 && tar --exclude=./tmp --exclude=./render/build-tt \
  --exclude=./render/bench/build --exclude=./generated -cf - . | tar -xf - -C $T && mkdir -p $T/tmp && echo copied; }" || exit 3
base=$("${SSH[@]}" "cat $T/SHA") || exit 3
echo "=== sync $1 ${base:0:8} -> ${REV:0:8} $(date -u +%FT%TZ)"
git diff --name-only --diff-filter=d "$base" "$REV" | grep -vE '^(opt/metal-screenshots|opt/profiler/ttw-)' > tmp/t465-files-$1.txt
git archive --format=tar "$REV" $(cat tmp/t465-files-$1.txt) | "${SSH[@]}" "cd $T && tar -xmf - && echo $REV > SHA" || exit 3
"${SSH[@]}" "cd $T && export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole && J=\$(( \$(nproc) / 2 )) &&
  { test -f render/build-tt/build.ninja || nice -n 19 ionice -c3 cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg465.log 2>&1 || { tail -30 tmp/cfg465.log; exit 1; }; } &&
  { nice -n 19 ionice -c3 cmake --build render/build-tt -j \$J > tmp/build465.log 2>&1 || { tail -30 tmp/build465.log; exit 1; }; } &&
  tail -1 tmp/build465.log && ls -la render/render_clean*.so | cut -c25- && cat SHA"
rc=$?; echo "=== sync rc=$rc $(date -u +%FT%TZ)"; exit $rc
