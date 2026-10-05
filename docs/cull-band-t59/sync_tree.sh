#!/bin/bash
# Sync a git tree-ish of this repo to /localdev/smarton/gstt2-<variant> on yyzo-bh-07.
# Usage: t59_sync.sh <variant> [treeish]   (treeish default HEAD; "WORK" = working tree via tar)
set -e
V=$1; T=${2:-HEAD}
cd "$(git rev-parse --show-toplevel)"
R="set -e; D=/localdev/smarton/gstt2-$V; rm -rf \$D; mkdir -p \$D; tar -x -C \$D; cd \$D; ln -s /localdev/smarton/gstt2/.venv .venv; ln -s /localdev/smarton/gstt2/scenes scenes; cp /localdev/smarton/gstt2/backends/cpu_cpp/_gsplat_cpu*.so backends/cpu_cpp/; echo synced \$D"
if [ "$T" = "WORK" ]; then
  tar -c CMakeLists.txt cmake render backends benchmarks gsplat src profiler tests docs/cull-band-t59 | ssh -o BatchMode=yes yyzo-bh-07 "$R"
else
  git archive "$T" CMakeLists.txt cmake render backends benchmarks gsplat src profiler tests docs/cull-band-t59 2>/dev/null | ssh -o BatchMode=yes yyzo-bh-07 "$R" || \
  git archive "$T" CMakeLists.txt cmake render backends benchmarks gsplat src profiler tests | ssh -o BatchMode=yes yyzo-bh-07 "$R"
fi
