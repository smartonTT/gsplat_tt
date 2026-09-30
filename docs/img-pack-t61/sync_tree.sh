#!/bin/bash
# Sync a git tree-ish to a fresh /localdev/smarton/gstt2-<variant> on yyzo-bh-07
# (tar -m so every file gets a fresh mtime). Usage: sync_tree.sh <variant> [treeish]
# Wrap in `ttp lock p100 -- ...`.
set -e
V=$1; T=${2:-HEAD}
cd "$(git rev-parse --show-toplevel)"
R="set -e; D=/localdev/smarton/gstt2-$V; rm -rf \$D; mkdir -p \$D; tar -m -x -C \$D; cd \$D; rm -f .venv; ln -s /localdev/smarton/gstt2/.venv .venv; ln -s /localdev/smarton/gstt2/scenes scenes; cp /localdev/smarton/gstt2/backends/cpu_cpp/_gsplat_cpu*.so backends/cpu_cpp/; echo synced \$D \$(git -C . rev-parse 2>/dev/null || true)"
git archive "$T" CMakeLists.txt cmake render backends benchmarks gsplat src profiler tests docs/img-pack-t61 | ssh -o BatchMode=yes yyzo-bh-07 "$R"
