#!/usr/bin/env bash
# Task #408: rebuild the microblock-shape model from scratch (CPU only, ~20 min on an M-series Mac).
#   run.sh [PLY] [WORKDIR]   default PLY = repo scenes/bicycle.ply, WORKDIR = /tmp/mb-shape-model
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PLY=${1:-$HERE/../../scenes/bicycle.ply}
WORK=${2:-/tmp/mb-shape-model}
mkdir -p "$WORK/cache" "$HERE/out"
SDK=()
if [[ "$(uname)" == Darwin && -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then
  SDK=(-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk)  # same workaround as tests/unit/run_cpp.sh
fi
c++ "${SDK[@]}" -O3 -std=c++17 -ffp-contract=off -pthread -o "$WORK/mbsim" "$HERE/mbsim.cpp"
python3 "$HERE/project.py" "$WORK/cache" --ply "$PLY"
for f in "$WORK"/cache/*.bin; do
  "$WORK/mbsim" "$f" > "$HERE/out/$(basename "$f" .bin).txt"
done
python3 "$HERE/aggregate.py" "$HERE/out"
