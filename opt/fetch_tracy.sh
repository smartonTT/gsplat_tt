#!/usr/bin/env bash
# Copy a remote render.tracy into opt/profiler/ttw-NNN/render.tracy so a device
# task can set a repo-relative "tracy" field in iters.jsonl and report links work.
#   opt/fetch_tracy.sh <host> <remote_tracy_path> <NNN>
# Path convention: opt/profiler/ttw-<NNN>/render.tracy (git-ignored: traces are large).
set -euo pipefail
HOST=${1:?host}; SRC=${2:?remote tracy path}; N=${3:?ttw number}
DST="opt/profiler/ttw-$N"; mkdir -p "$DST"
scp -o BatchMode=yes "$HOST:$SRC" "$DST/render.tracy"
echo "tracy: $DST/render.tracy"
