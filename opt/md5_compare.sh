#!/usr/bin/env bash
# Compare per-view md5 lists. Fails unless every file has exactly N_VIEWS
# non-empty lines and all files are identical (guards against empty-file false passes).
# Usage: opt/md5_compare.sh <n_views> <md5file> <md5file> [...]
# Exit: 0 match, 1 mismatch, 2 bad/short/empty file or usage.
set -u
[ $# -ge 3 ] || { echo "usage: $0 <n_views> <md5file> <md5file> [...]" >&2; exit 2; }
n=$1; shift
ref=""
for f in "$@"; do
  [ -f "$f" ] || { echo "[md5_compare] FAIL missing $f"; exit 2; }
  c=$(grep -c . "$f" || true)
  [ "$c" -eq "$n" ] || { echo "[md5_compare] FAIL $f has $c non-empty lines, want $n"; exit 2; }
  if [ -z "$ref" ]; then ref=$f; continue; fi
  if ! diff -q <(grep . "$ref") <(grep . "$f") >/dev/null; then
    echo "[md5_compare] FAIL $f differs from $ref"; exit 1
  fi
done
echo "[md5_compare] PASS $# files, $n views each"
