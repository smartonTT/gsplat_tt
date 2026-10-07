#!/bin/bash
# t212 re-baseline of the smarton/tt-project-opt tip on the measurement box, one
# `ttp lock p100` around everything: sync + build + device hero shot (opt/ttw/screenshot.sh,
# 30-view sweep), then 3 untraced 30-view timing rounds at defaults on the same tree.
#   docs/rebaseline-t212/drive.sh <iter> <rev>     (from the repo root; run via ttp detach)
set -u
cd "$(git rev-parse --show-toplevel)"
it=${1:?iter}; rev=${2:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t212}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=docs/rebaseline-t212/out; mkdir -p "$O"
if [ "${3:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/docs/rebaseline-t212/drive.sh" "$it" "$rev" --locked
fi
echo "=== drive t212 $(git rev-parse --short "$rev") on $H:$T $(date)"
opt/ttw/screenshot.sh "$it" "$rev" --locked; src=$?
echo "screenshot rc=$src"
[ $src -eq 0 ] || exit $src
scp -q -o BatchMode=yes docs/rebaseline-t212/remote_time.sh "$H:$T/tmp/t212_remote_time.sh" || exit 3
ssh -o BatchMode=yes "$H" 'cat /proc/cpuinfo | grep -m1 "model name"; tt-smi -ls 2>/dev/null | head -20' > "$O/board.txt" 2>&1
rc=0
for r in 1 2 3; do
  $DEVRUN --host "$H" --no-verify --timeout 420 --tag t212-r$r -- \
    "T=$T bash $T/tmp/t212_remote_time.sh $r base" 2>&1 | tee "$O/round$r.out" | grep -E "^(===|run rc|SUMMARY|ALL_|VIEWS)"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
done
scp -q -o BatchMode=yes "$H:$T/tmp/t212/run-r*-base.log" "$H:$T/tmp/t212/md5-r*-base.txt" "$O/" || rc=4
echo "=== drive done rc=$rc $(date)"
exit $rc
