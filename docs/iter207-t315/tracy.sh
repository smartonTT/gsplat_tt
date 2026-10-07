#!/bin/bash
# t315: Tracy trace for kept iter 207 on yyzo-bh-04, under one `ttp lock p100`. Uses the tree
# drive.sh synced and built (no resync). Default env first; if the profiler build overflows L1,
# retry with GSPLAT_TT_PFWC_WRITER_SPLIT=0 as iter 206 did. Copies render.tracy to opt/profiler/ttw-207.
set -u
cd "$(git rev-parse --show-toplevel)"
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t315}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
O=docs/iter207-t315/out; mkdir -p "$O" opt/profiler/ttw-207
if [ "${1:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/docs/iter207-t315/tracy.sh" --locked; fi
"$TTP_PROJECT/harness/bin/ssh-preflight" "$H" || exit 9
scp -q -o BatchMode=yes docs/iter207-t315/remote_tracy.sh "$H:$T/tmp/t315_remote_tracy.sh" || exit 3
ok=
for try in "ttw-207" "ttw-207 GSPLAT_TT_PFWC_WRITER_SPLIT=0"; do
  $DEVRUN --host "$H" --no-verify --timeout 560 --tag t315-tracy -- "T=$T bash $T/tmp/t315_remote_tracy.sh $try" \
    2>&1 | tee -a "$O/tracy.out" | grep -E "^===|capture_tracy\]|zones rc|TT_FATAL|HANG"
  if ssh -o BatchMode=yes "$H" test -s "$T/opt/profiler/ttw-207/render.tracy"; then ok="$try"; break; fi
done
[ -n "$ok" ] || { echo "=== no trace"; exit 2; }
scp -q -o BatchMode=yes "$H:$T/opt/profiler/ttw-207/render.tracy" "$H:$T/opt/profiler/ttw-207/zones.txt" opt/profiler/ttw-207/ || exit 4
echo "env=$ok" > opt/profiler/ttw-207/capture_env.txt
echo "=== tracy ok ($ok) $(ls -la opt/profiler/ttw-207/render.tracy)"
