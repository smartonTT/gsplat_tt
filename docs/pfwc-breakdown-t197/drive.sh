#!/bin/bash
# t197 driver (Mac side). Each device step is its own `ttp lock p100` hold.
#   drive.sh <rev> [steps=sync,pc1,pc2]
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t197; O=docs/pfwc-breakdown-t197/out; mkdir -p $O
REV=${1:-HEAD}; STEPS=${2:-sync,pc2:TR0+TR1+TR2+NC+BR}
lk() {
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo LOCK_BUSY; echo CHAIN_DONE; exit 75; fi
  return $rc
}
IFS=, read -ra ST <<< "$STEPS"
for s in "${ST[@]}"; do
  case $s in
    sync) lk opt/sync_remote.sh $H $T "$REV"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; } ;;
    pc*) l=${s#pc}; l=${l%%:*}; rs=${s#*:}; rs=${rs//+/ }   # e.g. pc2:TR0+TR1+TR2
         lk $DEVRUN --host $H --no-verify --timeout 560 --tag t197-pc$l -- "bash $T/docs/pfwc-breakdown-t197/remote_prof.sh $l $rs"
         echo "PC${l}_RC=$?"
         for r in $rs; do scp -q -o BatchMode=yes "$H:$T/tmp/t197/pc$l-$r.dprint" "$H:$T/tmp/t197/pc$l-$r.log" $O/; done ;;
  esac
done
echo CHAIN_DONE
