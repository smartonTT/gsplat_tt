#!/bin/bash
# t197 driver (Mac side). Each device step is its own `ttp lock p100` hold.
#   drive.sh <rev> [steps=sync,pc1,pc2]
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t197; O=docs/pfwc-breakdown-t197/out; mkdir -p $O
REV=${1:-HEAD}; STEPS=${2:-sync,pc2:9}
lk() {
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo LOCK_BUSY; echo CHAIN_DONE; exit 75; fi
  return $rc
}
IFS=, read -ra ST <<< "$STEPS"
for s in "${ST[@]}"; do
  case $s in
    sync) lk opt/sync_remote.sh $H $T "$REV"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; } ;;
    pc*) a=${s#pc}; l=${a%%:*}; sr=${a#*:}   # pc<stepcyc>:<steprisc>, e.g. pc2:9
         lk $DEVRUN --host $H --no-verify --timeout 560 --tag t197-pc$l-r$sr -- "bash $T/docs/pfwc-breakdown-t197/remote_prof.sh $l $sr"
         echo "PC${l}_R${sr}_RC=$?"
         D=$T/opt/profiler/t197-pc$l-r$sr
         scp -q -o BatchMode=yes "$H:$D/capture.log" "$H:$D/pc_split.txt" "$H:$D/dev.csv.gz" $O/ 2>/dev/null
         for x in capture.log pc_split.txt dev.csv.gz; do [ -f $O/$x ] && mv $O/$x $O/pc$l-r$sr-$x; done ;;
  esac
done
echo CHAIN_DONE
