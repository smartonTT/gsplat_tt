#!/bin/bash
# t213: sync + build the committed tree, then one devrun of three 30-view arms:
# base (defaults), ovf (GSPLAT_TT_PAIR_CAP_TEST=1000000, defaults on: early sort
# + CQ1) and ovf_nocq1 (the same with GSPLAT_TT_MAT_CQ1=0). One ttp lock p100 per step.
#   drive.sh [host] [rev]   (Mac)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=${1:-yyzo-bh-04}; T=/localdev/smarton/gstt2-t213
O=docs/pair-overflow-t213/out; mkdir -p $O
C=GSPLAT_TT_PAIR_CAP_TEST=1000000
export PATH="$PWD/docs/pair-overflow-t213/sshbin:$PATH"  # strict host keys for devrun
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
"${TTP_PROJECT:-tt-project}/harness/bin/ssh-preflight" $H || { echo PREFLIGHT_FAIL; echo CHAIN_DONE; exit 12; }
lk opt/sync_remote.sh $H $T "${2:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
lk $DEVRUN --host $H --no-verify --timeout 590 --tag t213-smoke -- \
  "bash $T/docs/pair-overflow-t213/remote_smoke.sh base ovf:$C ovf_nocq1:$C,GSPLAT_TT_MAT_CQ1=0"
echo "SMOKE_RC=$?"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$H:$T/tmp/t213/*" $O/
echo CHAIN_DONE
