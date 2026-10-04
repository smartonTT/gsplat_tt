#!/bin/bash
# t165: swapped paired A/B, untraced, 4 rounds (base/tp, tp/base, ...) in one device lock.
#   ab.sh   (Mac; remote tree already synced to HEAD)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t165; O=docs/emit-tpack-t165/out; mkdir -p $O
R="bash $T/docs/emit-tpack-t165/remote_time.sh"; TP=tp:GSPLAT_TT_OL_EMIT_TPACK=1
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t165-ab -- \
  "$R 1 base $TP; $R 2 $TP base; $R 3 base $TP; $R 4 $TP base"
echo "AB_RC=$?"
scp -q -o BatchMode=yes "$H:$T/tmp/t165/run-r[1-4]-*.log" "$H:$T/tmp/t165/md5-r[1-4]-*.txt" $O/
echo CHAIN_DONE
