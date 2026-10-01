#!/bin/bash
# t90: one 30-view run with GSPLAT_TT_SPLIT_BLEND=1 (Finish after mat and after cull)
# so STAGES shows mat / cull / blend separately.   remote_split.sh <tag> [tree]
set -u
T=${2:-/localdev/smarton/gstt2-t86}
echo "=== split $1 $(cat $T/SHA)"
GSPLAT_TT_SPLIT_BLEND=1 T86_TREE=$T bash /localdev/smarton/t86_scripts/remote_run.sh t90split-$1 \
  /localdev/smarton/t82_scripts/md5-r82new.txt | grep -E "^(SUMMARY|STAGES|SORT_STAGES)|IDENTICAL|DIFFER|rc="
