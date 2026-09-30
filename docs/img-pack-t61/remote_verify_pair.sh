#!/bin/bash
# Build + one dump run for base and candidate, then compare the 30 views.
# Usage: remote_verify_pair.sh <base_variant> <cand_variant>   (runs under devrun.sh)
set -u
S=/localdev/smarton/t61_scripts
bash $S/remote_build_dump.sh $1 || exit 1
bash $S/remote_build_dump.sh $2 || exit 1
A=/localdev/smarton/gstt2-$1/tmp/t61-md5.txt; B=/localdev/smarton/gstt2-$2/tmp/t61-md5.txt
[ "$(wc -l < $A)" = 30 ] && diff $A $B > /dev/null && echo "ALL_30_VIEWS_IDENTICAL $1 vs $2" || { echo "VIEWS_DIFFER"; diff $A $B | head; exit 2; }
