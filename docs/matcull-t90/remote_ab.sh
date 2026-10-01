#!/bin/bash
# t90: interleaved rounds, base tree (a5f2bd6 code: /localdev/smarton/gstt2-t85 @4d71d96,
# render/ identical) vs fix tree (/localdev/smarton/gstt2-t86), bicycle 30 views, untraced.
#   remote_ab.sh [rounds...]      e.g. remote_ab.sh 1 2
set -u
S=/localdev/smarton/t86_scripts
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
for r in ${*:-1 2}; do
  for v in base fix; do
    T=/localdev/smarton/gstt2-t86; [ $v = base ] && T=/localdev/smarton/gstt2-t85
    echo "=== round=$r $v $(cat $T/SHA)"
    T86_TREE=$T bash $S/remote_run.sh t90ab$r-$v $REF | grep -E "^(SUMMARY|STAGES|SORT_STAGES)|IDENTICAL|DIFFER|rc="
  done
done
