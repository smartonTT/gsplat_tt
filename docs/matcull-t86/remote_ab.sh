#!/bin/bash
# t86: interleaved rounds, base tree (a5f2bd6, /localdev/smarton/gstt2-t86b) vs fix tree, 30 views, untraced.
set -u
S=/localdev/smarton/t86_scripts
for r in ${*:-1 2 3}; do
  for v in base fix; do
    T=/localdev/smarton/gstt2-t86; [ $v = base ] && T=/localdev/smarton/gstt2-t86b
    echo "=== round=$r $v"
    T86_TREE=$T bash $S/remote_run.sh ab$r-$v /localdev/smarton/t82_scripts/md5-r82new.txt | grep -E "^(SUMMARY|STAGES|SORT_STAGES)|IDENTICAL|DIFFER|rc="
  done
done
