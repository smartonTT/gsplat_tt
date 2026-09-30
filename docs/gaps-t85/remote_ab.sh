#!/bin/bash
# t85: 3 interleaved rounds, base tree (9cced7e) vs fix tree, bicycle 30 views, untraced.
set -u
S=/localdev/smarton/t85_scripts
for r in ${*:-1 2 3}; do
  for v in base fix; do
    T=/localdev/smarton/gstt2-t85; [ $v = base ] && T=/localdev/smarton/gstt2-t85b
    echo "=== round=$r $v"
    T85_TREE=$T bash $S/remote_run.sh ab$r-$v /localdev/smarton/t82_scripts/md5-r82new.txt | grep -E "^(SUMMARY|STAGES|TILE_ASSIGN_STAGES)|IDENTICAL|DIFFER"
  done
done
