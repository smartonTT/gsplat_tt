#!/bin/bash
# Build + one dump run for a base and a candidate tree, then compare the
# 30 dumped views. Usage: remote_verify_pair.sh <base_variant> <cand_variant>
set -u
S=/localdev/smarton/t60_scripts
bash $S/remote_build_dump.sh $1 || exit 1
bash $S/remote_build_dump.sh $2 || exit 1
for v in $1 $2; do (cd /localdev/smarton/gstt2-$v/tmp/t60-dump && md5sum * | sort -k2 > /tmp/t60-md5-$v.txt; echo "$v views: $(wc -l < /tmp/t60-md5-$v.txt)"); done
diff /tmp/t60-md5-$1.txt /tmp/t60-md5-$2.txt > /dev/null && echo "ALL_30_VIEWS_IDENTICAL $1 vs $2" || { echo "VIEWS DIFFER"; diff /tmp/t60-md5-$1.txt /tmp/t60-md5-$2.txt | head; }
