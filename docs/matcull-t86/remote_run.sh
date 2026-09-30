#!/bin/bash
# t86: 30-view bicycle run, untraced; prints stage lines. <tag> [md5 ref file]
set -u
source /localdev/smarton/t86_scripts/remote_env.sh
tag=$1; ref=${2:-}
rm -rf tmp/t86-dump-$tag
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $T) timeout 330 \
  python3 render/run.py --no-ref --iter-dir t86-$tag --dump-views t86-dump-$tag > tmp/t86-run-$tag.log 2>&1
echo "run rc=$?"; grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|MAT_STATS|mat.*items" tmp/t86-run-$tag.log | head -12
d=$(find . -maxdepth 3 -type d -name t86-dump-$tag | head -1)
(cd $d && md5sum * | sort -k2) > $S/md5-$tag.txt; echo "$tag views: $(wc -l < $S/md5-$tag.txt)"
[ -z "$ref" ] && exit 0
diff $ref $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL vs $ref" || { echo "VIEWS DIFFER"; diff $ref $S/md5-$tag.txt | head -4; }
