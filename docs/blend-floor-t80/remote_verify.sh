#!/bin/bash
# 30-view dump; <tag> names the run. Compares md5s against tag <ref> if given.
set -u
source /localdev/smarton/t80_scripts/remote_env.sh
tag=$1; ref=${2:-}
rm -rf tmp/t80-dump-$tag
echo "== $tag run $(date +%T)"
TT_METAL_CACHE_RENDER=$(cache 0 -$tag) timeout 330 \
  python3 render/run.py --no-ref --iter-dir t80-$tag --dump-views t80-dump-$tag > tmp/t80-run-$tag.log 2>&1
echo "run rc=$? $(date +%T)"; grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL|hero" tmp/t80-run-$tag.log | head
d=$(find . -maxdepth 3 -type d -name t80-dump-$tag | head -1); echo "dump dir: $d"
(cd $d && md5sum * | sort -k2) > /localdev/smarton/t80_scripts/md5-$tag.txt; echo "$tag views: $(wc -l < /localdev/smarton/t80_scripts/md5-$tag.txt)"
[ -z "$ref" ] && exit 0
[ "$(wc -l < /localdev/smarton/t80_scripts/md5-$tag.txt)" -ge 30 ] && diff /localdev/smarton/t80_scripts/md5-$ref.txt /localdev/smarton/t80_scripts/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL $ref vs $tag" || { echo "VIEWS DIFFER"; diff /localdev/smarton/t80_scripts/md5-$ref.txt /localdev/smarton/t80_scripts/md5-$tag.txt | head -4; }
