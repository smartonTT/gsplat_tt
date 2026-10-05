#!/bin/bash
# 30-view dump; <tag> names the run. Compares md5s against <ref> (a tag or an md5 file) if given.
set -u
source /localdev/smarton/t83_scripts/remote_env.sh
tag=$1; ref=${2:-}
rm -rf tmp/t83-dump-$tag
echo "== $tag run $(date +%T)"
TT_METAL_CACHE_RENDER=$(cache 0 -v$tag) timeout 330 \
  python3 render/run.py --no-ref --iter-dir t83-$tag --dump-views t83-dump-$tag > tmp/t83-run-$tag.log 2>&1
echo "run rc=$? $(date +%T)"; grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL|hero" tmp/t83-run-$tag.log | head
d=$(find . -maxdepth 3 -type d -name t83-dump-$tag | head -1); echo "dump dir: $d"
(cd $d && md5sum * | sort -k2) > /localdev/smarton/t83_scripts/md5-$tag.txt; echo "$tag views: $(wc -l < /localdev/smarton/t83_scripts/md5-$tag.txt)"
[ -z "$ref" ] && exit 0
R=/localdev/smarton/t83_scripts/md5-$ref.txt; [ -f "$ref" ] && R=$ref
[ "$(wc -l < /localdev/smarton/t83_scripts/md5-$tag.txt)" -ge 30 ] && diff $R /localdev/smarton/t83_scripts/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL $ref vs $tag" || { echo "VIEWS DIFFER"; diff $R /localdev/smarton/t83_scripts/md5-$tag.txt | head -4; }
