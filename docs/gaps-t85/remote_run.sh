#!/bin/bash
# t85: 30-view bicycle run, untraced; prints all stage lines. <tag> [md5 ref file]
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T85_TREE:-/localdev/smarton/gstt2-t85}; cd $T || exit 1; source .venv/bin/activate
tag=$1; ref=${2:-}; S=/localdev/smarton/t85_scripts; mkdir -p $S
rm -rf tmp/t85-dump-$tag
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-t85-$tag timeout 330 \
  python3 render/run.py --no-ref --iter-dir t85-$tag --dump-views t85-dump-$tag > tmp/t85-run-$tag.log 2>&1
echo "run rc=$?"; grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL" tmp/t85-run-$tag.log
d=$(find . -maxdepth 3 -type d -name t85-dump-$tag | head -1)
(cd $d && md5sum * | sort -k2) > $S/md5-$tag.txt; echo "$tag views: $(wc -l < $S/md5-$tag.txt)"
[ -z "$ref" ] && exit 0
diff $ref $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL vs $ref" || { echo "VIEWS DIFFER"; diff $ref $S/md5-$tag.txt | head -4; }
