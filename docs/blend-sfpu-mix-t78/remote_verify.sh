#!/bin/bash
# 30-view dump for ablation <a> (0 = default, 2 = NOP padding); a=2 md5-compares against a=0.
set -u
source /localdev/smarton/t78_scripts/remote_env.sh
a=$1
rm -rf tmp/t78-dump-a$a
echo "== a=$a run $(date +%T)"
GSPLAT_TT_BLEND_ABL=$a TT_METAL_CACHE_RENDER=$(cache $a) timeout 330 \
  python3 render/run.py --no-ref --iter-dir t78-a$a --dump-views t78-dump-a$a > tmp/t78-run-a$a.log 2>&1
echo "run rc=$? $(date +%T)"; grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL|hero" tmp/t78-run-a$a.log | head
d=$(find . -maxdepth 3 -type d -name t78-dump-a$a | head -1); echo "dump dir: $d"
(cd $d && md5sum * | sort -k2) > /tmp/t78-md5-a$a.txt; echo "a=$a views: $(wc -l < /tmp/t78-md5-a$a.txt)"
[ $a -eq 0 ] && exit 0
[ "$(wc -l < /tmp/t78-md5-a$a.txt)" -ge 30 ] && diff /tmp/t78-md5-a0.txt /tmp/t78-md5-a$a.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL a0 vs a$a" || { echo "VIEWS DIFFER"; diff /tmp/t78-md5-a0.txt /tmp/t78-md5-a$a.txt | head -4; }
