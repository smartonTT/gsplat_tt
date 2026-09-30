#!/bin/bash
# 30-view dump for knob <k> (0 = old form, 1 = DEST staging). With k=1, md5-compare against k0.
set -u
source /localdev/smarton/t68_scripts/remote_env.sh
k=$1
rm -rf tmp/t68-dump-k$k
echo "== k=$k run $(date +%T)"
GSPLAT_TT_BLEND_COEF_DEST=$k TT_METAL_CACHE_RENDER=$(cache $k) timeout 360 \
  python3 render/run.py --no-ref --iter-dir t68-k$k --dump-views t68-dump-k$k > tmp/t68-run-k$k.log 2>&1
echo "run rc=$? $(date +%T)"; grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL|hero" tmp/t68-run-k$k.log | head
d=$(find . -maxdepth 3 -type d -name t68-dump-k$k | head -1); echo "dump dir: $d"
(cd $d && md5sum * | sort -k2) > /tmp/t68-md5-k$k.txt; echo "k=$k views: $(wc -l < /tmp/t68-md5-k$k.txt)"
[ $k -eq 1 ] || exit 0
[ "$(wc -l < /tmp/t68-md5-k1.txt)" -ge 30 ] && diff /tmp/t68-md5-k0.txt /tmp/t68-md5-k1.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL k0 vs k1" || { echo "VIEWS DIFFER"; diff /tmp/t68-md5-k0.txt /tmp/t68-md5-k1.txt | head; }
head -3 /tmp/t68-md5-k1.txt
