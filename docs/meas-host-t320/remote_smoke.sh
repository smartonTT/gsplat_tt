#!/bin/bash
# t320: first run on the new measurement box (yyzo-bh-04, IRD job 244892): board
# info, then one untraced bicycle 30-view run at defaults, md5 vs the golden list
# (906e0435, iter 205+). Run through devrun.sh, under ttp lock p100.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1
T=${T:-/localdev/smarton/gstt2}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t320; mkdir -p $S
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== board"
tt-smi -s > $S/tt-smi.json 2>/dev/null; python3 - $S/tt-smi.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for b in d.get("device_info", []):
    bi = b.get("board_info", {})
    print("BOARD", bi.get("board_type"), bi.get("board_id"), "fw", b.get("firmwares", {}).get("fw_bundle_version"))
PY
echo "=== smoke $(cut -c1-7 SHA) $(date +%T)"
rm -rf tmp/t320-dump
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2 timeout ${RUN_TIMEOUT:-500} \
  python3 render/run.py --no-ref --iter-dir t320-smoke --dump-views t320-dump > $S/run.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|Error|error:" $S/run.log | head -20
d=$(find . -maxdepth 3 -type d -name t320-dump | head -1)
if [ -n "$d" ]; then
  (cd "$d" && md5sum * | sort -k2) > $S/md5.txt
  echo "SWEEP_MD5 $(md5sum < $S/md5.txt | cut -c1-8) views=$(wc -l < $S/md5.txt)"
  diff -q $REF $S/md5.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL to golden" \
    || echo "VIEWS DIFFER ($(diff $REF $S/md5.txt | grep -c '^>') of $(wc -l < $S/md5.txt))"
  rm -rf "$d"
fi
[ $rc = 124 ] || [ $rc = 137 ] && { echo "HANG: tt-smi -r"; tt-smi -r > $S/reset.log 2>&1; }
exit $rc
