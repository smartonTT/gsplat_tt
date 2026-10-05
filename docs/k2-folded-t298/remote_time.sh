#!/bin/bash
# t298: one untraced bicycle 30-view run; sweep md5 (906e0435 = golden since iter 205).
#   remote_time.sh <round> <arm> [ENV=V ...]   (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t298; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t298; mkdir -p $S
tag=r$1-$2; shift 2
echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
rm -rf tmp/t298-dump-$tag
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t298 timeout ${RUN_TIMEOUT:-230} \
  python3 render/run.py --no-ref --iter-dir t298-$tag --dump-views t298-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^([A-Z_]*STAGES|SUMMARY)|\[TA\] K2|\[SORT\] ONELAUNCH|Traceback|TT_THROW|TT_FATAL|Error|error:" $S/run-$tag.log | head -20
d=$(find . -maxdepth 3 -type d -name t298-dump-$tag | head -1)
if [ -n "$d" ]; then
  (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
  echo "VIEWS=$(wc -l < $S/md5-$tag.txt) SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8)"
  rm -rf "$d"
fi
exit $rc
