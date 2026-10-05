#!/bin/bash
# t291: untraced bicycle 30-view A/B of GSPLAT_TT_K2_TRISC (one build, runtime knob).
#   remote_ab.sh <round> [arm ...]   arm = name[:ENV=V,ENV2=V]  (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t291; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t291; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
for s in ${*:-A}; do
  IFS=: read -r name envs <<< "$s"
  tag=r$r-$name
  echo "=== $tag $(cut -c1-7 SHA) ${envs:-} $(date +%T)"
  rm -rf tmp/t291-dump-$tag
  env ${envs//,/ } TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t291 timeout ${RUN_TO:-240} \
    python3 render/run.py --no-ref --iter-dir t291-$tag --dump-views t291-dump-$tag > $S/run-$tag.log 2>&1
  echo "run rc=$?"
  grep -E "^SUMMARY|Traceback|TT_THROW|TT_FATAL|error:|hard fail|too large" $S/run-$tag.log | sort | uniq -c | head -8
  if [ -d tmp/t291-dump-$tag ]; then
    (cd tmp/t291-dump-$tag && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt) $(diff -q $REF $S/md5-$tag.txt >/dev/null && echo ALL_VIEWS_IDENTICAL_TO_GOLDEN || echo differs_from_golden)"
    rm -rf tmp/t291-dump-$tag
  fi
done
echo "=== done round $r $(date +%T)"
