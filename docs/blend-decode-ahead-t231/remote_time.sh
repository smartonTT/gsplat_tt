#!/bin/bash
# t231 (copy of t229 remote_time.sh): untraced bicycle 30-view timing, md5 vs md5-r82new.txt, one or more arms per call.
#   remote_time.sh <round> [arm ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
# arms: base = default env (off); name:ENV=V,ENV2=V = default env plus those variables
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T231_TREE:-/localdev/smarton/gstt2-t231}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t231; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
run() {  # tag [ENV=V ...]
  local tag=r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
  rm -rf tmp/t231-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t231 timeout ${RUN_TO:-120} \
    python3 render/run.py --no-ref --iter-dir t231-$tag --dump-views t231-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|hard fail|\[DEV\]|Program size|too large|WRITER_SPLIT|KCFG|BLEND_SCHED|DECODE_AHEAD|RAW_STAGE|#error" $S/run-$tag.log | sort | uniq -c | sort -rn | head -16
  local d; d=$(find . -maxdepth 3 -type d -name t231-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    diff -q $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL ($(wc -l < $S/md5-$tag.txt) views)" \
      || echo "VIEWS DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $(wc -l < $S/md5-$tag.txt))"
    rm -rf "$d"
  fi
  return $rc
}
for s in ${*:-base}; do
  case $s in
    base) run base ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } ;;
  esac
done
echo "=== done round $r $(date +%T)"
