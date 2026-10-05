#!/bin/bash
# t275: untraced bicycle 30-view timing, latency vs back-to-back, md5 vs md5-r82new.txt.
#   remote_time.sh <round> [arm ...]   (on yyzo-bh-07 through devrun.sh, under ttp lock p100)
# arms: lat = render/run.py latency mode (primary ms/view)
#       b2b = --back-to-back (check pass + 3 measured passes that keep frames, md5-compared)
#       drop = --back-to-back --b2b-drop (measured passes drop each frame, viewer-like)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T275_TREE:-/localdev/smarton/gstt2-t275}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t275; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
run() {  # name [run.py args ...]
  local tag=r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* $(date +%T)"
  rm -rf tmp/t275-dump-$tag
  TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t275 timeout ${RUN_TO:-170} \
    python3 render/run.py --no-ref --iter-dir t275-$tag --dump-views t275-dump-$tag "$@" > $S/run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^(SUMMARY|B2B)|B2B pass|TTW_TIMING (ms_view|b2b)|Traceback|TT_THROW|TT_FATAL|hard fail" $S/run-$tag.log | head -12
  if [ -d tmp/t275-dump-$tag ]; then
    (cd tmp/t275-dump-$tag && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8) VIEWS=$(wc -l < $S/md5-$tag.txt) $(diff -q $REF $S/md5-$tag.txt >/dev/null && echo ALL_VIEWS_IDENTICAL_TO_GOLDEN || echo differs_from_golden)"
    rm -rf tmp/t275-dump-$tag
  fi
  return $rc
}
for a in ${*:-lat b2b drop}; do
  case $a in
    lat) run lat ;;
    b2b) run b2b --back-to-back ;;
    drop) run drop --back-to-back --b2b-drop ;;
  esac
done
echo "=== done round $r $(date +%T)"
