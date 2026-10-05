#!/bin/bash
# t146 (copy of the t124 script): blend diet A/B on the remote tree (default /localdev/smarton/gstt2-t124),
# bicycle, untraced, md5 vs md5-r82new.txt.
#   remote_job.sh <round> [steps]
# step: base, on (GSPLAT_TT_SORT_ONELAUNCH=1), or "name:ENV=V[,ENV2=V2]".
# VIEWS=0:5 limits the views (--view-range); the md5 compare then covers those views.
set -u; export LC_ALL=C
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T146_TREE:-/localdev/smarton/gstt2-t146}; cd "$T" || exit 1; source .venv/bin/activate
S=/localdev/smarton/t146_scripts; mkdir -p $S tmp
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
VR=(); [ -n "${VIEWS:-}" ] && VR=(--view-range "$VIEWS")
run() {  # tag [ENV=V ...]; non-zero if the run failed
  local tag=t146r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $* ${VIEWS:+views=$VIEWS}"
  rm -rf tmp/t146-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $T) timeout ${TMO:-330} \
    python3 render/run.py --no-ref "${VR[@]}" --iter-dir t146-$tag --dump-views t146-dump-$tag \
    > tmp/t146-run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|[Ww]atcher.*(assert|ASSERT|hang|error)|[Cc]ircular buffer|L1.*(clash|overflow|exceed)" \
    tmp/t146-run-$tag.log | head -14
  grep -E "ONELAUNCH_CHECK" tmp/t146-run-$tag.log | sort | uniq -c | head -5
  grep -oE "onelaunch=[0-9.]+ layout=[0-9.]+ pub_host=[0-9.]+ total=[0-9.]+" tmp/t146-run-$tag.log |
    awk -F'[= ]' '{n++; a+=$2; b+=$4; c+=$6; d+=$8} END {if (n) printf "ONELAUNCH n=%d onelaunch=%.3f layout=%.3f pub_host=%.3f total=%.3f\n", n, a/n, b/n, c/n, d/n}'
  grep -E "MAT_STATS|max_ncrisc" tmp/t146-run-$tag.log | tail -3
  grep -m1 -E "\[SORT\] ONELAUNCH v2" tmp/t146-run-$tag.log
  local d; d=$(find . -maxdepth 3 -type d -name t146-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    local n; n=$(wc -l < $S/md5-$tag.txt)
    local bad; bad=$(join -j2 <(sort -k2 $REF) <(sort -k2 $S/md5-$tag.txt) | awk '$2!=$3' | wc -l)
    local miss; miss=$(join -j2 -v2 <(sort -k2 $REF) <(sort -k2 $S/md5-$tag.txt) | wc -l)
    [ "$bad" = 0 ] && [ "$miss" = 0 ] && echo "ALL_VIEWS_IDENTICAL ($n views)" || echo "VIEWS DIFFER ($bad differ, $miss not in ref, of $n)"
  fi
  return $rc
}
for s in ${*:-base}; do
  case $s in
    base) run base || exit 2 ;;
    on) run on GSPLAT_TT_SORT_ONELAUNCH=1 || exit 3 ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } || exit 4 ;;
  esac
done
echo "=== done round $r"
