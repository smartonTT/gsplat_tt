#!/bin/bash
# t213: forced pair-overflow smoke. 30-view bicycle bench per arm, md5 of every
# dumped view; the overflow arms must match the default arm view for view.
#   remote_smoke.sh [arm ...]   (on the measurement box through devrun, under ttp lock p100)
# arms: base = default env; name:ENV=V,ENV2=V = default env plus those variables
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T213_TREE:-/localdev/smarton/gstt2-t213}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t213; mkdir -p $S
run() {  # tag [ENV=V ...]
  local tag=$1; shift
  echo "=== $tag $(cut -c1-8 SHA) $* $(date +%T)"
  rm -rf tmp/t213-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t213 timeout -k 20 ${RUN_TO:-240} \
    python3 render/run.py --no-ref --iter-dir t213-$tag --dump-views t213-dump-$tag > $S/run-$tag.log 2>&1
  local rc=$?
  echo "RUN_RC=$rc"
  grep -E "^SUMMARY|^TTW_TIMING ms_view|Traceback|TT_THROW|TT_FATAL|RuntimeError|hard fail|pair overflow|fused K2 failed|\[DEV\]" \
    $S/run-$tag.log | sort | uniq -c | sort -rn | head -12
  local d; d=$(find . -maxdepth 3 -type d -name t213-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "VIEWS=$(wc -l < $S/md5-$tag.txt) SETMD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8)"
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
for f in $S/md5-*.txt; do
  [ "$f" = $S/md5-base.txt ] && continue
  diff -q $S/md5-base.txt $f > /dev/null && echo "MATCH_BASE $(basename $f)" || echo "DIFFERS_FROM_BASE $(basename $f)"
done
echo "=== done $(date +%T)"
