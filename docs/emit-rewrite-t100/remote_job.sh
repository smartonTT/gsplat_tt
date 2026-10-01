#!/bin/bash
# t100: sort_bucket_emit runs on the remote tree (default /localdev/smarton/gstt2-t100),
# bicycle 30 views, untraced, md5 vs md5-r82new.txt.
#   remote_job.sh <round> [steps]
# steps: base, or aN = GSPLAT_TT_EMIT_ABLATE=N (profiling ablation, output wrong),
# or any ENV=VALUE-tagged step "name:ENV=V[,ENV2=V2]" for A/B variants.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T100_TREE:-/localdev/smarton/gstt2-t100}; cd "$T" || exit 1; source .venv/bin/activate
S=/localdev/smarton/t100_scripts; mkdir -p $S tmp
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
r=${1:-1}; shift
run() {  # tag [ENV=V ...]; non-zero if the run failed
  local tag=t100r$r-$1; shift
  echo "=== $tag $(cut -c1-7 SHA) $*"
  rm -rf tmp/t100-dump-$tag
  env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $T) timeout 330 \
    python3 render/run.py --no-ref --iter-dir t100-$tag --dump-views t100-dump-$tag > tmp/t100-run-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|EMIT_ABLATE" tmp/t100-run-$tag.log | head -12
  local d; d=$(find . -maxdepth 3 -type d -name t100-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    diff -q $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL ($(wc -l < $S/md5-$tag.txt) views)" \
      || echo "VIEWS DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $(wc -l < $S/md5-$tag.txt))"
  fi
  return $rc
}
for s in ${*:-base}; do
  case $s in
    base) run base || exit 2 ;;
    a[0-9]*) run $s GSPLAT_TT_EMIT_ABLATE=${s#a} || exit 3 ;;
    *:*) name=${s%%:*}; envs=${s#*:}; run $name ${envs//,/ } || exit 4 ;;
  esac
done
echo "=== done round $r"
