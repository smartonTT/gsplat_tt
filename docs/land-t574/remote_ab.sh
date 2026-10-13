#!/bin/bash
# t574 (land the #573 BRISC stack fix): one step of the binary A/B on the measurement box.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | dump   arm: base | new
#   base = best-iter-226 (90268457) in $TB with JIT cache ttmc-t574base
#   new  = the t574 rev in $T with JIT cache ttmc-t574new (fresh caches: every kernel compiles once)
#   b2b:  render/run.py --no-ref --back-to-back (headline metric, 30 bicycle views)
#   dump: render/run.py --no-ref --dump-views, md5 list checked with opt/md5_golden.py, then the
#         static stack margin of every brisc.elf in the arm's cache (opt/brisc_stack_check.py
#         --reserve 0, from $T) into $S/stack-<arm>.jsonl
#   (run through devrun.sh inside drive.sh's ttp lock p100; env as docs/land-t525/remote_ab.sh)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK
T=${T:-/localdev/smarton/gstt2-t513}; TB=${TB:-/localdev/smarton/gstt2-t574b}
C=/localdev/smarton/.cache; S=$T/tmp/t574; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
for arm in "$@"; do
  case $arm in base) dir=$TB ;; new) dir=$T ;; *) echo "bad arm $arm"; exit 2 ;; esac
  cache=$C/ttmc-t574$arm
  tag=r$r-$mode-$arm
  cd "$dir" || exit 1; source .venv/bin/activate
  args=(--no-ref --iter-dir t574-$tag)
  case $mode in b2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t574-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-8 SHA) $dir cache=$cache $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t574-dump-$tag
  TT_METAL_CACHE_RENDER=$cache timeout ${RUN_TIMEOUT:-150} python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|\[DEV\]|BRISC_STACK_FAIL)|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  echo "brisc_stack_lines=$(grep -c '^BRISC_STACK ' $S/run-$tag.log)"
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t574-dump-$tag | head -1)
    if [ -n "$d" ]; then
      (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
      python3 $T/opt/md5_golden.py $S/md5-$tag.txt $S/run-$tag.log || { [ $rc = 0 ] && rc=6; }
      rm -rf "$d"
    else echo "NO_DUMP_DIR"; [ $rc = 0 ] && rc=7; fi
    python3 $T/opt/brisc_stack_check.py --scan $cache --reserve 0 > $S/stack-$arm.jsonl
    echo "stack scan rc=$? elfs=$(wc -l < $S/stack-$arm.jsonl)"
    python3 - $S/stack-$arm.jsonl <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    k = r["elf"].split("/kernels/")[-1].split("/")[0]
    print(f"STACK {k} margin={r['margin']} depth={r['depth']} avail={r['avail']} "
          f"udivdi3={'__udivdi3' in r['chain']}")
PY
  fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $mode $(date +%T)"
exit $fail
