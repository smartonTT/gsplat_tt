#!/bin/bash
# t573: t568's remote_triage.sh plus, per arm, the BRISC/NCRISC/TRISC kernel ELF sizes of the
# fused mat+blend program (newest matblend_* ELFs in the JIT cache after the run) and, on a hang,
# tt-triage callstacks. Args: GSPLAT_TT_SIB_DBG arms; "<arm>f0" = that arm with GSPLAT_TT_B4A_FOLD=0.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OPACITY_PRUNE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF GSPLAT_TT_BLEND_ALIAS_CUT GSPLAT_TT_KCFG_EXTRA_KB GSPLAT_TT_OL_TOWN_NT GSPLAT_TT_OL_SOLO GSPLAT_TT_SORT_IN_BLEND GSPLAT_TT_B4A_FOLD
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t555; mkdir -p $S
C=/localdev/smarton/.cache/ttmc-gstt2-t513
RV=$(ls $TT_METAL_HOME/runtime/sfpi/compiler/bin/*readelf 2>/dev/null | head -1); RV=${RV:-readelf}
worst=0
for a in ${*:-1 5}; do
  dbg=${a%f0}; fold=; [ "$a" != "$dbg" ] && fold="GSPLAT_TT_B4A_FOLD=0"
  tag=r0-sibdbg$a
  echo "=== $tag $(cut -c1-7 SHA) $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg) $fold"
  rm -rf tmp/t555-dump-$tag; touch $S/stamp-$tag; sleep 1
  env $fold GSPLAT_TT_SIB_DBG=$dbg GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_SORT_IN_BLEND=1 TT_METAL_INSPECTOR=1 TT_METAL_INSPECTOR_RPC=1 \
    TT_METAL_CACHE_RENDER=$C \
    python3 render/run.py --no-ref --iter-dir t555-$tag --dump-views t555-dump-$tag > $S/run-$tag.log 2>&1 &
  pid=$!
  for _ in $(seq 1 "${RUN_TIMEOUT:-150}"); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  if kill -0 $pid 2>/dev/null; then
    echo "HANG in $tag"
    [ "${TRIAGE:-0}" = 1 ] && (cd "$TT_METAL_HOME" && timeout 300 /localdev/smarton/triage-venv/bin/python tools/tt-triage.py \
      --run=dump_callstacks --llm-output --disable-progress --skip-version-check > $S/triage-$tag.log 2>&1)
    [ "${TRIAGE:-0}" = 1 ] && echo "triage rc=$? lines=$(wc -l < $S/triage-$tag.log)"
    kill $pid; sleep 5; kill -9 $pid 2>/dev/null
    tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"
    rc=124
  else
    wait $pid; rc=$?
  fi
  echo "run rc=$rc"
  grep -E "^(SUMMARY|\[SORT\] (SORT_IN|stage)|\[B4A\])|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-300 | tail -8
  # ELFs of this arm (new in the cache) or, when cached, every matblend ELF with the arm's define
  find $C -path '*matblend_*' -name '*.elf' -newer $S/stamp-$tag 2>/dev/null > $S/elfs-$tag.txt
  echo "ELFS $tag new=$(wc -l < $S/elfs-$tag.txt)"
  while read -r e; do
    echo "ELF $e"; "$RV" -lSW "$e" | grep -E "LOAD|\.text|\.data|\.bss|Entry" | head -12
  done < $S/elfs-$tag.txt
  rm -rf tmp/t555-dump-$tag
  [ $rc -ne 0 ] && worst=$rc
done
exit $worst
