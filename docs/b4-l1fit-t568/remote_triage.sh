#!/bin/bash
# t568: where does sort in blend hang? Runs the fuse arm's 30-view dump with the process in the
# background; if it is still running after RUN_TIMEOUT s, tt-triage (tools/tt-triage.py, tt-exalens
# venv /localdev/smarton/triage-venv) dumps every RISC's callstack while the hung process is alive
# (inspector RPC), then the process is killed by its pid and the card reset (tt-smi -r).
#   remote_triage.sh   (on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OPACITY_PRUNE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF GSPLAT_TT_BLEND_ALIAS_CUT GSPLAT_TT_KCFG_EXTRA_KB GSPLAT_TT_OL_TOWN_NT GSPLAT_TT_OL_SOLO GSPLAT_TT_SORT_IN_BLEND
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t555; mkdir -p $S
# Args or DBG_ARMS (default "1 2"): GSPLAT_TT_SIB_DBG per arm (1 = barrier only, no sort/town;
# 2 = full sort, 3 = sort body without the PACK town, 4 = PACK town handshake only, 5 = arm 4's
# BRISC mailbox writes with the PACK town off; all discard
# the result so the standalone sort reruns). TRIAGE=1 adds callstacks.
worst=0
for dbg in ${*:-${DBG_ARMS:-1 2}}; do
  tag=r0-sibdbg$dbg
  echo "=== $tag $(cut -c1-7 SHA) $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t555-dump-$tag
  env GSPLAT_TT_SIB_DBG=$dbg GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_SORT_IN_BLEND=1 TT_METAL_INSPECTOR=1 TT_METAL_INSPECTOR_RPC=1 \
    TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t513 \
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
  grep -E "^(SUMMARY|\[SORT\] (SORT_IN|stage)|\[B4A\])|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-300 | tail -12
  rm -rf tmp/t555-dump-$tag
  [ $rc -ne 0 ] && worst=$rc
done
exit $worst
