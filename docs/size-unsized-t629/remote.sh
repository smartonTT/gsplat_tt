#!/bin/bash
# t629: one step of the b2b floor probes (bh-33, FOLD=1 default build).
#   remote.sh <round> <arm> ...   arms:
#     base    defaults, no instrumentation (headline config)
#     htl     GSPLAT_TT_HTL=1 host timeline (htl-<tag>.txt)
#     hsN     HTL + GSPLAT_TT_HTL_FOLD_SPIN_US=N (host delay between pfwc(N+1) and mat+blend(N) enqueue)
#     psN     HTL + GSPLAT_TT_HTL_PRE_SPIN_US=N (host delay just before the pfwc(N+1) enqueue)
#     dsN     HTL + GSPLAT_TT_PFWC_SPIN_US=N (device delay at the end of the pfwc BRISC writer)
#     tsN     HTL + GSPLAT_TT_PFWC_TSPIN_US=N (MATH TRISC busy N us per core per pfwc launch: knee probe)
#     f0      HTL + GSPLAT_TT_B4A_FOLD=0 (context only)
#     gN      run.py --view-gap-ms N/10 (host spin at the view boundary, outside the render window;
#             e.g. g3 = 0.3 ms): sizes device idle vs queued work between blend(N) done and view N+1
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_HTL_FOLD_SPIN_US GSPLAT_TT_PFWC_SPIN_US GSPLAT_TT_HTL_PRE_SPIN_US GSPLAT_TT_PFWC_TSPIN_US
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t629; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  tag=r$r-$arm
  env=(); args=()
  case $arm in
    base) ;;
    htl) env=(GSPLAT_TT_HTL=1) ;;
    ps*) env=(GSPLAT_TT_HTL=1 GSPLAT_TT_HTL_PRE_SPIN_US=${arm#ps}) ;;
    hs*) env=(GSPLAT_TT_HTL=1 GSPLAT_TT_HTL_FOLD_SPIN_US=${arm#hs}) ;;
    ts*) env=(GSPLAT_TT_HTL=1 GSPLAT_TT_PFWC_TSPIN_US=${arm#ts}) ;;
    ds*) env=(GSPLAT_TT_HTL=1 GSPLAT_TT_PFWC_SPIN_US=${arm#ds}) ;;
    f0) env=(GSPLAT_TT_HTL=1 GSPLAT_TT_B4A_FOLD=0) ;;
    g*) env=(GSPLAT_TT_HTL=1); args=(--view-gap-ms "$(awk "BEGIN{print ${arm#g}/10}")") ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]} ${args[*]+${args[*]}} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${env[@]}" GSPLAT_TT_HTL_OUT=$S/htl-$tag.txt TT_METAL_CACHE_RENDER=$C/ttmc-t629 \
    timeout ${RUN_TIMEOUT:-200} python3 render/run.py --no-ref --iter-dir t629-$tag --back-to-back ${args[@]+"${args[@]}"} > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|HTL )|TTW_TIMING (b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-300 | head -12
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ -f $S/htl-$tag.txt ] && gzip -f $S/htl-$tag.txt
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $(date +%T)"
exit $fail
