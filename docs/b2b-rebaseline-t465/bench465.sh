#!/bin/bash
# t465 (remote): re-baseline best-iter-216 (arm A) vs the opt tip (arm B) on one board, under the
# headline metric: back-to-back ms/frame (run.py --back-to-back, 1 check pass + 3 measured passes)
# and latency without --dump-views (secondary). Both trees run the same run.py (ttp/t465 copy,
# render/run_t465.py). One smoke per arm (kernel compile), then ROUNDS alternating rounds
# (odd A then B, even B then A); each round: b2b A/B, then latency A/B. b2b dumps pass 0 (untimed)
# for the md5 list. Writes $O/*.log, md5-*.txt, hero-<arm>.png, bench.rc; with VSTART set it runs
# that viewer start script at exit and touches $O/vstarted.
#   bench465.sh <p100|p150> <out dir> <tree A> <tree B> [rounds=3 | "<step tag>..."]
# Step mode (devrun's 600 s reservation ceiling): the 5th arg lists tags (init, s0-A, r2-lat-B, ...);
# init clears $O; the result goes to $O/step.rc, and the last step ending the plan writes bench.rc.
set -u
BOARD=$1; O=$2; TA=$3; TB=$4; ROUNDS=${5:-3}; STEPS=
case $ROUNDS in *[!0-9]*) STEPS=$ROUNDS ;; esac
RCF=$O/bench.rc; [ -n "$STEPS" ] && RCF=$O/step.rc
rc=99
case " ${STEPS:-init} " in *" init "*) rm -rf $O ;; esac; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $RCF
  if [ -n "${VSTART:-}" ]; then echo "=== viewer start $(date -u +%FT%TZ)"; bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
if [ $BOARD = p150 ]; then
  export TT_METAL_HOME=/localdev/smarton/viewer/tt-metal; CACHE=/localdev/smarton/p150bench/cache465
else
  export TT_METAL_HOME=/localdev/smarton/tt-metal MESH_DEVICE=P100; CACHE=/localdev/smarton/.cache/ttmc-t465
fi
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
echo "=== bench $BOARD host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type' 2>/dev/null) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "A=$TA sha=$(cut -c1-8 $TA/SHA)  B=$TB sha=$(cut -c1-8 $TB/SHA)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW|Traceback'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
run() {  # run <tag> <A|B> <smoke|b2b|lat>
  local tag=$1 arm=$2 mode=$3 T=$TA rr args; [ $arm = B ] && T=$TB
  local it=t465-$tag
  case $mode in
    smoke) args="--view-range 0:1" ;;
    b2b) args="--back-to-back --b2b-passes 3 --dump-views $it-dump" ;;
    lat) args="" ;;
  esac
  echo "=== $tag start $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/$it tmp/$it-dump
    TT_METAL_CACHE_RENDER=$CACHE-$arm/render timeout $([ $mode = smoke ] && echo ${SMOKE_TO:-900} || echo ${RUN_TO:-300}) \
      python3 render/run_t465.py --no-ref --iter-dir $it $args > $O/$tag.log 2>&1 ); rr=$?
  echo "$tag rc=$rr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^B2B scene|TTW_TIMING (ms_view|b2b_ms_frame(_median)?)=|$BAD" $O/$tag.log | cut -c1-500 | head -8
  hang $rr $tag
  [ $rr = 0 ] && ! grep -qE "$BAD" $O/$tag.log || return 1
  if [ $mode = b2b ]; then
    (cd $T/tmp/$it-dump && md5sum * | sort -k2) > $O/md5-$tag.txt
    echo "LIST_MD5 $tag $(md5sum < $O/md5-$tag.txt | cut -c1-8) views=$(wc -l < $O/md5-$tag.txt)"
    (cd $T && python3 opt/md5_golden.py $O/md5-$tag.txt $O/$tag.log) 2>&1 | tail -2
    [ -f $O/hero-$arm.png ] || cp $T/tmp/$it/hero_clean.png $O/hero-$arm.png
  fi
  rm -rf $T/tmp/$it $T/tmp/$it-dump
}
if [ -n "$STEPS" ]; then
  rc=0
  for t in $STEPS; do
    case $t in init) continue ;; s0-?) m=smoke ;; *) m=$(echo $t | cut -d- -f2) ;; esac
    run $t ${t##*-} $m || { rc=20; [ $m = smoke ] && rc=10; exit; }
  done
  exit
fi
rc=10
run s0-A A smoke && run s0-B B smoke || exit
rc=0
for r in $(seq 1 $ROUNDS); do
  if [ $((r % 2)) = 1 ]; then o="A B"; else o="B A"; fi
  for m in b2b lat; do for a in $o; do run r$r-$m-$a $a $m || { rc=20; exit; }; done; done
done
