#!/bin/bash
# t464: one viewer-downtime window on bh-30 (viewer stopped by the Mac driver; restarted here at exit).
# All arms: bench tree392 (#435 build, 31bf48c4) + its venv + env, render/run_t464.py (= ttp/t464-b2b-gap
# run.py: --view-gap-ms spin, per-pass B2B_STAGES), 30 bicycle views 1024x1024, --no-ref.
set -u
P=/localdev/smarton/p150bench; O=$P/out464; T=$P/tree392
rm -rf $O; mkdir -p $O
trap 'rm -f $T/render/run_t464.py; echo "=== viewer start $(date -u +%FT%TZ)"; bash $P/vstart464.sh > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted' EXIT
mv $P/run_t464.py $T/render/run_t464.py
echo "=== probe start $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
run() {  # arm mode gap [VAR=val ...]
  local a=$1 mode=$2 gap=$3 rr; shift 3
  ( cd $T || exit 9
    unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP
    export TT_METAL_HOME=/localdev/smarton/viewer/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
    export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache435/render
    for kv in "$@"; do export "$kv"; done
    dump=; [ "${DUMP:-0}" = 1 ] && dump="--dump-views t464-$a-dump"
    rm -rf tmp/t464-$a tmp/t464-$a-dump
    echo "=== $a start $(date -u +%T) mode=$mode gap=$gap env=[$*] load=$(cut -d' ' -f1 /proc/loadavg)"
    timeout 150 .venv/bin/python3 render/run_t464.py --no-ref --iter-dir t464-$a $dump --view-gap-ms $gap \
      $([ $mode = b2b ] && echo --back-to-back) > $O/$a.log 2>&1; rr=$?
    echo "$a rc=$rr $(date -u +%T)"
    [ $rr = 124 ] && { echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$a.log 2>&1; }
    grep -E "^B2B |^B2B_STAGES|TTW_TIMING (ms_view|b2b_ms_frame|stage_(project|sort|blend|xview|pybind|view_total))=|^STAGES|Traceback|TT_THROW|TT_FATAL" $O/$a.log | cut -c1-420
    if [ -d tmp/t464-$a-dump ]; then
      (cd tmp/t464-$a-dump && md5sum * | sort -k2) > $O/md5-$a.txt
      python3 $T/opt/md5_golden.py $O/md5-$a.txt $O/$a.log
    fi
    cp tmp/t464-$a/hero_clean.png $O/hero-$a.png 2>/dev/null
    rm -rf tmp/t464-$a tmp/t464-$a-dump )
}
# latency: reference condition of t460 BL (per-view stages + PNG dump between views)
run Lref lat 0 GSPLAT_PER_VIEW_STAGES=1 DUMP=1
# back-to-back: reference + host-gap sweep (spin outside the timed window)
run B0   b2b 0 DUMP=1
run B05  b2b 0.5
run B1   b2b 1
run B2   b2b 2
run B3   b2b 3
run BX0  b2b 0 GSPLAT_TT_XVIEW_OVERLAP=0
# latency without dump: only a print between views, then the spin gap
run L0   lat 0
run L1   lat 1
run L2   lat 2
run LX0  lat 0 GSPLAT_TT_XVIEW_OVERLAP=0
run LXH  lat 0 GSPLAT_TT_XVIEW_OVERLAP=0 GSPLAT_TT_HOST_PROFILE=1
echo "=== probe end $(date -u +%FT%TZ)"
