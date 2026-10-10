#!/bin/bash
# t471 (remote, bh-30; viewer stopped by drive471.sh, restarted at exit by vstart471.sh, then out471/vstarted).
# A/B of GSPLAT_TT_PFWC_REC32 on tree471 (t467 head + a rec32 log line), ETH 12x10 default dispatch.
#   O = REC32=0 (today's default), R = REC32=1.
# 1-view smoke per arm (kernel compile; R must log "rec32 on", O must not), then ROUNDS alternating rounds
# (odd O then R, even R then O). Each round per arm: back-to-back (headline; 1 check pass + PASSES measured,
# pass 0 dumped untimed for the md5 list), then latency without --dump-views (secondary).
set -u
P=/localdev/smarton/p150bench; T=$P/tree471; V=/localdev/smarton/viewer; O=$P/out471; ROUNDS=${1:-4}; PASSES=${2:-5}
rc=99
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc; echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart471.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED GSPLAT_TT_PFWC_REC32
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache471/render
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
armenv() { [ $1 = R ] && echo "GSPLAT_TT_PFWC_REC32=1" || echo "GSPLAT_TT_PFWC_REC32=0"; }
for a in O R; do
  echo "=== s0-$a start $(date -u +%T)"
  env $(armenv $a) timeout 300 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t471-s0-$a > $O/s0-$a.log 2>&1; s0=$?
  echo "s0-$a rc=$s0 $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|rec32|$BAD|Traceback" $O/s0-$a.log | cut -c1-300 | head -8
  if [ $s0 != 0 ] || grep -qE "$BAD" $O/s0-$a.log || ! grep -qE "^\[DEV\] dispatch eth .*compute grid 12x10" $O/s0-$a.log; then
    hang $s0 s0-$a; echo "S0_FAIL $a"; rc=10; exit
  fi
  if [ $a = R ] && ! grep -q 'rec32 on' $O/s0-$a.log; then echo "S0_FAIL R: no rec32 line"; rc=11; exit; fi
  if [ $a = O ] && grep -q 'rec32 on' $O/s0-$a.log; then echo "S0_FAIL O: rec32 line"; rc=12; exit; fi
  echo "S0_PASS $a"
done
rc=0
b2b() {  # b2b <tag> <O|R>
  local t=t471-$1 rr; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 start $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env $(armenv $2) timeout 300 python3 render/run.py --no-ref --back-to-back --b2b-passes $PASSES --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^B2B |$BAD|Traceback" $O/$1.log | cut -c1-400 | head -4
  hang $rr $1; [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png; rm -rf tmp/$t
  echo "LIST_MD5 $1 $(md5sum < $O/md5-$1.txt | cut -c1-8)"
  python3 opt/md5_golden.py $O/md5-$1.txt $O/$1.log || rc=20
}
lat() {  # lat <tag> <O|R>
  local t=t471-$1 rr; rm -rf tmp/$t
  echo "=== $1 start $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env $(armenv $2) timeout 300 python3 render/run.py --no-ref --iter-dir $t > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "TTW_TIMING ms_view=|$BAD|Traceback" $O/$1.log | cut -c1-300 | head -3
  rm -rf tmp/$t; hang $rr $1; [ $rr = 0 ] || { rc=$rr; return 1; }
}
for r in $(seq 1 $ROUNDS); do
  case $((r % 2)) in
    0) b2b r$r-b-R R && b2b r$r-b-O O && lat r$r-l-R R && lat r$r-l-O O ;;
    *) b2b r$r-b-O O && b2b r$r-b-R R && lat r$r-l-O O && lat r$r-l-R R ;;
  esac || break
done
