#!/bin/bash
# t394: remote half on bh-30 (viewer stopped by drive_bh30.sh). tree394 = t388's xview code.
# 3 rotating untraced 30-view rounds of base | xv | xvpin (same arms and env as #388's
# remote_time.sh, GSPLAT_PER_VIEW_STAGES unset like #388): md5 of the 30 dumped views vs the
# 906e0435 golden list, xview_hits == views-1 for the xv* arms, hero copy of every xvpin run.
# ALWAYS restarts the viewer itself on exit (same steps as opt/viewer/viewer.sh start), so the
# viewer comes back even if the Mac side dies. Viewer's tt-metal used read-only.
set -u
P=/localdev/smarton/p150bench; T=$P/tree394; V=/localdev/smarton/viewer; O=$P/out394; ROUNDS=${1:-3}
restart_viewer() {
  ( cd $V/tree || exit 1
    pid=$(cat $V/viewer.pid 2>/dev/null); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "viewer already running $pid"; exit 0; }
    export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TT_METAL_RUNTIME_ROOT=$V/tt-metal GSPLAT_SHA=$(cat SHA)
    export TT_METAL_CACHE=$V/tt-metal-cache NUMPY_MADVISE_HUGEPAGE=0
    unset TTW_DEVRUN PYTHONDONTWRITEBYTECODE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OUT_PINNED VIRTUAL_ENV
    [ -f $V/viewer.log ] && mv -f $V/viewer.log $V/viewer.prev.log
    rm -f $V/viewer.pid $V/viewer.stop
    setsid nohup bash opt/viewer/supervise.sh $V 8080 > $V/viewer.log 2>&1 < /dev/null &
    for _ in $(seq 25); do [ -s $V/viewer.pid ] && break; sleep 0.2; done
    echo "=== viewer restarted pid $(cat $V/viewer.pid 2>/dev/null) $(date -u +%FT%TZ)" )
}
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; restart_viewer; echo $rc > $O/bench.rc' EXIT
rc=99
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal TT_METAL_ARCH_NAME=blackhole
export TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
run_one() {  # round arm
  local t=t394-r$1-$2 rr n hits e=()
  case $2 in xv) e=(GSPLAT_TT_XVIEW_OVERLAP=1) ;; xvpin) e=(GSPLAT_TT_XVIEW_OVERLAP=1 GSPLAT_TT_OUT_PINNED=1) ;; esac
  rm -rf tmp/$t tmp/$t-dump
  echo "=== r$1 $2 start $(date -u +%T) ${e[*]:-defaults}"
  env "${e[@]}" TT_METAL_CACHE_RENDER=$P/cache394 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/run-r$1-$2.log 2>&1
  rr=$?; echo "run r$1 $2 rc=$rr $(date -u +%T)"
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/run-r$1-$2.log | cut -c1-400 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-r$1-$2.log 2>&1; return 124; fi
  [ $rr = 0 ] || return $rr
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-r$1-$2.txt; rm -rf tmp/$t-dump
  n=$(wc -l < $O/md5-r$1-$2.txt)
  diff -q $REF $O/md5-r$1-$2.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL r$1 $2 ($n)" \
    || { echo "VIEWS_DIFFER r$1 $2 ($(diff $REF $O/md5-r$1-$2.txt | grep -c '^>') of $n)"; return 6; }
  case $2 in xv*)
    hits=$(grep -m1 "^STAGES " $O/run-r$1-$2.log | sed -n 's/.* xview_hits=\([0-9]*\).*/\1/p')
    if [ "${hits:-x}" = $((n - 1)) ]; then echo "XVIEW_HITS_OK r$1 $2 hits=$hits views=$n"
    else echo "XVIEW_HITS_BAD r$1 $2 hits=${hits:-none} views=$n"; return 7; fi ;;
  esac
  [ $2 = xvpin ] && cp tmp/$t/hero_clean.png $O/hero-r$1-xvpin.png
  return 0
}
rc=0
for ra in 1:base,xv,xvpin 2:xvpin,base,xv 3:xv,xvpin,base; do r=${ra%%:*}
  for a in $(echo ${ra#*:} | tr , ' '); do run_one $r $a || { rc=$?; [ $rc = 124 ] && break 2; }; done
done
for a in base xv xvpin; do
  grep -h "^STAGES " $O/run-r[123]-$a.log 2>/dev/null | sed -n 's/.*avg_frame_ms=\([0-9.]*\).*/\1/p' |
    awk -v a=$a '{s+=$1; n++; v=v" "$1} END {if (n) printf "ARM %s mean_ms_view=%.3f n=%d rounds:%s\n", a, s/n, n, v}'
done | tee $O/summary.txt
grep -m2 -h -E "firmware bundle version|KMD version" $O/run-r1-base.log 2>/dev/null
