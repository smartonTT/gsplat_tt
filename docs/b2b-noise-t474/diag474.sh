#!/bin/bash
# t474 (remote, bh-30): find the b2b pass noise. Runs tree465b (13ce253e build) with render/run_t474.py
# (GSPLAT_B2B_ALL_STAGES=1: every stage timer per pass + pass end time), a 0.5 s host sampler
# (loadavg, aiclk, per-process CPU of the busiest other processes) and the listed variants.
#   diag474.sh <out dir> <tree> "<variant>..."   variants: base* | drop* (--b2b-drop) | pinN-M (taskset -c N-M)
#   | nice* (bench at nice -10 and session autogroup nice -10, via sudo -n; reverts when the session ends)
# With VSTART set it runs that viewer start script at exit and touches $O/vstarted.
set -u
O=$1; T=$2; VARS=$3; rc=99
rm -rf $O; mkdir -p $O
SAMP=
trap 'echo "=== diag end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/diag.rc; [ -n "$SAMP" ] && kill $SAMP 2>/dev/null
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 GSPLAT_B2B_ALL_STAGES=1
export TT_METAL_HOME=/localdev/smarton/viewer/tt-metal; export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
CACHE=/localdev/smarton/p150bench/cache465-B
{ echo "=== host $(hostname) $(date -u +%FT%TZ) nproc=$(nproc) cpuset=$(cat /sys/fs/cgroup/cpuset.cpus.effective)"
  echo "gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk') load=$(cat /proc/loadavg)"
  ps -eo pid,user,pcpu,etime,nlwp,comm --sort=-pcpu | head -8; } > $O/host.txt
cat $O/host.txt
# Sampler: every 0.5 s, epoch, load1, aiclk, and the CPU jiffies of every process using any (top 6 by delta).
( declare -A last; while :; do
    now=$(date +%s.%N); line="S $now load=$(cut -d' ' -f1 /proc/loadavg) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk')"
    out=""
    for st in /proc/[0-9]*/stat; do
      read -r pid comm rest < $st 2>/dev/null || continue
      set -- $rest; j=$(( ${12} + ${13} )); d=$(( j - ${last[$pid]:-$j} )); last[$pid]=$j
      [ $d -gt 5 ] && out="$out $pid$comm=$d"
    done
    echo "$line$out"; sleep 0.5
  done ) > $O/sampler.log 2>&1 &
SAMP=$!
rc=0
for v in $VARS; do
  pre=""; extra="--back-to-back --b2b-passes ${PASSES:-10}"; envs=""
  case $v in
    base*) ;;
    drop*) extra="$extra --b2b-drop" ;;
    pin*) r=${v#pin}; pre="taskset -c ${r%%_*}" ;;
    nice*) pre="nice -n 0" ;;  # niceness -10 + autogroup -10 set on the subshell below (sudo -n)
  esac
  it=t474-$v
  echo "=== $v start $(date -u +%T) $(date +%s.%N) load=$(cut -d' ' -f1 /proc/loadavg) pre='$pre' extra='$extra'"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/$it
    case $v in nice*) sudo -n renice -n -10 -p $BASHPID >/dev/null && echo -10 | sudo -n tee /proc/$BASHPID/autogroup >/dev/null
      echo "nice=$(ps -o ni= -p $BASHPID) autogroup=$(cat /proc/$BASHPID/autogroup)" ;; esac
    env $envs TT_METAL_CACHE_RENDER=$CACHE/render timeout 300 $pre python3 render/run_t474.py --no-ref --iter-dir $it $extra > $O/$v.log 2>&1 ); r=$?
  echo "$v rc=$r end $(date +%s.%N)"
  grep -E "^B2B scene|TT_FATAL|TT_THROW|Traceback" $O/$v.log | cut -c1-400
  [ $r = 0 ] || rc=20
  [ $r = 124 ] && { echo HANG; tt-smi -r > $O/reset-$v.log 2>&1; exit; }
  rm -rf $T/tmp/$it
done
