#!/bin/bash
# t582 (L2, 2-chip alternate-frame rendering): device half of docs/l2-2chip/drive.sh. Runs ON the
# 2-board p150 host (bh-50, IRD job 132971) under `ttp detach --remote`, inside drive.sh's
# `ttp lock p100`. Tree $T (synced by opt/sync_remote.sh), JIT cache $C, outputs in $S.
#   A = one chip (PCIe id $DA), render/run.py --no-ref --back-to-back
#   B = 2 chips, GSPLAT_TT_AFR=1 GSPLAT_TT_AFR_DEVICES=$DA,$DB, same command
#   C = the other chip alone ($DB), for MB1 (shared-host cost per chip)
# Steps: device list; md5 dumps (A latency mode, which also compiles every kernel; B with
# --dump-views, which also writes both chips' heroes); a discarded warm-up round A B; 3 rounds
# A/B alternating (A B C, B A C, A B C); 3 single-chip latency runs (AFR has no latency mode).
# Every step's rc goes to $S/steps.txt; the first non-zero one, or 0, goes to $S/DONE on exit.
set -u
R=${R:-/localdev/smarton/nosync/t582}   # outside the user's mutagen sync (see drive.sh)
export TT_METAL_HOME=$R/tt-metal TT_METAL_RUNTIME_ROOT=$R/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P150 TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export GSPLAT_TT_DISPATCH=worker
unset GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_AFR GSPLAT_TT_AFR_DEVICES GSPLAT_TT_AFR_VISIBLE \
  TT_VISIBLE_DEVICES GSPLAT_TT_DEVICE_ID
T=${T:-$R/gstt2}; S=$T/tmp/t582; C=$R/.cache/ttmc-t582
# The IRD container shows only its two boards (/dev/tenstorrent/5 and 6) and UMD numbers them 0 and 1.
DA=${DA:-0}; DB=${DB:-1}
mkdir -p "$S"; rm -f "$S/DONE"
first=0
trap 'echo "$first" > "$S/DONE"; echo "=== remote_drive done first_rc=$first $(date +%T)"' EXIT
cd "$T" || { first=1; exit 1; }
source .venv/bin/activate || { first=1; exit 1; }
VIS=1   # 0 once TT_VISIBLE_DEVICES turns out not to work (AFR falls back to GSPLAT_TT_DEVICE_ID)

pat="^(B2B |SUMMARY|\[DEV\]|\[afr[0-9]\] \[DEV\]|\[afr\] |BRISC_STACK_FAIL)|TTW_TIMING (ms_view|b2b_ms_frame)|Traceback|TT_THROW|TT_FATAL|error:"
step_rc() {  # step_rc <tag> <rc>
  echo "$1 rc=$2" >> "$S/steps.txt"
  [ "$2" = 0 ] || [ "$first" != 0 ] || first=$2
}
# one <tag> <arm A|B|C> <timeout s> <run.py args...>
one() {
  local tag=$1 arm=$2 to=$3; shift 3
  local -a envs=(TT_METAL_CACHE_RENDER=$C)
  case $arm in
    A) if [ $VIS = 1 ]; then envs+=(TT_VISIBLE_DEVICES=$DA GSPLAT_TT_DEVICE_ID=0); else envs+=(GSPLAT_TT_DEVICE_ID=$DA); fi ;;
    C) if [ $VIS = 1 ]; then envs+=(TT_VISIBLE_DEVICES=$DB GSPLAT_TT_DEVICE_ID=0); else envs+=(GSPLAT_TT_DEVICE_ID=$DB); fi ;;
    B) envs+=(GSPLAT_TT_AFR=1 GSPLAT_TT_AFR_DEVICES=$DA,$DB)
       [ $VIS = 1 ] || envs+=(GSPLAT_TT_AFR_VISIBLE=0) ;;
  esac
  echo "=== $tag arm=$arm $(cut -c1-8 SHA) vis=$VIS $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${envs[@]}" timeout "$to" python3 render/run.py --no-ref "$@" > "$S/run-$tag.log" 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "$pat" "$S/run-$tag.log" | cut -c1-600 | head -24
  step_rc "$tag" $rc
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: stopping (no reset; see the log)"; exit 124; fi
  return $rc
}
# md5 <tag> <dump dir name>: per-view md5 list, checked against the golden for the run's grid
md5() {
  local d=tmp/$2
  if [ -d "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > "$S/md5-$1.txt"
    python3 opt/md5_golden.py "$S/md5-$1.txt" "$S/run-$1.log"; local g=$?
    echo "MD5 $1 views=$(wc -l < "$S/md5-$1.txt") list_md5=$(md5sum < "$S/md5-$1.txt" | cut -c1-8) golden_rc=$g"
    step_rc "md5-$1" $g
    rm -rf "$d"
  else echo "NO_DUMP_DIR $d"; step_rc "md5-$1" 7; fi
}

echo "=== t582 remote_drive $(cut -c1-8 SHA) on $(hostname) $(date)"
ls /dev/tenstorrent
timeout 120 tt-smi -ls > "$S/tt-smi-ls.txt" 2>&1; echo "tt-smi -ls rc=$?"; grep -iE 'p150|board' "$S/tt-smi-ls.txt" | head -12
rm -rf "$C" tmp/t582-*

# Correctness: A (latency mode, compiles every kernel into $C), then B (2 chips, back-to-back).
one dumpA A 900 --iter-dir t582-dumpA --dump-views t582-dumpA-v; md5 dumpA t582-dumpA-v
pre=$first
one dumpB B 900 --back-to-back --iter-dir t582-dumpB --dump-views t582-dumpB-v; rcB=$?
if [ $rcB != 0 ] && ! grep -q '^B2B ' "$S/run-dumpB.log"; then
  # TT_VISIBLE_DEVICES may not select the chip in this tt-metal: the per-chip id path instead.
  echo "=== dumpB failed (rc $rcB): retrying with GSPLAT_TT_AFR_VISIBLE=0"
  VIS=0; first=$pre; rm -rf tmp/t582-dumpB tmp/t582-dumpB-v
  one dumpB-vis0 B 900 --back-to-back --iter-dir t582-dumpB --dump-views t582-dumpB-v; rcB=$?
  cp "$S/run-dumpB-vis0.log" "$S/run-dumpB.log"
fi
md5 dumpB t582-dumpB-v
cp tmp/t582-dumpA/hero_clean.png "$S/hero-1chip.png" 2>/dev/null
cp tmp/t582-dumpB/hero_clean.png "$S/hero-afr-chipA.png" 2>/dev/null
cp tmp/t582-dumpB/afr1/hero_clean.png "$S/hero-afr-chipB.png" 2>/dev/null
ls -la "$S"/hero-*.png
[ "$first" = 0 ] || { echo "=== correctness failed (first rc $first): no timing runs"; exit "$first"; }

# Timing: warm-up round (discarded), then 3 alternating rounds.
one warm-A A 420 --back-to-back --iter-dir t582-b2b
one warm-B B 600 --back-to-back --iter-dir t582-afr
for r in 1 2 3; do
  case $r in 2) arms="B A C" ;; *) arms="A B C" ;; esac
  for a in $arms; do
    case $a in
      A) one r$r-A A 420 --back-to-back --iter-dir t582-b2b ;;
      B) one r$r-B B 600 --back-to-back --iter-dir t582-afr ;;
      C) one r$r-C C 420 --back-to-back --iter-dir t582-b2bC ;;
    esac
  done
done
# Latency mode (secondary; no --dump-views): one chip. AFR's latency per view is the single chip's.
for r in 1 2 3; do one lat$r-A A 420 --iter-dir t582-lat; done
for f in "$S"/run-*.log; do
  echo "$(basename "$f" .log) $(grep -oE 'ms_frame_median=[0-9.]+|afr_chip_ms_own_view=[0-9.,]+|afr_hero_equal=[a-zA-Z]+|raw_md5=[0-9a-f]+' "$f" | tr '\n' ' ')$(grep -oE 'TTW_TIMING ms_view=[0-9.]+' "$f" | tr '\n' ' ')"
done > "$S/summary.txt"
cat "$S/summary.txt"
