#!/bin/bash
# t627: device half of the single-frame 2-chip split measurement (docs/split-frame-2chip/README.md).
# Runs ON the 2-board p150 host (bh-50, resource p150-2board) under `ttp detach --remote`, inside
# the Mac driver's `ttp lock p150-2board`. Tree $T (synced + built by opt/sync_remote.sh), JIT
# cache $C, outputs in $S. Arms (all render/run.py --no-ref, worker dispatch, same build):
#   A  one chip (id 0): --back-to-back, and latency mode (no --back-to-back, no --dump-views)
#   A1 chip 1 alone, --back-to-back (diagnostic: does chip 1 alone match chip 0's raw_md5?)
#   B  split latency: AFR 0,1 + SPLIT x at 16 + LOCKSTEP, --back-to-back (split_latency_ms)
#   C  split b2b: same without LOCKSTEP
#   D  AFR b2b: GSPLAT_TT_AFR=1 0,1 without SPLIT
# Order: correctness dumps (A latency --dump-views compiles every kernel; C --dump-views writes
# the joined frames + hero_clean.png), a discarded warm-up round, 3 rotated rounds A B C D, then
# 3 latency-mode runs of A. Every step's rc goes to $S/steps.txt; the first non-zero one, or 0,
# to $S/DONE on exit.
set -u
R=${R:-/localdev/smarton/nosync/t582}   # outside the user's mutagen sync (docs/l2-2chip/drive.sh)
export TT_METAL_HOME=$R/tt-metal TT_METAL_RUNTIME_ROOT=$R/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P150 TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export GSPLAT_TT_DISPATCH=worker
unset GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_AFR GSPLAT_TT_AFR_DEVICES GSPLAT_TT_AFR_VISIBLE \
  TT_VISIBLE_DEVICES GSPLAT_TT_DEVICE_ID GSPLAT_TT_SPLIT GSPLAT_TT_SPLIT_AXIS GSPLAT_TT_SPLIT_AT \
  GSPLAT_TT_SPLIT_LOCKSTEP GSPLAT_TT_SPLIT_WINDOW
T=${T:-$R/gstt2}; S=$T/tmp/t627; C=$R/.cache/ttmc-t627
# The container shows two boards; UMD numbers them 0 and 1 (TT_VISIBLE_DEVICES takes these ids).
DA=${DA:-0}; DB=${DB:-1}
ONLY=${ONLY:-all}   # all | check (correctness dumps only)
mkdir -p "$S"; rm -f "$S/DONE" "$S/steps.txt"
first=0
trap 'echo "$first" > "$S/DONE"; echo "=== remote_drive done first_rc=$first $(date +%T)"' EXIT
cd "$T" || { first=1; exit 1; }
source .venv/bin/activate || { first=1; exit 1; }

pat="^(B2B |SUMMARY|\[DEV\]|\[afr[0-9]\] \[DEV\]|\[afr\] |\[run\] AFR|BRISC_STACK_FAIL)|TTW_TIMING|Traceback|TT_THROW|TT_FATAL|error:|Error"
step_rc() {  # step_rc <tag> <rc>
  echo "$1 rc=$2" >> "$S/steps.txt"
  [ "$2" = 0 ] || [ "$first" != 0 ] || first=$2
}
SPLIT=(GSPLAT_TT_AFR=1 GSPLAT_TT_AFR_DEVICES=$DA,$DB GSPLAT_TT_SPLIT=1 GSPLAT_TT_SPLIT_AXIS=x GSPLAT_TT_SPLIT_AT=16)
# one <tag> <arm> <timeout s> <run.py args...>
one() {
  local tag=$1 arm=$2 to=$3; shift 3
  local -a envs=(TT_METAL_CACHE_RENDER=$C)
  case $arm in
    A)  envs+=(TT_VISIBLE_DEVICES=$DA GSPLAT_TT_DEVICE_ID=0) ;;
    A1) envs+=(TT_VISIBLE_DEVICES=$DB GSPLAT_TT_DEVICE_ID=0) ;;
    B)  envs+=("${SPLIT[@]}" GSPLAT_TT_SPLIT_LOCKSTEP=1) ;;
    C)  envs+=("${SPLIT[@]}") ;;
    D)  envs+=(GSPLAT_TT_AFR=1 GSPLAT_TT_AFR_DEVICES=$DA,$DB) ;;
  esac
  echo "=== $tag arm=$arm $(cut -c1-8 SHA) $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${envs[@]}" timeout "$to" python3 render/run.py --no-ref "$@" > "$S/run-$tag.log" 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -E "$pat" "$S/run-$tag.log" | cut -c1-700 | head -30
  step_rc "$tag" $rc
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: stopping (no reset; see the log)"; exit 124; fi
  return $rc
}

echo "=== t627 remote_drive $(cut -c1-8 SHA) on $(hostname) $(date)"
ls /dev/tenstorrent
timeout 120 tt-smi -ls > "$S/tt-smi-ls.txt" 2>&1; echo "tt-smi -ls rc=$?"
rm -rf "$C" tmp/t627-*

# Correctness. A latency mode compiles every kernel into $C and writes its hero.
one dumpA A 900 --iter-dir t627-dumpA --dump-views t627-dumpA-v
cp tmp/t627-dumpA/hero_clean.png "$S/hero-1chip.png" 2>/dev/null
one chkA A 600 --back-to-back --iter-dir t627-chkA
cp tmp/t627-chkA/hero_clean.png "$S/hero-1chip-b2b.png" 2>/dev/null
one chkA1 A1 600 --back-to-back --iter-dir t627-chkA1   # diagnostic only
cp tmp/t627-chkA1/hero_clean.png "$S/hero-chip1.png" 2>/dev/null
pre=$first
one dumpC C 900 --back-to-back --iter-dir t627-dumpC --dump-views t627-dumpC-v
cp tmp/t627-dumpC/hero_clean.png "$S/hero-split.png" 2>/dev/null
for d in t627-dumpA-v t627-dumpC-v; do
  [ -d tmp/$d ] && (cd tmp/$d && md5sum * | sort -k2) > "$S/md5-$d.txt"
done
ls -la "$S"/hero-*.png
# A1 (chip 1 alone) is diagnostic: only A and the split dump gate the timing runs.
bad=$(grep -E '^(dumpA|chkA|dumpC) rc=[^0]' "$S/steps.txt")
[ -z "$bad" ] || { echo "=== correctness step failed: $bad"; exit "$first"; }
[ "$ONLY" = all ] || exit "$first"

# Timing: warm-up round (discarded), then 3 rotated rounds.
one warm-A A 600 --back-to-back --iter-dir t627-b2bA
one warm-B B 900 --back-to-back --iter-dir t627-splitlat
one warm-C C 900 --back-to-back --iter-dir t627-split
one warm-D D 900 --back-to-back --iter-dir t627-afr
for r in 1 2 3; do
  case $r in 1) arms="A B C D" ;; 2) arms="D C B A" ;; *) arms="B A D C" ;; esac
  for a in $arms; do
    case $a in
      A) one r$r-A A 600 --back-to-back --iter-dir t627-b2bA ;;
      B) one r$r-B B 900 --back-to-back --iter-dir t627-splitlat ;;
      C) one r$r-C C 900 --back-to-back --iter-dir t627-split ;;
      D) one r$r-D D 900 --back-to-back --iter-dir t627-afr ;;
    esac
  done
done
# Latency mode of one chip (no --back-to-back, no --dump-views).
for r in 1 2 3; do one lat$r-A A 600 --iter-dir t627-lat; done
for f in "$S"/run-*.log; do
  echo "$(basename "$f" .log) $(grep -oE 'ms_frame_median=[0-9.]+|afr_chip_ms_own_view=[0-9.,]+|split_copy_ms=[0-9.]+|split_latency_ms(_median)?=[0-9.]+|raw_md5=[0-9a-f]+' "$f" | tr '\n' ' ')$(grep -oE 'TTW_TIMING [a-z_]+=[0-9.]+' "$f" | tr '\n' ' ')"
done > "$S/summary.txt"
cat "$S/summary.txt"
