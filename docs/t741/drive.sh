#!/bin/bash
# t741: device A/B of the fp32 op/colour side bucket (GSPLAT_TT_TRAIN_OPC_SIDE, #741)
# (copy of docs/t734/drive.sh). Arms (docs/t741/remote.sh): on (default), aa (A/A of on),
# rec32off (GSPLAT_TT_PFWC_REC32=0), side (rec32off + GSPLAT_TT_TRAIN_OPC_SIDE=1),
# dump/dumpr32/dumpside (fp32 train buffer of view 0).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T (opt/sync_remote.sh),
# the dumps first (compile check; stops on failure), device hero + 30-view md5 sweep for side
# (opt/ttw/screenshot.sh), warm-up, 4 b2b rounds per arm of orbit (cameras_v2, 30 views) with the
# order alternated, per-arm JIT listings.
#   ttp detach t741-bench -- docs/t741/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/t741; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t741-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t741-bin/$c
  chmod +x tmp/t741-bin/$c
done
export PATH=$PWD/tmp/t741-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" @p100 || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t741 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t741 $sha opc side bucket A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t741 /localdev/smarton/.cache/ttmc-t741-* $T/tmp/ttm-eth12-cache-t741-*; mkdir -p $T/tmp/t741"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t741-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t741-remote.sh.tmp $T/tmp/t741-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t741-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t741-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# PHASE=diag (after the first A/B): does the side read replace the gather? GSPLAT_TT_T727_DIAG=2
# drops the per-record gather (fallback), so sided2 keeps md5 3fada021 only if every subchunk
# took the side read; r32d2 is the control (gather dropped, no side: output wrong) and the bound.
if [ "${PHASE:-}" = diag ]; then
  step d0 400 0 dumpsided2 dumpr32d2
  $SSH "$H" "cd $T/tmp/t741 && md5sum train-v0-*.bin > dump-md5.txt; cat dump-md5.txt"
  step d1 500 1 side sided2 rec32off r32d2
  step d2 500 2 r32d2 rec32off sided2 side
  scp -q -o BatchMode=yes "$H:$T/tmp/t741/*.*" "$O/" || rc=4
  echo "=== drive t741 diag done rc=$rc $(TZ=America/Los_Angeles date)"; exit $rc
fi
# Dumps first: they compile the side kernels; stop here if they fail.
step u0 400 0 dump dumpr32 dumpside
$SSH "$H" "cd $T/tmp/t741 && md5sum train-v0-on.bin train-v0-r32.bin train-v0-side.bin > dump-md5.txt; cat dump-md5.txt"
if [ $rc != 0 ] || [ "${DUMP_ONLY:-0}" = 1 ]; then
  scp -q -o BatchMode=yes "$H:$T/tmp/t741/*.*" "$O/"
  echo "=== drive t741 stopped after dumps rc=$rc $(TZ=America/Los_Angeles date)"; exit $rc
fi
# Device hero + 30-view md5 sweep for side (diff + PSNR vs benchmarks/reference_v2/hero.png).
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shots skipped (SKIP_SHOT=1)"; else
for st in side; do
  echo "=== shot $st $(TZ=America/Los_Angeles date +%T)"
  ev=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_PFWC_REC32=0 GSPLAT_TT_TRAIN_OPC_SIDE=1)
  H=$H T=$T NO_SYNC=1 NAME=t741-$st SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 741 "$rev" --locked ${ev[@]+"${ev[@]}"} > "$O/shot-$st.out" 2>&1; s=$?
  grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot-$st.out"; echo "shot $st rc=$s"
  [ $s = 0 ] || rc=5
done
fi
# Warm-up (fills each arm's JIT cache), 4 b2b rounds per arm with the order alternated.
# Each devrun step stays under 500 s.
step w0 500 0 on aa rec32off side
step r1 500 1 on side aa rec32off
step r2 500 2 rec32off aa side on
step r3 500 3 side on rec32off aa
step r4 500 4 aa rec32off on side
# Per-arm JIT listings, for
# python3 opt/ab_jit_check.py --arm on=$O/jit-on.txt --arm side=$O/jit-side.txt \
#   --kernel reader_alpha_blend_mb_devcull --kernel sort_bin_onelaunch --kernel sort_ol_town_compute
for a in on aa rec32off side; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t741-$a /localdev/smarton/.cache/ttmc-t741-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t741/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t741/jit-$a.txt"
done
scp -q -o BatchMode=yes "$H:$T/tmp/t741/*.*" "$O/" || rc=4
$SSH "$H" "du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t741 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
