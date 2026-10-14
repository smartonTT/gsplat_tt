#!/bin/bash
# t755: device A/B of the 3-slot op/colour ring (GSPLAT_TT_TRAIN_OPC_RING3, #755)
# (copy of docs/t755/drive.sh). Arms (docs/t755/remote.sh): on (default), aa (A/A of on),
# r3 (GSPLAT_TT_TRAIN_OPC_RING3=1), dump/dumpr3 (fp32 train buffer of view 0 + check_dump).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T (opt/sync_remote.sh),
# the dumps first (compile check; stops on failure), device hero + 30-view md5 sweep for r3
# (opt/ttw/screenshot.sh), warm-up, 4 b2b rounds per arm of orbit (cameras_v2, 30 views) with the
# order alternated, per-arm JIT listings.
#   ttp detach t755-bench -- docs/t755/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/t755; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t755-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t755-bin/$c
  chmod +x tmp/t755-bin/$c
done
export PATH=$PWD/tmp/t755-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" @p100 || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t755 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t755 $sha opc ring3 A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t755 /localdev/smarton/.cache/ttmc-t755-* $T/tmp/ttm-eth12-cache-t755-*; mkdir -p $T/tmp/t755"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t755-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t755-remote.sh.tmp $T/tmp/t755-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t755-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t755-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# Dumps first: they compile the ring3 kernels; stop here if they fail.
step u0 400 0 dump dumpr3
$SSH "$H" "cd $T/tmp/t755 && md5sum train-v0-on.bin train-v0-r3.bin > dump-md5.txt; cat dump-md5.txt"
if [ $rc != 0 ] || [ "${DUMP_ONLY:-0}" = 1 ]; then
  scp -q -o BatchMode=yes "$H:$T/tmp/t755/*.*" "$O/"
  echo "=== drive t755 stopped after dumps rc=$rc $(TZ=America/Los_Angeles date)"; exit $rc
fi
# Device hero + 30-view md5 sweep for r3 (diff + PSNR vs benchmarks/reference_v2/hero.png).
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shots skipped (SKIP_SHOT=1)"; else
for st in r3; do
  echo "=== shot $st $(TZ=America/Los_Angeles date +%T)"
  ev=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_OPC_RING3=1)
  H=$H T=$T NO_SYNC=1 NAME=t755-$st SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 755 "$rev" --locked ${ev[@]+"${ev[@]}"} > "$O/shot-$st.out" 2>&1; s=$?
  grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot-$st.out"; echo "shot $st rc=$s"
  [ $s = 0 ] || rc=5
done
fi
# Warm-up (fills each arm's JIT cache), 4 b2b rounds per arm with the order alternated.
# Each devrun step stays under 500 s.
step w0 500 0 on aa r3
step r1 500 1 on r3 aa
step r2 500 2 r3 aa on
step r3 500 3 aa on r3
step r4 500 4 r3 on aa
# Per-arm JIT listings, for
# python3 opt/ab_jit_check.py --arm on=$O/jit-on.txt --arm r3=$O/jit-r3.txt \
#   --kernel reader_alpha_blend_mb_devcull
for a in on aa r3; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t755-$a /localdev/smarton/.cache/ttmc-t755-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t755/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t755/jit-$a.txt"
done
scp -q -o BatchMode=yes "$H:$T/tmp/t755/*.*" "$O/" || rc=4
$SSH "$H" "du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t755 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
