#!/bin/bash
# t708: forward training flag A/B on the p150 box (from docs/p150-rebaseline-t687/drive.sh).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T (opt/sync_remote.sh),
# device hero + 30-view md5 sweep for flag off and on (opt/ttw/screenshot.sh), warm-up, 4 b2b
# rounds per arm of orbit (cameras_v2, 30 views) with the order alternated, the fp32 train-buffer
# dump of view 0, per-arm JIT listings.
#   ttp detach t708-bench -- docs/t708/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/t708; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t708-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t708-bin/$c
  chmod +x tmp/t708-bin/$c
done
export PATH=$PWD/tmp/t708-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t708 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t708 $sha forward train flag A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t708 /localdev/smarton/.cache/ttmc-t708-* $T/tmp/ttm-eth12-cache-t708-*; mkdir -p $T/tmp/t708"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t708-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t708-remote.sh.tmp $T/tmp/t708-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t708-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t708-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# Device hero + 30-view md5 sweep for both flag states (diff + PSNR vs benchmarks/reference_v2/hero.png).
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shots skipped (SKIP_SHOT=1)"; else
for st in off on; do
  echo "=== shot $st $(TZ=America/Los_Angeles date +%T)"
  ev=(); [ $st = on ] && ev=(GSPLAT_TT_BLEND_TRAIN=1)
  H=$H T=$T NO_SYNC=1 NAME=t708-$st SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 708 "$rev" --locked ${ev[@]+"${ev[@]}"} > "$O/shot-$st.out" 2>&1; s=$?
  grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot-$st.out"; echo "shot $st rc=$s"
  [ $s = 0 ] || rc=5
done
fi
# Warm-up (fills both arms' JIT caches), 4 b2b rounds per arm with the order alternated,
# then the fp32 train-buffer dump of view 0. Each devrun step stays under 500 s.
step w0 500 0 off on
step r1 500 1 off on
step r2 500 2 on off
step r3 500 3 off on
step r4 500 4 on off
step d1 300 5 dump
# Per-arm JIT listings (p150a: eth dispatch caches under $T/tmp/ttm-eth12-cache-t708-<arm>), for
# python3 opt/ab_jit_check.py --arm off=$O/jit-off.txt --arm on=$O/jit-on.txt \
#   --kernel alpha_blend_compute_mb --kernel writer_alpha_blend   (tool on ttp/t656-...)
for a in off on; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t708-$a /localdev/smarton/.cache/ttmc-t708-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t708/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t708/jit-$a.txt"
done
scp -q -o BatchMode=yes "$H:$T/tmp/t708/*.*" "$O/" || rc=4
$SSH "$H" "du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t708 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
