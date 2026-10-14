#!/bin/bash
# t724: train-mode fp32 op/colour A/B (copy of docs/t708/drive.sh; arm on = GSPLAT_TT_BLEND_TRAIN=1 with the #717 fp32 records).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T (opt/sync_remote.sh),
# device hero + 30-view md5 sweep for flag off and on (opt/ttw/screenshot.sh), warm-up, 4 b2b
# rounds per arm of orbit (cameras_v2, 30 views) with the order alternated, the fp32 train-buffer
# dump of view 0, per-arm JIT listings.
#   ttp detach t724-bench -- docs/t724/drive.sh <rev>   (Mac, worktree root)
# Fp32 identity: the dump step writes view 0's fp32 train buffer with DA 2 and DA 0; md5s compared.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/t724; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t724-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t724-bin/$c
  chmod +x tmp/t724-bin/$c
done
export PATH=$PWD/tmp/t724-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" @p100 || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t724 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t724 $sha fp32 decode-ahead train mode A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t724 /localdev/smarton/.cache/ttmc-t724-* $T/tmp/ttm-eth12-cache-t724-*; mkdir -p $T/tmp/t724"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t724-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t724-remote.sh.tmp $T/tmp/t724-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t724-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t724-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# Device hero + 30-view md5 sweep for both flag states (diff + PSNR vs benchmarks/reference_v2/hero.png).
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shots skipped (SKIP_SHOT=1)"; else
for st in off on; do
  echo "=== shot $st $(TZ=America/Los_Angeles date +%T)"
  ev=(); [ $st = on ] && ev=(GSPLAT_TT_BLEND_TRAIN=1)
  H=$H T=$T NO_SYNC=1 NAME=t724-$st SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 724 "$rev" --locked ${ev[@]+"${ev[@]}"} > "$O/shot-$st.out" 2>&1; s=$?
  grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot-$st.out"; echo "shot $st rc=$s"
  [ $s = 0 ] || rc=5
done
fi
# Warm-up (fills both arms' JIT caches), 4 b2b rounds per arm with the order alternated,
# then the fp32 train-buffer dump of view 0. Each devrun step stays under 500 s.
step w0 500 0 off on0 on
step r1 500 1 off on0 on
step r2 500 2 on on0 off
step r3 500 3 on0 off on
step r4 500 4 on off on0
step d1 300 5 dump dump0
# Per-arm JIT listings (p150a: eth dispatch caches under $T/tmp/ttm-eth12-cache-t724-<arm>), for
# python3 opt/ab_jit_check.py --arm off=$O/jit-off.txt --arm on=$O/jit-on.txt \
#   --kernel alpha_blend_compute_mb --kernel writer_alpha_blend   (tool on ttp/t656-...)
for a in off on0 on; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t724-$a /localdev/smarton/.cache/ttmc-t724-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t724/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t724/jit-$a.txt"
done
$SSH "$H" "cd $T/tmp/t724 && md5sum train-v0.bin train-v0-da0.bin > dump-md5.txt; cat dump-md5.txt"
scp -q -o BatchMode=yes "$H:$T/tmp/t724/*.*" "$O/" || rc=4
$SSH "$H" "du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t724 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
