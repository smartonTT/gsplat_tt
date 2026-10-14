#!/bin/bash
# t727: train-mode fp32 op/colour gather profile + A/B (copy of docs/t724/drive.sh).
# Arms (docs/t727/remote.sh): on (t724 default), d1/d2 (gather without DRAM reads / no gather, wrong
# output, attribution only), b2k (2048 reads per barrier), sd (skip dead records), sdb (both),
# prof/profsdb/profd2 (device-profiler zones), dump/dumpsdb (fp32 train buffer of view 0).
# One `ttp lock p100` around: ssh preflight, sync + build of <rev> into $T (opt/sync_remote.sh),
# device hero + 30-view md5 sweep for on and sdb (opt/ttw/screenshot.sh), warm-up, 4 b2b
# rounds per arm of orbit (cameras_v2, 30 views) with the order alternated, the fp32 train-buffer
# dump of view 0, per-arm JIT listings.
#   ttp detach t727-bench -- docs/t727/drive.sh <rev>   (Mac, worktree root)
# Fp32 identity: dump and dumpsdb write view 0's fp32 train buffer; md5s compared with 3fada021.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release here),
# never the viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/t727; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t727-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t727-bin/$c
  chmod +x tmp/t727-bin/$c
done
export PATH=$PWD/tmp/t727-bin:$PATH
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
"$PRE" @p100 || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t727 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t727 $sha train opc gather profile + A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t727 /localdev/smarton/.cache/ttmc-t727-* $T/tmp/ttm-eth12-cache-t727-*; mkdir -p $T/tmp/t727"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t727-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t727-remote.sh.tmp $T/tmp/t727-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t727-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t727-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# Device hero + 30-view md5 sweep for both flag states (diff + PSNR vs benchmarks/reference_v2/hero.png).
if [ "${SKIP_SHOT:-0}" = 1 ]; then echo "=== shots skipped (SKIP_SHOT=1)"; else
for st in on sdb; do
  echo "=== shot $st $(TZ=America/Los_Angeles date +%T)"
  ev=(GSPLAT_TT_BLEND_TRAIN=1); [ $st = sdb ] && ev+=(GSPLAT_TT_TRAIN_OPC_BATCH=2048 GSPLAT_TT_TRAIN_OPC_SKIPDEAD=1)
  H=$H T=$T NO_SYNC=1 NAME=t727-$st SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 727 "$rev" --locked ${ev[@]+"${ev[@]}"} > "$O/shot-$st.out" 2>&1; s=$?
  grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot-$st.out"; echo "shot $st rc=$s"
  [ $s = 0 ] || rc=5
done
fi
# Warm-up (fills both arms' JIT caches), 4 b2b rounds per arm with the order alternated,
# then the fp32 train-buffer dump of view 0. Each devrun step stays under 500 s.
step w0 500 0 on d1 d2
step w1 500 0 b2k sd sdb
step r1 500 1 on sdb b2k sd d1 d2
step r2 500 2 d2 sd b2k sdb on d1
step r3 500 3 sdb on sd b2k
step r4 500 4 sd b2k on sdb
step p1 500 5 prof profsdb profd2
step u1 300 6 dump dumpsdb
# Per-arm JIT listings (p150a: eth dispatch caches under $T/tmp/ttm-eth12-cache-t727-<arm>), for
# python3 opt/ab_jit_check.py --arm on=$O/jit-on.txt --arm sdb=$O/jit-sdb.txt \
#   --kernel alpha_blend_compute_mb --kernel writer_alpha_blend   (tool on ttp/t656-...)
for a in on sdb b2k sd; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t727-$a /localdev/smarton/.cache/ttmc-t727-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t727/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t727/jit-$a.txt"
done
$SSH "$H" "cd $T/tmp/t727 && md5sum train-v0-on.bin train-v0-sdb.bin > dump-md5.txt; cat dump-md5.txt"
scp -q -o BatchMode=yes "$H:$T/tmp/t727/*.*" "$O/" || rc=4
$SSH "$H" "du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t727 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
