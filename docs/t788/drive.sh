#!/bin/bash
# t788: train-forward blend levers (exact T early-out, fused program, SFPSWAP clamps), each behind
# an env knob (docs/t788/remote.sh lists the arms), on the orbit and the first 30 dataset views.
# One `ttp lock p100` around: ssh preflight (@p100), sync + build of <rev> into $T (opt/sync_remote.sh,
# one-way, repo-scoped, this worktree), per-forward train-buffer hash run per arm
# (identity, fills each JIT cache), 4 b2b rounds per arm with the order rotated, per-arm JIT listings + opt/ab_jit_check.py.
#   ttp detach t788 -- docs/t788/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
export H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
export T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
D=docs/t788; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t788-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
[ -n "$H" ] || { echo "=== no p100 host in $TP/harness/project.json"; exit 2; }
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t788-bin/$c
  chmod +x tmp/t788-bin/$c
done
export PATH=$PWD/tmp/t788-bin:$PATH
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t788 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
$SSH "$H" "rm -rf $T/tmp/t788 $T/tmp/t788-r* /localdev/smarton/.cache/ttmc-t788-* $T/tmp/ttm-eth12-cache-t788-*; mkdir -p $T/tmp/t788"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t788-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t788-remote.sh.tmp $T/tmp/t788-remote.sh" || exit 3
pat="^(===|run rc|hashes|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t788-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t788-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# Identity first (latency mode, per-forward train-buffer hash; also compiles each cfg), then
# 4 timed b2b rounds per arm with the order rotated.
step h0a 540 0 troh tpoh fzoh
step h0b 540 0 mmoh alloh
step h0c 540 0 trnh tpnh fznh
step h0d 540 0 mmnh allnh
step r1a 600 1 tro tpo fzo mmo allo
step r1b 600 1 trn tpn fzn mmn alln
step r2a 600 2 allo mmo fzo tpo tro
step r2b 600 2 alln mmn fzn tpn trn
step r3a 600 3 fzo allo tro mmo tpo
step r3b 600 3 fzn alln trn mmn tpn
step r4a 600 4 mmo tpo allo tro fzo
step r4b 600 4 mmn tpn alln trn fzn
# Per-arm JIT listings, then the JIT check of each lever against the baseline.
for a in tr tp fz mm all; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t788-$a /localdev/smarton/.cache/ttmc-t788-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t788/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t788/jit-$a.txt"
done
scp -q -o BatchMode=yes "$H:$T/tmp/t788/*.*" "$O/" || rc=4
: > "$O/jit-check.txt"
for p in "tp alpha_blend_compute_mb" "mm alpha_blend_compute_mb" "fz alpha_blend_compute_mb" "all alpha_blend_compute_mb"; do
  set -- $p
  echo "--- tr vs $1 ($2)" >> "$O/jit-check.txt"
  python3 opt/ab_jit_check.py --arm "tr=$O/jit-tr.txt" --arm "$1=$O/jit-$1.txt" --kernel "$2" >> "$O/jit-check.txt" 2>&1
  echo "rc=$?" >> "$O/jit-check.txt"
done
for p in fz all; do  # fused arms run matblend_compute instead (#789 mode)
  echo "--- tr vs $p (--allow-kernel-swap alpha_blend_compute_mb / matblend_compute)" >> "$O/jit-check.txt"
  python3 opt/ab_jit_check.py --allow-kernel-swap --arm "tr=$O/jit-tr.txt" --arm "$p=$O/jit-$p.txt" \
    --kernel alpha_blend_compute_mb --kernel matblend_compute >> "$O/jit-check.txt" 2>&1
  echo "rc=$?" >> "$O/jit-check.txt"
done
cat "$O/jit-check.txt"
$SSH "$H" "rm -rf $T/tmp/t788-r* $T/tmp/t788-remote.sh; du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t788 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
