#!/bin/bash
# t778: split the train-flag forward cost (+5.52 ms/view, #712) by env-knob arms, no code change
# (docs/t778/remote.sh lists the arms), plus the train forward on 30 dataset views (b2b and per view).
# One `ttp lock p100` around: ssh preflight (@p100), sync + build of <rev> into $T (opt/sync_remote.sh,
# one-way, repo-scoped, this worktree), unscored warm-up (fills each arm's JIT cache), 4 b2b rounds
# per arm with the order rotated, one per-view latency run, per-arm JIT listings + opt/ab_jit_check.py.
#   ttp detach t778 -- docs/t778/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box bh-30; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
TP=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}
export H=${H:-$(python3 -c "import json;print(json.load(open('$TP/harness/project.json'))['device']['hosts']['p100'])")}
export T=${T:-/localdev/smarton/gstt2-sync} MESH_DEVICE=${MESH_DEVICE:-P150}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
D=docs/t778; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t778-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
[ -n "$H" ] || { echo "=== no p100 host in $TP/harness/project.json"; exit 2; }
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t778-bin/$c
  chmod +x tmp/t778-bin/$c
done
export PATH=$PWD/tmp/t778-bin:$PATH
"$TP/harness/bin/ssh-preflight" @p100 || { p=$?; echo "=== ssh preflight @p100 ($H) failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t778 $sha on $H:$T $(TZ=America/Los_Angeles date)"
$SSH "$H" "tt-smi -ls 2>&1 | head -20" > "$O/tt-smi.txt" 2>&1; grep -iE 'p150|p100|blackhole' "$O/tt-smi.txt" | head -4
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
$SSH "$H" "rm -rf $T/tmp/t778 $T/tmp/t778-r* /localdev/smarton/.cache/ttmc-t778-* $T/tmp/ttm-eth12-cache-t778-*; mkdir -p $T/tmp/t778"
scp -q -o BatchMode=yes "$D/remote.sh" "$H:$T/tmp/t778-remote.sh.tmp" && $SSH "$H" "mv $T/tmp/t778-remote.sh.tmp $T/tmp/t778-remote.sh" || exit 3
pat="^(===|run rc|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout round arms...
  local n=$1 to=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout "$to" --tag t778-$n -- \
    "T=$T MESH_DEVICE=$MESH_DEVICE RUN_TIMEOUT=240 bash $T/tmp/t778-remote.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat" | cut -c1-200
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
# Warm-up (unscored; compiles every arm's kernels), then 4 rounds with the order rotated.
step w0a 900 0 rnd unf s0 tp0
step w0b 900 0 tr rndnn trnn
step r1a 600 1 rnd unf s0 tp0
step r1b 600 1 tr rndnn trnn
step r2a 600 2 trnn tr tp0
step r2b 600 2 s0 unf rnd rndnn
step r3a 600 3 tp0 s0 unf
step r3b 600 3 rnd rndnn trnn tr
step r4a 600 4 tr trnn rndnn rnd
step r4b 600 4 unf s0 tp0
step pv 600 5 pvnn
# Per-arm JIT listings, then the JIT check on each adjacent pair of the ladder.
for a in rnd unf s0 tp0 tr; do
  $SSH "$H" "find $T/tmp/ttm-eth12-cache-t778-$a /localdev/smarton/.cache/ttmc-t778-$a -maxdepth 8 -path '*/kernels/*' -type d > $T/tmp/t778/jit-$a.txt 2>/dev/null; wc -l < $T/tmp/t778/jit-$a.txt"
done
scp -q -o BatchMode=yes "$H:$T/tmp/t778/*.*" "$O/" || rc=4
: > "$O/jit-check.txt"
for p in "rnd unf matblend_compute" "unf s0 alpha_blend_compute_mb" "s0 tp0 alpha_blend_compute_mb" "tp0 tr alpha_blend_compute_mb" "rnd tr alpha_blend_compute_mb"; do
  set -- $p
  echo "--- $1 vs $2 ($3)" >> "$O/jit-check.txt"
  python3 opt/ab_jit_check.py --arm "$1=$O/jit-$1.txt" --arm "$2=$O/jit-$2.txt" --kernel "$3" >> "$O/jit-check.txt" 2>&1
  echo "rc=$?" >> "$O/jit-check.txt"
done
cat "$O/jit-check.txt"
$SSH "$H" "rm -rf $T/tmp/t778-r* $T/tmp/t778-remote.sh; du -sxh $T /localdev/smarton/.cache 2>/dev/null; df -h /localdev/smarton | tail -1" > "$O/disk.txt" 2>&1; cat "$O/disk.txt"
echo "=== drive t778 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
