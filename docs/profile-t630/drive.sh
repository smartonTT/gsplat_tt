#!/bin/bash
# t630 (Mac, worktree root): FOLD=1 fixed-cost attribution of best-iter-227 + the t630 counters.
# One `ttp lock p100` around: ssh preflight, sync + build of <rev>, then short devrun steps:
#   b2b r1 (untraced headline, counters compiled out), disasm (trisc1 SFPU count),
#   device-profiler chunks P0,P5,..,P25 (5 views each, FOLD=1 default) with GSPLAT_TT_BZ_PROF=1.
# The FOLD=1 traced program (106528 B with fz_*) exceeds the +32 KB profiler kcfg; +34 KB (105472 B)
# is the most the static CBs allow. Ladder: FZ_OFF at +34; else also MATCULL_DEPTH=1 (frees 16 KB of
# CBs) at +44.
#   ttp detach t630 -- docs/profile-t630/drive.sh <rev>
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/profile-t630; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t630-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t630-bin/$c
  chmod +x tmp/t630-bin/$c
done
export PATH=$PWD/tmp/t630-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t630 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t630 $sha FOLD=1 fixed-cost profile ($H)"
$SSH "$H" "mkdir -p $T/tmp/t630"
pat="^(===|VIEWS=|run rc|tracy |device csv|no device|B2B |SUMMARY|objdump=|disasm|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name bench-args...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t630-$n -- \
    "T=$T bash $T/$D/bench.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
  return 0
}
step b2b-r1 b2b 1
step disasm disasm
E="GSPLAT_TT_FZ_OFF=1 GSPLAT_TT_BZ_PROF=1 GSPLAT_TT_KCFG_EXTRA_KB=34"
step p0 tracy P0 0 5 $E
if ! grep -q "^device csv P0 rows" "$O/p0.out"; then
  E="GSPLAT_TT_FZ_OFF=1 GSPLAT_TT_BZ_PROF=1 GSPLAT_TT_MATCULL_DEPTH=1 GSPLAT_TT_KCFG_EXTRA_KB=44"
  echo "=== P0 failed; fallback env: $E"
  step p0b tracy P0 0 5 $E
fi
if grep -q "^device csv P0 rows" "$O"/p0*.out; then
  echo "=== profiler env: $E"
  for v in 5 10 15 20 25; do step p$v tracy P$v $v $((v + 5)) $E; done
fi
echo "$E" > "$O/prof-env.txt"
scp -q -o BatchMode=yes "$H:$T/tmp/t630/T-P*.log" "$H:$T/tmp/t630/dev-P*.csv.gz" \
  "$H:$T/tmp/t630/run-b2b-r1.log" "$H:$T/tmp/t630/matblend_trisc1.*" "$O/" || rc=4
echo "=== drive t630 done rc=$rc env=$E $(TZ=America/Los_Angeles date)"
exit $rc
