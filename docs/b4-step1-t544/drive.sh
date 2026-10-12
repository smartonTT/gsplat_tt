#!/bin/bash
# t544 (B4' step 1): one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the
# measurement box, the kcfg probes (remote_kcfg.sh a, b: fused mat+blend program size from the
# kcfg TT_FATAL, per-RISC ELF sizes), the device hero of the default arm (opt/ttw/screenshot.sh:
# 30-view md5 sweep, hero + x10 diff + PSNR vs benchmarks/reference_v2/hero.png), a discarded
# warm-up round, then 3 alternating b2b rounds of def (GSPLAT_TT_BLEND_ALIAS_CUT unset = cut)
# / off (=0, the best-iter-225 layout).
#   ttp detach t544 -- docs/b4-step1-t544/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Each remote step is its own short devrun ssh session (<= 600 s), so an ssh drop loses one step only.
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4-step1-t544; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t544-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t544-bin/$c
  chmod +x tmp/t544-bin/$c
done
export PATH=$PWD/tmp/t544-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t544 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t544 $sha blend alias fifo cut ($H)"
$SSH "$H" "rm -rf $T/tmp/t544; mkdir -p $T/tmp/t544"
pat="^(===|run rc|KCFG|ELF|B2B |SUMMARY|HANG|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name script args...
  local n=$1 s=$2; shift 2
  $DEVRUN --host "$H" --no-verify --timeout 590 --tag t544-$n -- \
    "T=$T bash $T/$D/$s $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local r=${PIPESTATUS[0]}; [ "$r" = 0 ] || { echo "=== step $n rc=$r"; rc=$r; }
  [ "$r" = 124 ] && exit 124
}
step kcfg-a remote_kcfg.sh a
step kcfg-b remote_kcfg.sh b
echo "=== shot $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t544 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 226 "$rev" --locked \
  > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
step warm remote_ab.sh 0 b2b off def
for r in 1 2 3; do
  case $r in 2) arms="def off" ;; *) arms="off def" ;; esac
  step r$r-b2b remote_ab.sh $r b2b $arms
done
scp -q -o BatchMode=yes "$H:$T/tmp/t544/*.log" "$H:$T/tmp/t544/elfsize.txt" "$O/" || rc=4
echo "=== drive t544 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
