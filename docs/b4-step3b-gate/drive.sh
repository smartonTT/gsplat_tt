#!/bin/bash
# t550 (B4' step 3b gate): one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the
# measurement box, the device hero with profiling OFF (opt/ttw/screenshot.sh: 30-view md5 sweep, hero +
# x10 diff + PSNR vs benchmarks/reference_v2/hero.png), one default b2b round (sanity), then Tracy
# captures in 3 chunks of 10 views for arm def (GSPLAT_TT_T2_PROF=1) and arm nt1 (+ town NT=1, emit
# counters), and the analysis (ana.py) into $O.
#   ttp detach t550 -- docs/b4-step3b-gate/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release), never the
# viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-t513} MESH_DEVICE=P100
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/b4-step3b-gate; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t550-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t550-bin/$c
  chmod +x tmp/t550-bin/$c
done
export PATH=$PWD/tmp/t550-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t550 $sha on $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t550 $sha T2_PROF counters ($H)"
$SSH "$H" "rm -rf $T/tmp/t550; mkdir -p $T/tmp/t550"
pat="^(===|run rc|tracy |B2B |B2B_STAGES|SUMMARY|HANG|device csv|no device|Trace|TT_FATAL|TT_THROW|error:)"
rc=0
step() {  # name timeout script args...
  local n=$1 to=$2 s=$3; shift 3
  $DEVRUN --host "$H" --no-verify --timeout $to --tag t550-$n -- \
    "T=$T bash $T/$s $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local r=${PIPESTATUS[0]}; [ "$r" = 0 ] || { echo "=== step $n rc=$r"; rc=$r; }
  [ "$r" = 124 ] && exit 124
}
if [ "${SKIP_SHOT:-0}" != 1 ]; then  # SKIP_SHOT=1: rerun only the Tracy captures + analysis
echo "=== shot (profiling off) $(TZ=America/Los_Angeles date +%T)"
H=$H T=$T NO_SYNC=1 NAME=ttw-t550 SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 550 "$rev" --locked \
  > "$O/shot.out" 2>&1; s=$?
grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"; echo "shot rc=$s"
[ $s = 0 ] || rc=5
step b2b 590 docs/b4-step3a-t546/remote_ab.sh 1 b2b nt3
fi
for arm in ${ARMS:-def nt1}; do
  for a in 0 10 20; do step $arm-c$a 590 $D/remote_tracy.sh $arm $a; done
done
scp -q -o BatchMode=yes "$H:$T/tmp/t550/*.csv.gz" "$H:$T/tmp/t550/*.log" "$O/" || rc=4
python3 $D/ana.py --def $O/dev-def-c*.csv.gz --nt1 $O/dev-nt1-c*.csv.gz --out $O > "$O/ana.out" 2>&1 || { echo "ana failed"; tail -5 "$O/ana.out"; rc=8; }
echo "=== drive t550 done rc=$rc $(TZ=America/Los_Angeles date)"
exit $rc
