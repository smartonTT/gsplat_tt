#!/bin/bash
# t490 (review of #489): one `ttp lock p100` around: ssh preflight, sync + build of <rev> (the
# merged landing tip) on the measurement box, a discarded warm-up, 5 alternating rounds of b2b +
# latency for off (GSPLAT_TT_PFWC_ACQ_FUSE=0 GSPLAT_TT_PFWC_SKIP_RGB=0, the pre-#489 default) /
# def (both unset: ACQ_FUSE=3, SKIP_RGB=1), then one --dump-views md5 pass per arm and the device hero.
#   ttp detach t490 -- docs/land-t490/drive.sh <rev>   (Mac, worktree root; O=<out dir>)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t469}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/land-t490; O=${O:-$D/out}; mkdir -p "$O" tmp/ttw-state tmp/t490-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t490-bin/$c
  chmod +x tmp/t490-bin/$c
done
export PATH=$PWD/tmp/t490-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t490 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t490 $sha land #489 b2b A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t490; mkdir -p $T/tmp/t490"
pat="^(===|run rc|B2B |SUMMARY|MD5_GOLDEN|HANG|NO_DUMP|Trace|TT_FATAL|TT_THROW)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 560 --tag t490-$n -- \
    "T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
step warm 0 b2b off def
for r in 1 2 3 4 5; do
  case $r in 2|4) arms="def off" ;; *) arms="off def" ;; esac
  step r$r-b2b $r b2b $arms
  step r$r-lat $r lat $arms
done
step dump 9 dump off def
scp -q -o BatchMode=yes "$H:$T/tmp/t490/run-*.log" "$H:$T/tmp/t490/md5-*.txt" "$O/" || rc=4
scp -q -o BatchMode=yes "$H:$T/tmp/t490-r9-dump-def/hero_clean.png" "$O/hero-def.png" || rc=4
echo "=== drive t490 done rc=$rc $(date)"
exit $rc
