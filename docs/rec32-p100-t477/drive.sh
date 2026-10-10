#!/bin/bash
# t477: p100 check of the REC32 default (best-iter-221). One `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# a discarded warm-up (JIT for both arms), 3 alternating rounds of b2b + latency for
# def (env unset = REC32 on) / off (GSPLAT_TT_PFWC_REC32=0), then one --dump-views md5 pass per arm.
#   ttp detach t477 -- docs/rec32-p100-t477/drive.sh <rev>   (Mac, worktree root)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t477}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/rec32-p100-t477; O=$D/out; mkdir -p "$O" tmp/ttw-state tmp/t477-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t477-bin/$c
  chmod +x tmp/t477-bin/$c
done
export PATH=$PWD/tmp/t477-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t477 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t477 $sha REC32 default p100 check ($H)"
$SSH "$H" "rm -rf $T/tmp/t477; mkdir -p $T/tmp/t477"
pat="^(===|run rc|B2B |SUMMARY|MD5_GOLDEN|HANG|NO_DUMP|Trace|TT_FATAL|TT_THROW)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t477-$n -- \
    "T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${MODE:-ab}" = confirm ]; then
  # Kept-default check on the landing commit: env unset (default on) vs =0, then a dump pass
  # (md5 + hero) for the default.
  step c-b2b 10 b2b def off
  step c-dump 11 dump def
  scp -q -o BatchMode=yes "$H:$T/tmp/t477/run-r1[01]-*.log" "$H:$T/tmp/t477/md5-r11-*.txt" "$O/" || rc=4
  scp -q -o BatchMode=yes "$H:$T/tmp/t477-r11-dump-def/hero_clean.png" "$O/hero-def.png" || rc=4
  echo "=== drive t477 confirm done rc=$rc $(date)"
  exit $rc
fi
step warm 0 b2b def off
for r in 1 2 3; do
  case $r in 2) arms="off def" ;; *) arms="def off" ;; esac
  step r$r-b2b $r b2b $arms
  step r$r-lat $r lat $arms
done
step dump 9 dump def off
scp -q -o BatchMode=yes "$H:$T/tmp/t477/run-*.log" "$H:$T/tmp/t477/md5-*.txt" "$O/" || rc=4
scp -q -o BatchMode=yes "$H:$T/tmp/t477-r9-dump-def/hero_clean.png" "$O/hero-def.png" || echo "=== no hero (not fatal)"
echo "=== drive t477 done rc=$rc $(date)"
exit $rc
