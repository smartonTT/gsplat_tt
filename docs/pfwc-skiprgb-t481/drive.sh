#!/bin/bash
# t481: one `ttp lock p100` around: ssh preflight, sync + build of <rev> on the measurement box,
# a discarded warm-up (JIT for both arms), 3 alternating rounds of b2b + latency for
# off (GSPLAT_TT_PFWC_SKIP_RGB=0) / on (=1), then one --dump-views md5 pass per arm.
#   ttp detach t481 -- docs/k2-b2b-t481/drive.sh <rev>   (Mac, worktree root; O=<out dir> ON=on|def)
# Restrictions: only the existing measurement reservation (no ird reserve/extend/release),
# never the viewer box; every ssh/scp goes through shims forcing StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
export H=${H:-yyzo-bh-04} T=${T:-/localdev/smarton/gstt2-t469}
DEVRUN=${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
D=docs/pfwc-skiprgb-t481; O=${O:-$D/out}; ON=${ON:-on}; mkdir -p "$O" tmp/ttw-state tmp/t481-bin
if [ "${2:-}" != --locked ]; then
  exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked
fi
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes "$@"\n' $c > tmp/t481-bin/$c
  chmod +x tmp/t481-bin/$c
done
export PATH=$PWD/tmp/t481-bin:$PATH
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH="ssh -o BatchMode=yes"
sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
  "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
export TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state
[ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
sha=$(git rev-parse --short "$rev")
echo "=== drive t481 $sha on $H:$T $(date)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
tail -2 "$O/sync.log"
~/dev/tt-workflows/scripts/buildid.sh stamp cpp "t481 $sha PFWC_SKIP_RGB b2b A/B ($H)"
$SSH "$H" "rm -rf $T/tmp/t481; mkdir -p $T/tmp/t481"
pat="^(===|run rc|B2B |SUMMARY|MD5_GOLDEN|HANG|NO_DUMP|Trace|TT_FATAL|TT_THROW)"
rc=0
step() {  # name round mode arms...
  local n=$1; shift
  $DEVRUN --host "$H" --no-verify --timeout 380 --tag t481-$n -- \
    "T=$T bash $T/$D/remote_ab.sh $*" 2>&1 | tee "$O/$n.out" | grep -E "$pat"
  local s=${PIPESTATUS[0]}; [ "$s" = 0 ] || { echo "=== step $n rc=$s"; rc=$s; }
  [ "$s" = 124 ] && exit 124
}
if [ "${MODE:-ab}" = confirm ]; then
  # Kept-default check on the landing commit: env unset (default on) vs =0, then a dump pass
  # (md5 + hero) for the default.
  step c-b2b 10 b2b def off
  step c-dump 11 dump def
  scp -q -o BatchMode=yes "$H:$T/tmp/t481/run-r1[01]-*.log" "$H:$T/tmp/t481/md5-r11-*.txt" "$O/" || rc=4
  scp -q -o BatchMode=yes "$H:$T/tmp/t481-r11-dump-def/hero_clean.png" "$O/hero-def.png" || rc=4
  echo "=== drive t481 confirm done rc=$rc $(date)"
  exit $rc
fi
step warm 0 b2b off $ON
for r in 1 2 3; do
  case $r in 2) arms="$ON off" ;; *) arms="off $ON" ;; esac
  step r$r-b2b $r b2b $arms
  step r$r-lat $r lat $arms
done
step dump 9 dump off $ON
scp -q -o BatchMode=yes "$H:$T/tmp/t481/run-*.log" "$H:$T/tmp/t481/md5-*.txt" "$O/" || rc=4
scp -q -o BatchMode=yes "$H:$T/tmp/t481-r9-dump-$ON/hero_clean.png" "$O/hero-$ON.png" || rc=4
echo "=== drive t481 done rc=$rc $(date)"
exit $rc
