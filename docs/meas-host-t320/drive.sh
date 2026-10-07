#!/bin/bash
# t320: full sync + build of the opt tip on the new measurement box, then the smoke
# run, all under one ttp lock p100 (Mac, repo root). Remote tree = the box's base
# dir, which other task dirs link .venv/scenes and copy _gsplat_cpu from.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=${H:-yyzo-bh-04}; T=/localdev/smarton/gstt2; O=docs/meas-host-t320/out
PLY=/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply; LMD5=$(md5 -q $PLY)
# bicycle.ply on the box was truncated (1374027776 of 1520726124 B): resume + verify md5.
ttp lock p100 -- bash -c "rsync -e 'ssh -o BatchMode=yes' --partial --append-verify $PLY $H:$T/scenes/bicycle.ply && \
  [ \"\$(ssh -o BatchMode=yes $H md5sum $T/scenes/bicycle.ply | cut -c1-32)\" = $LMD5 ] && echo PLY_MD5_OK && \
  opt/sync_remote.sh $H $T HEAD && \
  $DEVRUN --host $H --no-verify --timeout 600 --tag t320-smoke -- 'bash $T/docs/meas-host-t320/remote_smoke.sh'"
rc=$?; echo "DRIVE_RC=$rc"
mkdir -p $O
scp -q -o BatchMode=yes "$H:$T/tmp/t320/run.log" "$H:$T/tmp/t320/md5.txt" "$H:$T/tmp/t320/tt-smi.json" $O/ 2>/dev/null
exit $rc
