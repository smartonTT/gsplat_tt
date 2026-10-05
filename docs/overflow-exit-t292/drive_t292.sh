#!/bin/sh
# Mac side: run under `ttp lock p100 -- sh docs/overflow-exit-t292/drive_t292.sh <rev>`.
set -e
cd "$(dirname "$0")/../.."
REV=${1:-HEAD}
/Users/smarton/dev/gsplat_tt/tt-project/harness/bin/ssh-preflight yyzo-bh-07
opt/sync_remote.sh yyzo-bh-07 /localdev/smarton/gstt2-t292 "$REV"
ssh -o BatchMode=yes yyzo-bh-07 'test -e /localdev/smarton/gstt2-t290/render/render_clean.cpython-310-x86_64-linux-gnu.so && cut -c1-7 /localdev/smarton/gstt2-t290/SHA'
scp -o BatchMode=yes docs/overflow-exit-t292/remote_t292.sh yyzo-bh-07:/localdev/smarton/gstt2-t292/remote_t292.sh
ssh -o BatchMode=yes yyzo-bh-07 'bash /localdev/smarton/gstt2-t292/remote_t292.sh'
