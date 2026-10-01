#!/bin/bash
# Mac side: sync + build + A/B under the p100 lock. Usage: run_all.sh <rev>
set -euo pipefail
REV=${1:-HEAD}
opt/sync_remote.sh yyzo-bh-07 /localdev/smarton/gstt2-t111 "$REV"
ssh -o BatchMode=yes yyzo-bh-07 "bash /localdev/smarton/gstt2-t111/docs/blend-fpu-qf-t111/remote_ab.sh"
