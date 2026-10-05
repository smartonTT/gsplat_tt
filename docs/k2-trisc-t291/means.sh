#!/bin/bash
# t291: mean of the 30 per-view times per run log on the device host. means.sh <glob>
ssh -o BatchMode=yes yyzo-bh-07 "cd /localdev/smarton/gstt2-t291/tmp/t291; for f in \$(ls ${1:-run-*.log}); do printf '%s ' \$f; grep -oE 'view=[^ ]+ [0-9.]+ms' \$f | awk '{sub(\"ms\",\"\",\$2); s+=\$2; n++} END {printf \"%.3f n=%d\n\", s/n, n}'; done"
