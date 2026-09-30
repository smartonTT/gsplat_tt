#!/bin/bash
# local driver: sync base + cand, build, dump, compare md5
set -e
cd /Users/smarton/dev/gsplat_tt/tt-project/worktrees/t61
S=docs/img-pack-t61
bash $S/sync_tree.sh t61base 469b812
bash $S/sync_tree.sh t61 HEAD
ssh -o BatchMode=yes yyzo-bh-07 "bash -s t61base" < $S/remote_build_dump.sh
ssh -o BatchMode=yes yyzo-bh-07 "bash -s t61" < $S/remote_build_dump.sh
ssh -o BatchMode=yes yyzo-bh-07 'diff /localdev/smarton/gstt2-t61base/tmp/t61-md5.txt /localdev/smarton/gstt2-t61/tmp/t61-md5.txt >/dev/null && echo ALL_VIEWS_IDENTICAL || { echo VIEWS_DIFFER; diff /localdev/smarton/gstt2-t61base/tmp/t61-md5.txt /localdev/smarton/gstt2-t61/tmp/t61-md5.txt | head; }'
