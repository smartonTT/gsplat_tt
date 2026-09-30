#!/bin/bash
# Verify (30-view md5) then, if identical, 3 interleaved A/B rounds. Run under `ttp lock p100 --`.
cd /Users/smarton/dev/gsplat_tt/tt-project/worktrees/t61
S=docs/img-pack-t61
bash $S/local_verify.sh 2>&1 | tee /dev/stderr | grep -q ALL_VIEWS_IDENTICAL || { echo "VERIFY_FAILED"; exit 2; }
ssh -o BatchMode=yes yyzo-bh-07 "bash -s t61base t61 3" < $S/remote_ab.sh
