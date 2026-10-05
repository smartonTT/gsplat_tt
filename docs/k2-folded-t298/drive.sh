#!/bin/bash
# t298: build, smoke (gen on), then 3 alternating rounds folded / off. Stops on a hang.
#   drive.sh <first_round>   (Mac, repo root; via ttp detach)
set -u
cd "$(git rev-parse --show-toplevel)"
D=docs/k2-folded-t298; r0=${1:-1}
echo "=== t298 drive $(git rev-parse --short HEAD) $(date +%T)"
bash $D/build.sh || { echo "BUILD_FAIL"; exit 1; }
for i in 0 1 2 3; do
  r=$((r0 + i))
  if [ $((i % 2)) = 0 ]; then arms="fold off"; else arms="off fold"; fi
  for a in $arms; do
    if [ $a = fold ]; then bash $D/run.sh $r fold; else bash $D/run.sh $r off GSPLAT_TT_K2_FOLDED=0; fi
    rc=$?
    if [ $rc != 0 ]; then echo "STOP r$r-$a rc=$rc"; exit $rc; fi
  done
done
echo "=== t298 drive done $(date +%T)"
