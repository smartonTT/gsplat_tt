#!/bin/bash
# t726: rerun the fix arm's T0 oracle (sh0, sh3) on bh-49 p150a, 110 cores, after the drive.sh cleanup fix.
#   ttp detach t726oracle -- docs/t726/oracle.sh <fix-rev>
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
fix=${1:?fix rev}
export H=bh-49 T2_CORES=110
O=docs/t726/out/fix-oracle; rm -rf "$O"
OUT=$O TTMC=ttmc-t726-fix RUNS="sh0 sh3" docs/t713/drive.sh "$fix"; echo "oracle rc=$?"
