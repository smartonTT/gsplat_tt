#!/bin/bash
# t315: 3 alternating A/B rounds (default = TRISC_FILL on vs =0), then the device hero
# screenshot at HEAD with defaults. Each step takes its own ttp lock p100. Mac, repo root.
#   ttp detach t315-all -- docs/iter207-t315/run_all.sh
set -u
cd "$(git rev-parse --show-toplevel)"
docs/iter207-t315/drive.sh 1:base,off 2:off,base 3:base,off; d=$?
echo "DRIVE_RC=$d"
NAME=ttw-207 opt/ttw/screenshot.sh 207 HEAD; s=$?
echo "SHOT_RC=$s"
[ $d = 0 ] && [ $s = 0 ]
