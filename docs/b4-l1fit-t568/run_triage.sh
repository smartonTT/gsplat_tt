#!/bin/bash
# t568: PLAN=triage of the fuse arm through t555's drive.sh (its own ttp lock p100, bh-33 measurement
# reservation only, StrictHostKeyChecking=yes shims).   ttp detach t568tri -- docs/b4-l1fit-t568/run_triage.sh <rev>
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
PLAN=triage ARM=fuse O=docs/b4-l1fit-t568/out-triage docs/b4-step3b-t555/drive.sh "${1:?rev}"
