#!/bin/bash
# t573: SIB_DBG arms through t555's drive.sh (its own ttp lock p100, bh-33 measurement reservation
# only, StrictHostKeyChecking=yes shims).   ttp detach t573arms -- docs/b4-l1fit-t573/run_arms.sh <rev> "<arms>"
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
PLAN=arms DBG_ARMS="${2:?arms}" TRI_SH=../b4-l1fit-t573/remote_triage.sh O=${O:-docs/b4-l1fit-t573/out-arms} \
  docs/b4-step3b-t555/drive.sh "${1:?rev}"
