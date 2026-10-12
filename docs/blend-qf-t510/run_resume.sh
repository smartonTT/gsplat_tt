#!/bin/bash
# t510 stage 0 resume after the bh-33 reboot: the views from slice index $4 (default 7 = orb_06) on,
# into the same out dir; finished views (rows in views.log, PNGs) are kept. Run from the staged dir:
#   QF_ROOT=$PWD nice -n 19 ionice -c3 bash docs/blend-qf-t510/run_resume.sh <out> <python> <procs> [from]
set -eo pipefail
O=${1:?out}; PY=${2:?python}; P=${3:-12}; A=${4:-7}
trap 'echo $? > "$O/DONE"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OK=$(sed -n 's/^views variants //p' "$O/steps.log" | tail -1)
echo "resume $A:30 variants $OK" >> "$O/steps.log"
$PY docs/blend-qf-t510/qfmodel.py --views "$A:30" --out "$O" --variants "$OK" --threads 4 --procs "$P" > "$O/views2.log" 2>&1
echo "resume rc $?" >> "$O/steps.log"
