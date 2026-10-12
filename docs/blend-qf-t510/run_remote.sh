#!/bin/bash
# t510 stage 0 on bh-33 (CPU only, no device): hero over all variants, then the 29 other views
# over base + every variant that held >=42.4 dB on the hero. Run from the staged dir:
#   QF_ROOT=$PWD nice -n 19 ionice -c3 bash docs/blend-qf-t510/run_remote.sh <out> <python> <procs>
set -eo pipefail
O=${1:?out}; PY=${2:?python}; P=${3:-12}
mkdir -p "$O"
trap 'echo $? > "$O/DONE"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=docs/blend-qf-t510/qfmodel.py
$PY $M --views hero --out "$O" --threads 4 --procs "$P" > "$O/hero.log" 2>&1
echo "hero rc $?" >> "$O/steps.log"
OK=$($PY -c "
import glob,json
r=[json.loads(l) for f in glob.glob('$O/rows-hero-*.jsonl') for l in open(f)]
print(','.join(['base']+[x['variant'] for x in r if x['variant']!='base' and x['psnr_vs_ref']>=42.4]))")
echo "views variants $OK" >> "$O/steps.log"
$PY $M --views 1:30 --out "$O" --variants "$OK" --threads 4 --procs "$P" > "$O/views.log" 2>&1
echo "views rc $?" >> "$O/steps.log"
