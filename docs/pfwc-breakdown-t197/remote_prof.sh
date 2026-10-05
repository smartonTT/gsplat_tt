#!/bin/bash
# t197 (remote, under ttp lock p100): pfwc per-step cycle split via DPRINT.
#   remote_prof.sh <stepcyc=1|2> <riscs...>   1 = per-step marks, 2 = also per-op split inside cov_cam
# One RISC per run: DPRINT on all 5 RISCs overflows the 70656 B kernel config buffer.
set -u; export LC_ALL=C
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t197; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t197; mkdir -p $S
L=${1:-1}; shift; for R in "${@:-TR1}"; do
O=$S/pc$L-$R.dprint; rm -f $O
echo "=== stepcyc=$L risc=$R $(cut -c1-7 SHA) $(date +%T)"
env TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t197-pc$L-$R GSPLAT_TT_PFWC_STEPCYC=$L \
  TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=$R TT_METAL_DPRINT_FILE=$O \
  timeout 500 python3 render/run.py --no-ref --view-range 0:2 --iter-dir t197-pc$L-$R > $S/pc$L-$R.log 2>&1
echo "rc=$?"; grep -E "^(STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|exceeds|too large" $S/pc$L-$R.log | head -6
echo "PC: $(grep -c ' PC ' $O) PO: $(grep -c ' PO ' $O) PR: $(grep -c ' PR ' $O) PW: $(grep -c ' PW ' $O)"
done
echo "=== done $(date +%T)"
