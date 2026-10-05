#!/bin/bash
# t172 step 1 (remote, under ttp lock p100): Tracy views 0:2 with the per-tile fixed-cost
# sub-zones (GSPLAT_TT_BLEND_PROF=2), then the per-tile cycle split (GSPLAT_TT_MB_TILECYC=1,
# DPRINT on TRISC1) over the same 2 views.
set -u; export LC_ALL=C
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t172; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t172; mkdir -p $S
MODE=${1:-tracy+tc}   # tracy, tc or both (each devrun call must stay under the 600 s ceiling)
if [[ $MODE == *tracy* ]]; then
echo "=== tracy $(cut -c1-7 SHA) $(date +%T)"
env GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t172-prof GSPLAT_TT_BLEND_PROF=2 \
  timeout 400 bash opt/profiler/capture_tracy.sh t172-prof 0:2 > $S/prof-tracy.log 2>&1
echo "tracy rc=$?"; grep -E 'capture_tracy\] (OK|FAIL|DONE)|Traceback|overflow|dropped' $S/prof-tracy.log | head
fi
if [[ $MODE == *tc* ]]; then
echo "=== tilecyc $(date +%T)"
O=$S/tc.dprint; rm -f $O
env TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t172-tc GSPLAT_TT_MB_TILECYC=1 \
  TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=TR1 TT_METAL_DPRINT_FILE=$O \
  timeout 500 python3 render/run.py --no-ref --view-range 0:2 --iter-dir t172-tc > $S/prof-tc.log 2>&1
echo "tc rc=$?"; grep -E "^(STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL" $S/prof-tc.log | head -4
echo "TC lines: $(grep -c ' TC ' $O) TCEND: $(grep -c TCEND $O)"
fi
echo "=== done $(date +%T)"
