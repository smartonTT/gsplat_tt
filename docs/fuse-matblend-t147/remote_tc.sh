#!/bin/bash
# t147 remote job: base 30-view run (stage timings) then the per-tile cycle run.
set -u; export LC_ALL=C
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=/localdev/smarton/gstt2-t147; cd "$T" || exit 1; source .venv/bin/activate; mkdir -p tmp
C=TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t147
env $C timeout 400 python3 render/run.py --no-ref --iter-dir t147-base > tmp/t147-base.log 2>&1
echo "base rc=$?"
grep -E "^(STAGES|SORT_STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL" tmp/t147-base.log | head -6
O=$T/tmp/t147-tc.dprint; rm -f "$O"
env $C GSPLAT_TT_MB_TILECYC=1 TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=TR1 \
  TT_METAL_DPRINT_FILE="$O" timeout 600 python3 render/run.py --no-ref --view-range 0:2 \
  --iter-dir t147-tc > tmp/t147-tc.log 2>&1
echo "tc rc=$?"
grep -E "^(STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL" tmp/t147-tc.log | head -4
echo "TC lines: $(grep -c ' TC ' "$O") TCEND lines: $(grep -c 'TCEND' "$O")"
