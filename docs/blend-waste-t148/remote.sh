#!/bin/bash
# t148 remote job: base 30-view run (stage timings, md5) then the stats run.
set -u; export LC_ALL=C
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=/localdev/smarton/gstt2-t148; cd "$T" || exit 1; source .venv/bin/activate; mkdir -p tmp
C=TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t148
env $C timeout 400 python3 render/run.py --no-ref --iter-dir t148-base > tmp/t148-base.log 2>&1
echo "base rc=$?"
grep -E "^(STAGES|SORT_STAGES|SUMMARY)|md5|Traceback|TT_THROW|TT_FATAL" tmp/t148-base.log | head -8
O=$T/tmp/t148-stats.dprint; rm -f "$O"
env $C GSPLAT_TT_MB_STATS=1 TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=TR1 \
  TT_METAL_DPRINT_FILE="$O" timeout 2400 python3 render/run.py --no-ref \
  --iter-dir t148-stats > tmp/t148-stats.log 2>&1
echo "stats rc=$?"
grep -E "^(STAGES|SUMMARY)|md5|Traceback|TT_THROW|TT_FATAL" tmp/t148-stats.log | head -6
echo "MBSTATS lines: $(grep -c MBSTATS "$O")"
