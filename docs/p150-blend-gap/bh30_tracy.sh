#!/bin/bash
# t366: remote half on bh-30 (viewer stopped by drive_bh30.sh). One 30-view Tracy capture
# (device + host zones) of iter-209 code (#355 tree 010939d2, built; viewer's tt-metal
# 437bc366 used read-only), then csvexport -u. Pattern: docs/p150-gap-t350/bench_gap.sh tracy.
set -u
P=/localdev/smarton/p150bench-t355; V=/localdev/smarton/viewer; TAG=${1:-t366}
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 GSPLAT_PER_VIEW_STAGES=1
export TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1
cd $P/tree || exit 1; source .venv/bin/activate
O=$P/out366; PR=$P/prof366/$TAG; rm -rf $PR; mkdir -p $O $PR
export TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-}
cat > $PR/inner.sh <<EOI
#!/bin/bash
cd $P/tree; source .venv/bin/activate
TT_METAL_CACHE_RENDER=$P/cache-prof python3 render/run.py --no-ref --iter-dir t366-$TAG-T
EOI
chmod +x $PR/inner.sh
echo "=== $TAG tracy start $(date -u +%FT%TZ) sha=$(cut -c1-8 SHA)"
timeout ${RUN_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/$TAG-T.log 2>&1
rc=$?; echo "tracy rc=$rc $(date -u +%FT%TZ)"
grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL" $O/$TAG-T.log | head -5
tr=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
[ -n "$tr" ] && $CSVX -u "$tr" | gzip > $O/$TAG-tracy-u.csv.gz
[ -n "$dl" ] && gzip -c "$dl" > $O/$TAG-dev.csv.gz
ls -la $O
echo "=== end $TAG rc=$rc $(date -u +%FT%TZ)"; exit $rc
