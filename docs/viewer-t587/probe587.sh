#!/bin/bash
# t587 (box side): pipeline_probe.py with the B4a fold off and on, viewer stopped. Env as
# viewer.sh do_start (ETH dispatch 12x10 on bh-30). Usage: probe587.sh <out dir> [probe args]
VDIR=/localdev/smarton/viewer; DIR=$VDIR/tree; TTMH=$VDIR/tt-metal; ETHOV=/localdev/smarton/viewer-eth-overlay
O=${1:?out}; shift; mkdir -p "$O"
cd "$DIR" || exit 3
export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole TT_METAL_RUNTIME_ROOT=$TTMH GSPLAT_SHA=$(cat SHA)
export TT_METAL_CACHE=$VDIR/tt-metal-cache GSPLAT_TT_ETH_OVERLAY=$ETHOV GSPLAT_TT_ETH_CACHE=$ETHOV-cache
unset GSPLAT_TT_DISPATCH
export NUMPY_MADVISE_HUGEPAGE=0 GSPLAT_VIEWER_PROF=1
rc=0
for f in 0 1; do
  GSPLAT_TT_B4A_FOLD=$f timeout 900 "$VDIR/venv/bin/python" opt/viewer/pipeline_probe.py --out "$O/fold$f" \
    --passes 5 --port 8097 "$@" > "$O/probe-fold$f.log" 2>&1 || rc=$?
  echo "fold$f rc=$rc"
done
echo "$rc" > "$O/probe.rc"
