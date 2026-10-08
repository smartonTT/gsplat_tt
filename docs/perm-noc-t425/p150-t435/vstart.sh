#!/bin/bash
# t435: remote viewer start = body of do_start() in opt/viewer/viewer.sh at origin/smarton/tt-project-opt a18c4ee2
# (ETH overlay, task #415); failsafe so the viewer restarts even if the Mac-side driver dies.
VDIR=/localdev/smarton/viewer; DIR=$VDIR/tree; PORT=8080; TTMH=$VDIR/tt-metal; ETHOV=/localdev/smarton/viewer-eth-overlay; DISPATCH=
GSPLAT_VIEWER_STALL_FILE=; GSPLAT_VIEWER_PYSPY=
set -eu
pid=$(cat "$VDIR/viewer.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "[viewer] already running pid $pid"; exit 0; fi
sp=$(cat "$VDIR/viewer.sup.pid" 2>/dev/null || true)
if [ -n "$sp" ] && kill -0 "$sp" 2>/dev/null; then echo "[viewer] supervisor pid $sp is restarting it"; exit 0; fi
cd "$DIR"
export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME GSPLAT_SHA=$(cat SHA)
export TT_METAL_CACHE=$VDIR/tt-metal-cache
export GSPLAT_TT_ETH_OVERLAY=$ETHOV GSPLAT_TT_ETH_CACHE=$ETHOV-cache
unset GSPLAT_TT_DISPATCH
export NUMPY_MADVISE_HUGEPAGE=0
[ -f "$VDIR/viewer.log" ] && mv -f "$VDIR/viewer.log" "$VDIR/viewer.prev.log"
rm -f "$VDIR/viewer.pid" "$VDIR/viewer.stop"
unset GSPLAT_VIEWER_STALL_FILE GSPLAT_VIEWER_PYSPY
setsid nohup bash opt/viewer/supervise.sh "$VDIR" "$PORT" > "$VDIR/viewer.log" 2>&1 < /dev/null &
for _ in $(seq 25); do [ -s "$VDIR/viewer.pid" ] && break; sleep 0.2; done
echo "[viewer] started pid $(cat "$VDIR/viewer.pid") supervisor $(cat "$VDIR/viewer.sup.pid" 2>/dev/null) sha $GSPLAT_SHA port $PORT (log $VDIR/viewer.log)"
