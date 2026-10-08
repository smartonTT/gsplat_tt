#!/bin/bash
# t427 (copy of docs/eth-dispatch-t397/vstart.sh): remote viewer start, the body of do_start() in opt/viewer/viewer.sh (origin/smarton/tt-project-opt),
# used by bench409.sh as a failsafe so the viewer restarts even if the Mac-side driver dies.
VDIR=/localdev/smarton/viewer; DIR=$VDIR/tree; PORT=8080; TTMH=$VDIR/tt-metal
GSPLAT_VIEWER_STALL_FILE=; GSPLAT_VIEWER_PYSPY=
set -eu
pid=$(cat "$VDIR/viewer.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "[viewer] already running pid $pid"; exit 0; fi
sp=$(cat "$VDIR/viewer.sup.pid" 2>/dev/null || true)
if [ -n "$sp" ] && kill -0 "$sp" 2>/dev/null; then echo "[viewer] supervisor pid $sp is restarting it"; exit 0; fi
cd "$DIR"
export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME GSPLAT_SHA=$(cat SHA)
# JIT cache on /localdev: the default ~/.cache sits on the 9.4 GB home quota, which filled on bh-30 (task #263).
export TT_METAL_CACHE=$VDIR/tt-metal-cache
export NUMPY_MADVISE_HUGEPAGE=0  # THP compaction stalls on bh-35 (see viewer_clean.py)
[ -f "$VDIR/viewer.log" ] && mv -f "$VDIR/viewer.log" "$VDIR/viewer.prev.log"
# The pid file is written by the viewer process itself (exec keeps the pid): $! can be
# setsid's pid when setsid forks, and then stop misses the real viewer (task #257).
rm -f "$VDIR/viewer.pid" "$VDIR/viewer.stop"
[ -n "$GSPLAT_VIEWER_STALL_FILE" ] || unset GSPLAT_VIEWER_STALL_FILE
[ -n "$GSPLAT_VIEWER_PYSPY" ] || unset GSPLAT_VIEWER_PYSPY
# supervise.sh restarts the viewer after a watchdog exit or a crash (task #357).
setsid nohup bash opt/viewer/supervise.sh "$VDIR" "$PORT" > "$VDIR/viewer.log" 2>&1 < /dev/null &
for _ in $(seq 25); do [ -s "$VDIR/viewer.pid" ] && break; sleep 0.2; done
echo "[viewer] started pid $(cat "$VDIR/viewer.pid") supervisor $(cat "$VDIR/viewer.sup.pid" 2>/dev/null) sha $GSPLAT_SHA port $PORT (log $VDIR/viewer.log)"
