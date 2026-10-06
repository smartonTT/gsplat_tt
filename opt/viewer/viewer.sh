#!/usr/bin/env bash
# Live viewer on its own IRD box (resource key 'viewer'; never the measurement
# box or the p100 lock). Run from a checkout/worktree of gstt2 on the Mac.
#   opt/viewer/viewer.sh deploy [rev]  sync+build rev (default: newest best-iter-* tag) and restart
#   opt/viewer/viewer.sh start | stop | restart | status | log | tunnel
# Env: VIEWER_HOST (bh-35), VIEWER_PORT (8080 on the box), VIEWER_LOCAL_PORT (8091 on the Mac; 8081 is taken by the LTX relay),
#      VIEWER_TT_METAL_HOME (/localdev/smarton/tt-metal; bh-30 uses its own build, /localdev/smarton/viewer/tt-metal).
# One-time box setup (tt-metal, venv, scenes): opt/viewer/setup_box.sh.
set -euo pipefail
HOST=${VIEWER_HOST:-bh-35}
PORT=${VIEWER_PORT:-8080}
LPORT=${VIEWER_LOCAL_PORT:-8091}
TTMH=${VIEWER_TT_METAL_HOME:-/localdev/smarton/tt-metal}
VDIR=/localdev/smarton/viewer
DIR=$VDIR/tree
# Host-key checking stays on: StrictHostKeyChecking=yes fails on a changed key.
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
cd "$(git rev-parse --show-toplevel)"

rsh() { "${SSH[@]}" "$HOST" "$@"; }

do_stop() {
  rsh "VDIR=$VDIR PORT=$PORT bash -s" <<'R'
# viewer.pid can hold setsid's pid rather than python's, so also match our own
# viewer_clean.py on this port; a survivor would keep the chip lock.
pids=$( { cat "$VDIR/viewer.pid" 2>/dev/null; pgrep -u "$USER" -f "opt/viewer/viewer_clean.py .*--port $PORT"; } | sort -u)
alive=; for p in $pids; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
if [ -n "$alive" ]; then
  kill -INT $alive 2>/dev/null   # KeyboardInterrupt -> pipeline.close() -> device_shutdown
  for p in $alive; do
    for _ in $(seq 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && { kill -TERM "$p"; sleep 3; }
    kill -0 "$p" 2>/dev/null && kill -KILL "$p"
  done
  echo "[viewer] stopped pid(s)$alive"
else
  echo "[viewer] not running"
fi
rm -f "$VDIR/viewer.pid"
R
}

do_start() {
  rsh "VDIR=$VDIR DIR=$DIR PORT=$PORT TTMH=$TTMH bash -s" <<'R'
set -eu
pid=$(cat "$VDIR/viewer.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "[viewer] already running pid $pid"; exit 0; fi
cd "$DIR"
export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME GSPLAT_SHA=$(cat SHA)
# JIT cache on /localdev: the default ~/.cache sits on the 9.4 GB home quota, which filled on bh-30 (task #263).
export TT_METAL_CACHE=$VDIR/tt-metal-cache
export NUMPY_MADVISE_HUGEPAGE=0  # THP compaction stalls on bh-35 (see viewer_clean.py)
[ -f "$VDIR/viewer.log" ] && mv -f "$VDIR/viewer.log" "$VDIR/viewer.prev.log"
# The pid file is written by the viewer process itself (exec keeps the pid): $! can be
# setsid's pid when setsid forks, and then stop misses the real viewer (task #257).
rm -f "$VDIR/viewer.pid"
setsid nohup bash -c 'echo $$ > "$0/viewer.pid"; exec .venv/bin/python opt/viewer/viewer_clean.py scenes/bicycle.ply --port "$1"' \
  "$VDIR" "$PORT" > "$VDIR/viewer.log" 2>&1 < /dev/null &
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$VDIR/viewer.pid" ] && break; sleep 0.2; done
echo "[viewer] started pid $(cat "$VDIR/viewer.pid") sha $GSPLAT_SHA port $PORT (log $VDIR/viewer.log)"
R
}

do_status() {
  rsh "VDIR=$VDIR DIR=$DIR bash -s" <<'R'
pid=$(cat "$VDIR/viewer.pid" 2>/dev/null || true)
state=stopped; [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && state="running pid $pid"
echo "[viewer] $state sha=$(cat "$DIR/SHA" 2>/dev/null || echo none)"
grep -E "SELFTEST|READY|Error|Traceback" "$VDIR/viewer.log" 2>/dev/null | tail -4 || true
R
}

# Copy this checkout's gsplat/viewer.py (uint8 frame fix) onto the tagged tree
# only when it is safe: skip if the tag already has the fix; refuse unless the
# tag's viewer.py differs from ours only in the fix block and its render/run.py
# still exposes what viewer_clean.py and the fixed viewer.py use.
overlay_viewer_py() {
  local sha=$1 tagged run extra
  tagged=$(git show "$sha:gsplat/viewer.py") || { echo "[viewer] tag has no gsplat/viewer.py" >&2; return 1; }
  if grep -q 'render_clean packs 8-bit' <<<"$tagged"; then
    echo "[viewer] tag already has the uint8 fix: no overlay"; return 0
  fi
  run=$(git show "$sha:render/run.py" 2>/dev/null) || run=
  grep -qE '^class CleanBackend\b' <<<"$run" && grep -qE '^def build_intrinsics\(' <<<"$run" || {
    echo "[viewer] tag's render/run.py lacks CleanBackend/build_intrinsics: refusing overlay" >&2; return 1; }
  # Lines that differ, minus the fix block itself (the uint8 branch and its re-indented else).
  extra=$(diff <(printf '%s\n' "$tagged") gsplat/viewer.py | grep -E '^[<>]' |
    grep -vE "^[<>] +(if result\.image\.dtype == np\.uint8:|image_np = result\.image  # render_clean packs 8-bit on device|else:|image_np = \(np\.clip\(result\.image, 0\.0, 1\.0\) \* 255\)\.astype\(np\.uint8\))$" || true)
  if [ -n "$extra" ]; then
    echo "[viewer] tag's gsplat/viewer.py differs beyond the uint8 fix: refusing overlay" >&2
    printf '%s\n' "$extra" | head -10 >&2; return 1
  fi
  echo "[viewer] overlay gsplat/viewer.py (uint8 frame fix; rest of the file identical)"
  rsh "cat > $DIR/gsplat/viewer.py" < gsplat/viewer.py
}

case "${1:-status}" in
  deploy)
    # GitHub can be unreachable from the Mac (port 22 timeouts); deploy from local refs then.
    timeout 60 git fetch -q --tags origin || echo "[viewer] git fetch failed: using local refs" >&2
    REV=${2:-$(git tag -l 'best-iter-*' --sort=-v:refname | head -1)}
    SHA=$(git rev-parse "$REV^{commit}")
    "${SSH[@]}" "$HOST" true   # aborts here on a changed host key
    echo "[viewer] deploy $REV = $SHA to $HOST:$DIR"
    REMOTE_TT_METAL_HOME=$TTMH opt/sync_remote.sh "$HOST" "$DIR" "$SHA"
    # Tagged trees older than this script lack the launcher: ship it from here.
    rsh "mkdir -p $DIR/opt/viewer && cat > $DIR/opt/viewer/viewer_clean.py" < opt/viewer/viewer_clean.py
    # Trees before the viewer's uint8 fix show render_clean frames all white.
    overlay_viewer_py "$SHA"
    do_stop; do_start ;;
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  status) do_status ;;
  log) rsh "tail -n ${2:-40} $VDIR/viewer.log" ;;
  tunnel) echo "ssh -N -o StrictHostKeyChecking=yes -L $LPORT:localhost:$PORT $HOST   # then open http://localhost:$LPORT" ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
