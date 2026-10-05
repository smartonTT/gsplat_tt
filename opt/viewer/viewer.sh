#!/usr/bin/env bash
# Live viewer on its own IRD box (resource key 'viewer'; never the measurement
# box or the p100 lock). Run from a checkout/worktree of gstt2 on the Mac.
#   opt/viewer/viewer.sh deploy [rev]  sync+build rev (default: newest best-iter-* tag) and restart
#   opt/viewer/viewer.sh start | stop | restart | status | log | tunnel
# Env: VIEWER_HOST (bh-35), VIEWER_PORT (8080 on the box), VIEWER_LOCAL_PORT (8091 on the Mac).
# One-time box setup (tt-metal, venv, scenes): opt/viewer/setup_box.sh.
set -euo pipefail
HOST=${VIEWER_HOST:-bh-35}
PORT=${VIEWER_PORT:-8080}
LPORT=${VIEWER_LOCAL_PORT:-8091}
VDIR=/localdev/smarton/viewer
DIR=$VDIR/tree
# Host-key checking stays on: StrictHostKeyChecking=yes fails on a changed key.
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
cd "$(git rev-parse --show-toplevel)"

rsh() { "${SSH[@]}" "$HOST" "$@"; }

do_stop() {
  rsh "VDIR=$VDIR bash -s" <<'R'
pid=$(cat "$VDIR/viewer.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  kill -INT "$pid"   # KeyboardInterrupt -> pipeline.close() -> device_shutdown
  for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { kill -TERM "$pid"; sleep 3; }
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
  echo "[viewer] stopped pid $pid"
else
  echo "[viewer] not running"
fi
rm -f "$VDIR/viewer.pid"
R
}

do_start() {
  rsh "VDIR=$VDIR DIR=$DIR PORT=$PORT bash -s" <<'R'
set -eu
pid=$(cat "$VDIR/viewer.pid" 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "[viewer] already running pid $pid"; exit 0; fi
cd "$DIR"
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME GSPLAT_SHA=$(cat SHA)
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

case "${1:-status}" in
  deploy)
    git fetch -q --tags origin
    REV=${2:-$(git tag -l 'best-iter-*' --sort=-v:refname | head -1)}
    SHA=$(git rev-parse "$REV^{commit}")
    "${SSH[@]}" "$HOST" true   # aborts here on a changed host key
    echo "[viewer] deploy $REV = $SHA to $HOST:$DIR"
    opt/sync_remote.sh "$HOST" "$DIR" "$SHA"
    # Tagged trees older than this script lack the launcher: ship it from here.
    rsh "mkdir -p $DIR/opt/viewer && cat > $DIR/opt/viewer/viewer_clean.py" < opt/viewer/viewer_clean.py
    # Trees before the viewer's uint8 fix show render_clean frames all white.
    rsh "grep -q 'render_clean packs 8-bit' $DIR/gsplat/viewer.py" ||
      { echo "[viewer] overlay gsplat/viewer.py (uint8 frame fix)"; rsh "cat > $DIR/gsplat/viewer.py" < gsplat/viewer.py; }
    do_stop; do_start ;;
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  status) do_status ;;
  log) rsh "tail -n ${2:-40} $VDIR/viewer.log" ;;
  tunnel) echo "ssh -N -L $LPORT:localhost:$PORT $HOST   # then open http://localhost:$LPORT" ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
