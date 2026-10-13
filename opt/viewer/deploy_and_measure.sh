#!/usr/bin/env bash
# Per-landing viewer duty (user msg #293) in one command (task #613): deploy best-iter-<N> to the
# live viewer box (bh-30, resource 'viewer') and measure end-to-end page FPS through the Mac
# tunnel localhost:8091. Run from a worktree of gstt2 on the Mac, NOT inside `ttp lock viewer`
# (renew.sh and each sweep take that lock themselves).
#   opt/viewer/deploy_and_measure.sh <N> [--runs 3] [--secs 23] [--force] [--no-deploy]
#   opt/viewer/deploy_and_measure.sh --rev REF ...
# 1. opt/ird/renew.sh viewer --rev best-iter-<N>: extends the viewer reservation and deploys when
#    the tree differs. If the viewer already runs a descendant of best-iter-<N> (newer viewer-only
#    commits), that commit is kept (deploy_note says so); --force deploys the tag anyway.
#    VIEWER_B4A_FOLD passes through to renew.sh/viewer.sh (empty = viewer.sh default).
# 2. waits (--ready-timeout, 600 s) for page 200 and websocket 101 on localhost:8091.
# 3. --runs sweeps of opt/viewer/ws_sweep_t532.py --no-decode (30 bench poses at 240 Hz) through
#    the tunnel, each under `ttp lock viewer`, with the Mac venv that has msgspec/zstandard/
#    websockets (VIEWER_SWEEP_PY, else <repo>/venv-macos/bin/python).
# Prints exactly one JSON line on stdout and exits with its rc:
#   {"deployed_sha", "page_fps" (median of runs), "device_ms" (median HUD 'Device render'),
#    "rc", "error" (only on failure), "rev", "host", "renew_status", "deploy", "deploy_note",
#    "page_fps_runs", "device_ms_runs", "interval_ms_p50", "link_mbs", "ready", "expiry_utc"}
# rc: 0 ok; renew.sh's codes 2-6 and 10-13 unchanged (see opt/ird/renew.sh); 2 also usage/bad rev;
#     20 viewer not up in time; 21 sweep failed; 22 no python with the sweep deps; 75 viewer lock busy.
# Never port 8081 (the user's LTX relay). Ssh to the box goes only through renew.sh/viewer.sh,
# which keep StrictHostKeyChecking=yes.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python3 opt/viewer/deploy_and_measure.py "$@"
