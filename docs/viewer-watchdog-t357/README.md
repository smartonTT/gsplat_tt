# Viewer render watchdog (task #357)

## The 2026-10-07 hang (bh-30, viewer at 4307f076)

Evidence: bh-30:/localdev/smarton/viewer/t347-test/hang_dump_2155.txt (py-spy --locals)
and viewer_hang.log (the viewer log of that run).

- Python side: FastRenderer thread `Thread-2` in `CleanBackend.render_fused`
  (render/run.py:120), i.e. inside the native `_clean.render_view` call, at
  1024x1024, contrib floor 1/255. All other threads idle; the heartbeat kept
  logging, so render_view held no GIL.
- Native side: render_view writes `[SUBCHUNK]` and `[SORT] stage=ONELAUNCH` to
  unbuffered stderr once per render, right after the sort's bin_emit counts come
  back (render/host/sort_device.cpp:2603-2626). The log has 433 of each = 34
  SELFTEST renders + 399 viewer frames; the hung render printed neither and no
  `[render_clean] tile overflow` line. So it never left stages 1-3 (project,
  gather, tile_assign) or the sort's bin_emit launch/readback, and the overflow
  retry ladder was not involved.
- 99 % CPU on one thread with no progress fits a host thread spinning in a
  tt-metal `Finish()` / blocking readback (it busy-polls the completion queue)
  for a device program that never finished. Which Finish cannot be told from a
  Python-only dump; set VIEWER_PYSPY to a py-spy binary so the watchdog adds a
  `--native` dump next time.
- Pose: not recoverable (py-spy shows no array values). All 433 renders had the
  same P=1263763 pairs (the hero view), and the hang began right after
  `client 1 camera restored to the pose it had 1s ago` (20:26:40Z), so the hung
  pose was most likely the hero view itself: a rare, timing-dependent device
  hang rather than one pose. That also makes it unlike the #284/#330 fill-off
  far-pose hang, which overflows tiles and logs the retry first.

Replay / soak (device lock held):

    TTW_ALLOW_DIRECT=1 .venv/bin/python opt/viewer/replay_pose.py --hero -n 2000 --timeout 5
    TTW_ALLOW_DIRECT=1 .venv/bin/python opt/viewer/replay_pose.py <viewer_hang_pose_*.json> --timeout 30

## What the viewer does now

- gsplat/render_watchdog.py wraps `pipeline.render`. Each render's w2c, K, W, H
  and cull settings go into a 256-entry ring (viewer_poses.jsonl on watchdog and
  on a clean stop). A render over 5 s (GSPLAT_VIEWER_WATCHDOG_S; a resolution's
  first render 600 s for the JIT) appends its pose, the replay command, the ring
  tail and all thread stacks to <viewer dir>/viewer_hang.log, writes
  viewer_hang_pose_<UTC>.json and exits 86.
- opt/viewer/supervise.sh (run by viewer.sh start) restarts on any non-zero exit,
  backoff 5 s doubling to 300 s, at most 6 restarts an hour, log
  viewer_supervisor.log. `viewer.sh stop` touches viewer.stop so it stays down.
- Test hook: `VIEWER_STALL_FILE=<path> opt/viewer/viewer.sh start`; writing a number of
  seconds to that file makes the next render sleep that long once.
