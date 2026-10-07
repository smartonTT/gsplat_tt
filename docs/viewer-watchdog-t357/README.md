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
- opt/viewer/supervise.sh (run by viewer.sh start) restarts on a non-zero exit,
  backoff 5 s doubling to 300 s, at most 6 restarts an hour, log
  viewer_supervisor.log. `viewer.sh stop` touches viewer.stop so it stays down.
  It also stops on exit 75 (viewer_clean.py found port 8080 taken) and on death by
  SIGINT/SIGTERM/SIGKILL (an outside stop). Reason: in the first live test, task
  #369's bench stopped the viewer with an older viewer.sh (no stop file) and started
  its own; the supervisor kept restarting, and viser quietly moved the extra viewer to
  port 8081. An OOM kill (SIGKILL) now leaves the viewer down, by design.
- Test hook: `VIEWER_STALL_FILE=<path> opt/viewer/viewer.sh start`; writing a number of
  seconds to that file makes the next render sleep that long once.

## Live test on bh-30 (2026-10-07, head 8329be50 = this branch + opt iter 210)

Deployed with `VIEWER_STALL_FILE=/localdev/smarton/viewer/stall_once opt/viewer/viewer.sh deploy`,
then `node opt/viewer/page_reconnect_check.mjs` drove a headless Chrome on the Mac through
localhost:8091: page open, frames, arm the hook (`echo 600 > stall_once`), drag the camera.

    23:38:10.769Z client 0 connected
    23:38:31.054Z [watchdog] STALL hook: render sleeps 600 s
    23:38:36.199Z [watchdog] render seq=241 1024x1024 stuck 5.1 s: pose and stacks in viewer_hang.log ... exiting 86
    23:38:36Z     [supervisor] viewer exited rc=86 after 141s; restart 1/6 this hour in 5s
    23:38:41Z     [supervisor] starting viewer_clean.py port 8080
                  SELFTEST hero 1024x1024 n=30 median=11.55 ms (86.6 FPS); HERO PSNR vs reference_v2/hero.png = 42.51 dB; READY
    23:38:54.009Z page: [gsplat] viewer connected   (reconnect shim, 22 s after the drag, no reload)

viewer_hang.log has the WATCHDOG header, POSE (w2c, c2w, K, 1024x1024, cull settings),
POSE_JSON, REPLAY command, RING and THREADS (render thread in render_watchdog._maybe_stall,
plus the sender, executor, watchdog, UI and main threads). hero_viewer_8329be50.jpg (device
frame after the restart) and page_after_reconnect.jpg (page after the reconnect) show no
tile seams. The viewer was then restarted without the hook (stall_hook=off, SELFTEST
11.53 ms, localhost:8091 -> 200).
