# t340: fill-off far-pose hang soak

During #330 one far-pose render with `GSPLAT_TT_MATCULL_TRISC_FILL=0` hung on
its 3rd timed view (blend after sort, rc=124;
yyzo-bh-04:/localdev/smarton/gstt2-t284/tmp/t284/run-rv3-far2_255nofill.log,
build 19531e3b). This soak tried to reproduce it.

Setup: tree 3d6cacd3 (smarton/tt-project-opt 3a9c2519 + this script; the render
code differs from 19531e3b only by profiler-only `#if PROFILE_KERNEL` counters),
yyzo-bh-04 (p100a), bicycle hero dollied -2, contrib floor 1/255, so every
process grows the bucket to 65472 and runs the big-tile path (tile 554, 35032
records). `remote_soak.sh <tag> <n> [ENV=V]`, one devrun + `ttp lock p100` per
process, per-run timeout 150 s, 2026-10-07 09:14-09:19 UTC.

| arm | processes | timed views | hangs | ms/view | md5 (all views) |
|---|---|---|---|---|---|
| fill off, 20 views/process | 7 | 140 | 0 | 14.20-14.35 | c5605dd6 |
| fill off, 3 views/process (as in #330) | 10 | 30 | 0 | 14.21-14.38 | c5605dd6 |
| fill on (default), 20 views/process | 3 | 60 | 0 | 13.85-13.89 | c5605dd6 |

Result: not reproduced in 170 fill-off views (17 processes). No fix made.
Fill off and fill on give the identical image. The #330 hang stays a single
unexplained event on the kill-switch path; the default path is fill on.
