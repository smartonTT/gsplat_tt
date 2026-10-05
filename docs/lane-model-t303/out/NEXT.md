# t303 notes for the next run (delete before final commit)
State: worktree tt-project/worktrees/t303, branch ttp/t303-lane-model (pushed, head 4bfa2d7e).
Doc draft docs/lane-model-t303.md has placeholders RESULTS_TABLE and VERDICT.
Sweep (detached, /tmp/t303_sweep.sh) writes docs/lane-model-t303/out/*.txt; done when
tt-project/state/runs/813/t303sweep2.rc exists.

Known so far:
- Bug fixed in csv calibration (ready times were /1000 too small -> every tile looked ready at t=0;
  first sweep invalid, 0.80 ms there is WRONG). After the fix, 4-view test, R=64 sw=5 preempt=0:
  saving 0.687 ms (min 0.375, max 0.858), lane 474 tiles/view, lane work 0.757 ms/core, lane first
  start 1.77 ms (small tiles become ready late in mat), lane runs past mat end by up to 0.12 ms.
- Base reproduces measured program end (6.24-6.29 vs 6.247 measured, traced).
- Yield cost mechanism: cull uses DST 0-4, blend holds R/G/B/T in DST 0-3 + X/Y/S 4-6 for a whole
  tile -> each yield = pack 4 fp32 tiles + unpack + reinit ~2-5 us; yields every ~12.9 us/core
  while a lane tile runs (CULL_DEPTH=2). Spill also needs ~16 KB L1. Run-to-completion is not
  viable (stalls movers by the lane work).
- Untraced factor 0.5-1x (t176 / L1).
Verdict rule: gate >=0.4 ms modeled at 5 us switch. Judge on the realistic config (real64:
  preempt 2, load_mult 2, l1_cost) and pess64; if real64 >= 0.4 -> build with sketch (kernels:
  mat_cull_compute pick_stream gets MSG_LANE stream w/ DST spill; matblend_ncrisc mover loop polls a
  reverse claim counter (second NoC atomic semaphore, small end of desc list) and loads lane tiles
  into a lane ring carved from CB4; BRISC writes lane output via non-aliased OUT/IMG_U8; host
  matblend_fuse.h/sort_device.cpp: carve ring+25KB+16KB spill from CB4, kill switch
  GSPLAT_TT_MATBLEND_LANE=0, L1 fit check prod + KCFG_EXTRA_KB; main claim must stop at the lane
  counter (two counters meet in the middle). Device A/B under ttp lock p100, md5 906e0435,
  screenshot, gate 0.3). Else shelve with rationale.
