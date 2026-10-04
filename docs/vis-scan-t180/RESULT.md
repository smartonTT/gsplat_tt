# Task #180: fused multi-core proj_vis_scan — not built

**Finding: proj_vis_scan does not run in the default bicycle chain at the tip (f39a023), so
folding it into the scatter would save ~0 ms/view on the benchmark. No code, no device run.**

Evidence (no device time spent; all from committed captures and code):

- `render/host/vis_mode.h` `pfwc_fuse_mode()`: GSPLAT_TT_PFWC_FUSE defaults to 1 with
  GSPLAT_TT_SFPU_VIS=1 (default since task #122). Then `gather_visible_device.cpp` takes the
  `pfwc_ran_fused()` branch: `tile_assign_fused_k2` publishes proj_M / ta_pairs_P, and
  `wl_vis_scan` / `wl_vis` (gather_vis_scan + gather_vis_scatter) are never enqueued.
- The newest capture of the default chain, task #170 Tracy (yyzo-bh-07 p100a, 30 views,
  docs/k2-diet-t170/README.md and out/tracy-off-gaps.txt), lists five programs per view:
  pfwc 2.98 | K2 0.99 | sort_ol 3.55 | mat 3.01 | blend 5.87 ms (base arm). No proj_vis_scan,
  no proj_scatter, no ta_bucket_scatter.
- The t170 base run log has no "[gsplat_tt::pfwc] ... running the PFWC_VIS program" fallback.
- The 0.40 ms figure that #179 modelled came from docs/t159, a re-analysis of the older t142-pc
  capture, whose chain was pfwc | proj_vis_scan+proj_scatter | ta_bucket_scatter | ...: the
  lever 2 path (fuse not engaged in that run). It does not describe the current default.

proj_vis_scan still runs with GSPLAT_TT_PFWC_FUSE=0 or GSPLAT_TT_SFPU_VIS=2 (cross-check), which
are kill switch / debug modes and not worth optimizing.

Design notes in docs/proj-vis-scan-model/RESULT.md (branch ttp/t179-proj-vis-scan-model) stay valid
if the lever 2 path ever becomes the default again.
