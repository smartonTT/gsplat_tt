# t551 B4a step 1: K2(N+1) fold into fused mat+blend(N) — ceiling probe first (bh-33, 2026-10-12)

Base: t548 head 795bc471 (B4a shadow K2, default off). Probe commit 28e557b9.

## Why a probe before the fold

The real fold is a multi-file change, not a kernel tweak. What it needs (from the code):

- K2's runtime args are view-independent (buffer addresses, nseg, tiles, core ids, mover
  split); only its inputs and outputs move. So K2(N+1)'s args can be set when blend(N) is
  enqueued.
- K2(N+1) reads pfwc(N+1)'s outputs (proj_m_offs, proj_m_aabb, pfwc_fuse_counts). The blend
  occupies all 110 cores, so pfwc(N+1) cannot run beside it on a second CQ; it must be enqueued
  before mat+blend(N) on the same queue (today the xview hook enqueues it after blend(N)).
- pfwc(N+1) then overwrites buffers mat+blend(N) still reads: proj_m_blendrec
  (launch_subchunk_materialize, sort_device.cpp:833) and the blend's proj_m_a/b/c/px/py/opacity
  (blend_device.cpp process_frame). These, the K2 outputs (gids, tids, pairs_P, proj_M, k2 rows)
  and pfwc's own count/offset buffers need two copies selected by view parity, through the
  pfwc, gather, K2, early sort, mat and blend runtime args.
- Host order changes too: a "before blend enqueue" hook for pfwc(N+1); view N+1 skips the K2
  enqueue and reads proj_M(N+1) after blend(N); the early sort(N+1) goes right behind blend(N).

Before building that, GSPLAT_TT_K2_EXTRA=<n> (profiling only, default 0) enqueues the segment K2
n extra times per view. K2 rewrites the same outputs from the same inputs, so the image must not
change (raw md5 identical). The b2b ms/view added per extra K2 is the K2 time on the critical
path: the most the fold can remove (it also removes one program dispatch, which the extra run
adds back, so the two match).

## A/B (render/run.py --back-to-back, 30 bicycle views, 20 passes, median ms/view)

PENDING (drive docs/b4a-step1/drive.sh 28e557b9, out/)
