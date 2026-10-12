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

bh-33 (P150 measurement reservation), commit 28e557b9, drive docs/b4a-step1/drive.sh, logs in
out/ (run-r<round>-<mode>-<arm>.log). Warm-up round r0, then 3 rotating rounds. Per-run value =
run.py's ms_frame_median over 20 passes.

| arm (GSPLAT_TT_K2_EXTRA) | r1 | r2 | r3 | median | delta vs x0 |
|---|---|---|---|---|---|
| x0 (unset, default) | 8.987 | 8.986 | 8.986 | **8.986** | — |
| x1 (one extra K2) | 9.630 | 9.641 | 9.635 | **9.635** | +0.649 |
| x2 (two extra K2) | 10.282 | 10.281 | 10.285 | **10.282** | +1.296 |

- Linear: +0.648 ms/view per extra K2 (x1 +0.649, x2 +0.648 per K2). Spread within a round
  under 0.015 ms; r2-x1 had a 6-pass excursion (mean 9.760) but its median is in line.
- raw_md5 5a438f5a in all 11 b2b runs (x0, x1, x2): the extra K2 runs do not change the image.
- Latency mode (secondary, no --dump-views, r1): x0 8.856, x1 9.524 ms/view (+0.668).
- x0 (8.986) matches the t548 baseline (8.989 off / 8.992 shadow on).

## Verdict: GO for the real fold (ceiling about 0.65 ms/view, 7.2 %)

K2 sits fully on the critical path: about 0.65 ms/view of b2b time (the upper end of the
0.4-0.76 ms device-time range). Step 0 (t548) showed BRISC in mat+blend runs the full K2 for free
(8.989 vs 8.992). So a fold that moves K2(N+1) into blend(N) can save up to ~0.65 ms/view
(8.99 -> ~8.34, 111 -> ~120 FPS b2b), far above the 0.1 ms keep gate.

Not done in this task: the fold itself (the multi-file change listed above). Its b2b A/B, md5
906e0435 30/30, device hero.png + diff + PSNR are still owed by the follow-up. The probe flag
stays default off and changes nothing unless set.

One open point for the follow-up design: in today's order the host's sort work (bin_emit
~0.41 ms, publish_host ~0.33 ms, host stage timings) runs after K2(N+1); if the device idles
while it runs, the fold also lets pfwc/gather(N+2) fill that gap. Check the device timeline
(device profiler or per-stage Finish probes) before choosing the enqueue order.

# t552 B4a step 2: the real K2(N+1) fold (bh-33, 2026-10-12)

Commits e6af4f5e (fold), af4778bd (drive), 4b54d12b (build fix), on top of t551 aa8681db.
Flag GSPLAT_TT_B4A_FOLD=1 (default off): pfwc/gather(N+1) is enqueued before mat+blend(N) into
a parity-selected blendrec, BRISC in the fused blend runs K2(N+1), and view N+1 skips its own K2
enqueue. GSPLAT_TT_B4A_VERIFY=<n> reads back and compares the folded K2 outputs against a
standalone K2.

Drive docs/b4a-t552/drive.sh (remote docs/b4a-t552/remote_ab.sh) under `ttp lock p100`, build
#124 sha 4b54d12b on bh-33; logs in docs/b4a-t552/out/.

## Correctness

- Fold verify (VERIFY=3, latency and b2b mode): proj_M, pairs_P, gids, tids (2.16M / 2.26M
  words) and k2_rows (225280 words) equal to the standalone K2 on every checked view; 0 bad.
- Fold on: sweep md5 **906e0435**, 30/30 views (VIEWS_DIFFER 0 of 30). b2b raw_md5 5a438f5a in
  all off and on runs (same as t551), identical_across_passes=yes.
- Device hero (bh-33, fold on, commit 4b54d12b): opt/metal-screenshots/ttw-t552/hero.png and
  hero_diff10.png. PSNR vs benchmarks/reference_v2/hero.png **42.51 dB**; golden 8-bit match,
  max LSB 0 vs hero_golden_8bit. Looked at both images: no tile seams or block artifacts; the
  diff shows only the usual edge/spoke residue of the reference comparison.

## A/B (render/run.py --back-to-back, 30 bicycle views, 20 passes, ms_frame_median)

Warm-up round r0, then 3 alternating rounds (r1 off first, r2 on first, r3 off first).

| arm | r1 | r2 | r3 | median |
|---|---|---|---|---|
| off (flag unset) | 8.988 | 8.994 | 8.989 | **8.989** |
| on (GSPLAT_TT_B4A_FOLD=1) | 8.598 | 8.600 | 8.591 | **8.598** |

- Gain **0.391 ms/view b2b (4.3 %)**, 111.2 -> 116.3 FPS. That is 60 % of the 0.648 ms K2
  ceiling measured in t551. Warm-up round r0 agrees (off 8.985, on 8.591).
- Latency mode (secondary, r1): off 8.901, on 8.671 ms/view.
- Stage view (b2b pass 10, r3): project_gather_wait 2.103 -> 0.239 ms (pfwc/gather and K2 now
  ride inside the previous blend), blend 5.706 -> 7.287 ms (it now also covers them), sort
  0.994 -> 0.828 ms. Host sort work is unchanged: bin_emit 0.41/0.36 ms, publish_host 0.33 ms.

## Verdict: KEEP (gate >=0.1 ms/view, md5 906e0435 30/30, PSNR >=42.4 dB, no seams: all met)

The flag stays default off in this commit. Next: make it the default and land it as a kept
iteration (dashboard entry with this hero), then look at the ~0.26 ms of the ceiling not
recovered; the host sort work (bin_emit + publish_host, ~0.69 ms) is the first suspect.
