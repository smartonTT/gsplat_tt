# t548 B4a step 0, re-measured: shadow K2 on the fused mat+blend BRISC (bh-33, 2026-10-12)

Fixes review #547. Commit 3c9bcd13 (on t542 c0959533; base best-iter-225 b612d4ed + t540 77d8fc3c).
Flag GSPLAT_TT_B4A_SHADOW=1 (default off). The t542 numbers measured a no-op: the fused BRISC
K2 args were gated on ctx->fused_ready, which tile_assign_tt clears every view before blend, so
they were always zero and the shadow never ran.

## Fix

- tile_assign_b4a_args gates on a new ctx->b4a_valid (set by tile_assign_fused_k2 when it fills
  b4a_args, cleared by blend after reading), not fused_ready.
- GSPLAT_TT_B4A_VERIFY=1: after a view, read back b4a_gids/b4a_tids (first P_pub), b4a_pp/b4a_mp
  and b4a_rows and compare word for word with buf_gids/buf_tids, ta_pairs_P, proj_M, buf_k2_rows.
- Per-run log of cores that got non-zero K2 args; a[16] guarded when b4a_rows is unallocated;
  drain the queue before reallocating the b4a_* buffers.

## Proof the shadow ran

- Verify, latency and b2b, views 1-3: gids/tids 2,895,629 / 2,895,629 / 2,090,575 words equal,
  pairs_P and proj_M 16 words equal, k2_rows 225,280 words equal. 0 bad views.
- `[B4A] cores_with_k2_args=110/110` on every logged view of every on-arm run.

## A/B (render/run.py --back-to-back, 30 bicycle views, 20 passes each, median ms/view)

| round | off   | on    |
|-------|-------|-------|
| warm  | 8.989 | 8.995 |
| r1    | 8.989 | 8.992 |
| r2    | 8.988 | 8.992 |
| r3    | 8.993 | 8.990 |
| median of r1-r3 | 8.989 | 8.992 |

Blend stage (B2B_STAGES): off 5.869 / 5.711 / 5.726, on 5.725 / 5.744 / 5.688. Latency mode 8.9 ms
both arms (p50 9.0). raw md5 5a438f5a on every b2b run, both arms, identical across passes.
On-arm hero (shadow running): 30-view sweep md5 906e0435 (0 of 30 views differ), golden match
(max LSB 0), PSNR 42.51 dB vs benchmarks/reference_v2/hero.png,
opt/metal-screenshots/ttw-t548/hero.png + hero_diff10.png; checked by eye, no tile seams.

## Verdict

Not kept (step 0 cannot give a gain; +0.003 ms/view is noise). Now backed by device proof: BRISC
in fused mat+blend reproduces this view's full K2 output on all 110 cores at no measurable cost to
the blend or the frame. So BRISC has the slack to host K2(N+1).

Step 1 (real fold) is worth building: K2 is ~0.4-0.76 ms of device time per view on p150
(docs/pfwc-skiprgb-t481, docs/k2-split-t274), all of which can leave the critical path. Suggested
order: (a) BRISC-only K2(N+1), since step 0 shows BRISC alone produces the full output for free;
the TRISC2 part only if (a) leaves K2 time exposed. (b) The hard part is the dependency: K2(N+1)
needs pfwc/gather(N+1) done before blend(N) reaches it, but under xview those are enqueued after
blend(N). It needs either pfwc(N+1) enqueued before blend(N) on another CQ with a device-side
wait in BRISC, or K2(N+1) placed in blend(N+1)'s predecessor slot. (c) Double-buffer the pfwc and
K2 outputs (and stop reallocating b4a_* while a blend may still write them).
