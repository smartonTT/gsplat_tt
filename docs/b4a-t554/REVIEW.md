# t554: independent review of #552 (B4a K2(N+1) fold, 624032db) and default-on landing

Base best-iter-225 = b612d4ed (origin/main). Reviewed range for the fold itself: 795bc471..624032db
(t551 K2_EXTRA probe + t552 fold); the shadow (t542/t548) was reviewed in #547/#549.
Commit 69ee8b28 switches GSPLAT_TT_B4A_FOLD to default on (=0 is the kill switch).

## Code review (correctness risks)

- Order: pfwc/gather(N+1) is enqueued (before-enqueue hook) ahead of fused mat+blend(N) on the
  same CQ, so it completes before blend(N) starts. Anything blend(N) reads that pfwc(N+1)
  overwrites would be wrong deterministically; proj_m_blendrec, the one such buffer, is
  double-buffered (gather_fold_swap_blendrec) and the mat/blend args capture the old address
  before the swap. md5 906e0435 30/30 confirms nothing else is shared.
- Intra-program race: K2(N+1) on BRISC writes ta_pairs_gid/tid/keep, ta_pairs_P, proj_M and the
  K2 count rows. Fused mat+blend(N) reads sort outputs (sort_l1_recs, sorted_ids, tile_ranges,
  subchunk payload/meta/dir) and the blendrec, never the K2 outputs (sort(N) consumed those
  earlier on the queue). L1: same aliased scratch CB as the reviewed shadow. No race found.
- Fallbacks: non-fused blend (fz false, launch-only phase) runs the hook after the enqueue and
  b4a_view_cores stays 0, so fold_ran is false and view N+1 runs its own K2. A pfwc key miss
  (camera change, scene change) calls tile_assign_fold_cancel and the real K2 reruns. Pair
  overflow after a skipped K2 reruns K2 on pass 1. First view and views without a next pose
  run their own K2. fold_ran is recomputed at every fold end_view.
- Non-blocking notes: fold_cores counts every core when nseg != 0 (the function's nseg, not
  per-core), which only makes the "all cores got args" check stricter. The BRISC K2 is
  compiled with K2_DIET=1 fixed; with GSPLAT_TT_K2_DIET=0 the standalone and folded K2 differ
  in code path (outputs should still match; not verified).

## Device A/B (bh-33) and screenshot

See out/ and RESULT.md once docs/b4a-t554/drive.sh 69ee8b28 finishes.
