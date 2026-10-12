# t525: review of #519 MAT_EARLY_CULL (ttp/t513 head f0b2e31e)

No correctness defects found. Checked:
- Host gate: only with MATCULL_TRISC_FILL + SORT_ONELAUNCH and without OL_MAT_SELECT (kernel #error guards the same).
  Now default on; GSPLAT_TT_MAT_EARLY_CULL=0 turns it off.
- Whole-tile path: keys taken (record_keys) before post_job lets TRISC2 overwrite word 3; at most two jobs
  in flight (pend + early) and CB_JOB has 2 pages; the TRISCs pop a page on pick and finish jobs in order, so
  a page is reused only after its done word was seen. The radix ping-pong (kB) placement is unchanged and
  stays clear of the bucket records being culled. emit_slab after wait_job emits already-culled records
  (no second cull).
- Big-item path (N <= MAX_N, NCRISC only): k is in CB_BSORT, separate from CB_BUCKET; select_compact_ids
  rewrites k only when the candidates fit (else k keys stay intact for the full-sort fallback). Gather
  stages 8192-record chunks through CB_SLAB (same as the N > MAX_N path), barriers before reuse; candidate
  sort arrays 4 m u32 in CB_SLAB (checked against scap); stable sort over index-ordered candidates gives
  the whole tile's ranks [lo, hi) at [lo - base, hi - base). m >= L_item >= 1, so no empty job.
- Host test tests/unit/test_mat_early_cull.cpp passes on the Mac (all ok; 14/80 big items fall back).
