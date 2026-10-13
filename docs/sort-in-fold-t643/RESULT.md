# t643 — L-A': enqueue sort(N+2) in the fold hook (GSPLAT_TT_SORT_IN_FOLD)

Base: best-iter-227. Branch `ttp/t643-l-a-enqueue-sort-n-2-in-the-fold-hook-be`, code @098d2bd6.
Box: bh-33 (p100 lock), 2026-10-13 13:10-13:15 PDT. Driver: `drive.sh` / `remote_ab.sh`; raw logs in `out/`.

## Change
With the switch on, view N+1's fold after-hook records a k2_done event and then enqueues the next
view's one-launch sort on the b4a buffers that K2(N+2) writes. At N+2, CQ1 reads proj_M after the
pre-sort event, so the sort no longer runs at the start of N+2. Clamp path (proj_M[2]!=0: Finish,
regrow, rerun) is unchanged, and the hook calls Finish before it grows any sort buffer (#552).
CQ1 path only. Verify mode and the non-CQ1 path behave as before. Default off.

## Correctness
- 11x10 md5 sweep, 30 bicycle views: base 077caaba 30/30, new 077caaba 30/30 (golden).
- Every b2b run: raw_md5 1d372b94 on both arms, identical_across_passes=yes.
- Device hero (switch on): `opt/metal-screenshots/ttw-t643-sortfold/hero.png`, diff
  `hero_diff10.png`; PSNR 42.85 dB vs benchmarks/reference_v2/hero.png, golden_match true (max 0 LSB).
  Inspected by eye: no tile seams or block artifacts; the diff shows only the usual edge noise.

## b2b (render/run.py --back-to-back, 30 views, 20 passes), ms/view
| round | base (off) | new (on) | gain |
|---|---|---|---|
| r0 (warm) | 8.487 | 8.293 | 0.194 |
| r1 | 8.456 | 8.305 | 0.151 |
| r2 (order flipped) | 8.459 | 8.245 | 0.214 |
| **r1+r2 mean** | **8.458** | **8.275** | **0.183 (-2.2%)** |

FPS 118.2 -> 120.8. Pass spread is about ±0.03 ms in each run, so the arms don't overlap.
The gain falls inside #629's 0.15-0.20 ms/view estimate.

## Verdict
KEEP (gain 0.18 ms/view, above the 0.10 kill line; md5 30/30; no hang). Landing is a separate step:
flip the default on, best-iter tag, verify_main.py --ff. Interaction: #639 (L-A, gather N+2) also
touches the fold after-hook. The two are independent here, but whichever lands second should rebase
onto the other and re-run this A/B.
