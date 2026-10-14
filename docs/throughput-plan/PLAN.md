# Multi-view throughput plan (task #673, 2026-10-13)

Goal (user msg #358): render many views of one fixed scene (Mip-NeRF360
bicycle, 1237x822, full model) at the highest throughput. Any view order is
allowed (a random set can be reordered into a tour), and every cache or
reuse trick is allowed. Target: Faster-GS 1.83 ms/view, RTX 4090
(**published, not measured**).

This is a paper plan. No device work was done here. Labels: **M** = measured
(source given), **E** = estimate from measured parts, **P** = published.

## 1. Starting point and a figure correction

| Figure | Value | Label | Source |
|---|---|---|---|
| 1-chip b2b, best-iter-229, p100 bh-33 | 8.063 ms/view | M | iters.jsonl iter 229 |
| 2-chip AFR, pre-227 build | 4.226 ms/view (1.99x) | M | docs/l2-2chip/RESULT.md, #615 |
| 2-chip AFR at 229 | ~4.03 ms/view | E | 8.063 / 2 x 1.007 (chip-0 host skew from l2-2chip) |
| Ideal 1-chip floor (stage sums) | ~3.75 serial, ~2.6 overlapped | E | docs/structural-plan-4/PLAN.md |
| Faster-GS, RTX 4090 | 1.83 ms/view | P | charter (msg #348) |

Correction: ~4.2 ms/view is the **2-chip AFR** number, not 1-chip. The 1-chip
b2b headline at 229 is 8.06 ms/view. Single chip is 4.4x off the target;
multi-chip work cannot close that alone at 2 chips.

### Honest throughput bench (proposed rule)

Render V distinct views once per pass: the dataset's training cameras, in
any order (a nearest-neighbour tour is allowed). Any precompute for the view
set (clustering, tour, scene invariants) is charged to the pass. No result
may be reused across repeats of the *same* view. Without this rule, a cache
of final images would "win". Under it, a view-set-fitted prune
(drop Gaussians that never contribute in any view of the set) nets zero: it
needs a full prior render of every view, which costs a whole pass.

## 2. Ranked levers

Bound = ms/view saved at the steady b2b point, on 1 chip unless noted.
"Reopens shelve?" applies memory rule #299/#300: a shelved lever needs new
evidence.

| # | Lever | Family | Mechanism | Basis | Bound (ms/view b2b) | Exactness | Cost | Reopens shelve? |
|---|---|---|---|---|---|---|---|---|
| T1 | AFR on 4+ chips | (e) multi-chip DP | One host thread per chip, views dealt round-robin; no inter-chip traffic | 2 chips 1.99x M (l2-2chip); host phase B ~1.15 ms/view/chip M (t614) | 4 chips: 8.06 -> ~2.0 E; 5 chips ~1.6 E (beats 1.83 P); after R1-R5, 4 chips ~1.7 E | md5-exact per view (same kernels) | Low code (AFR exists); **needs a 4-board p150 host the charter does not name** | No |
| T2 | 2-chip AFR re-measure at 229, training views | (e) | Same as T1 on the 'p150-2board' host | 1.99x M pre-227 | 8.06 -> ~4.03 E (4.03 saved vs 1 chip) | md5-exact | Low; one device session | No |
| T3 | Single-chip R1-R6 (structural-plan-4) | per-view | Blend body, SIB, mat, stealing | SP4 sizing | 0.4-1.2 each (SP4); multiplies through T1/T2 | per SP4 | Queued (#664 R1 etc.) | No (no new task here) |
| T4 | Temporal saturation-depth cull | (b) cross-view reuse | Previous view's per-tile saturation depth (all pixels T<eps), reprojected with a conservative margin, drops pairs past it at K2 emit. Blend end checks every pixel reached T<eps before the cut; else that tile reruns with the full list | ~11% of pairs blend after saturation (blend-waste-t148, old build) M | <=0.3-0.4 E upper; 0.1-0.2 E realistic | Bit-exact by construction (cut only where verified saturated; fallback otherwise) | Medium: K2 emit filter + per-tile depth readback + fallback path | New lever |
| T5 | K-view batched dispatch | (a) | One program renders K views; launch/boundary and exposed reads amortized | t668 bound 0.24/0.36/0.42 at K=2/4/8 E; L-A/L-A' prefetch (229) already hides most gaps; #644 reads hidden M | 0.10-0.17 E for SH0 | md5-exact | High: persistent or multi-view program (kcfg limit #579) | No (never tried) |
| T6 | K-batch + SH3 colour read | (a)+(b) | Read SH3 coefficients once per K views | SH3 read 0.78 ms/view M at K=1; saves 0.39/0.58/0.68 at K=2/4/8 E (t668) | 0.39-0.68 E, **only with SH3 enabled** | exact | Part of #672 | No |
| T7 | Sort/order reuse along a dense tour | (b) | Warm-start sort from the previous view's order + insertion repair | 6.27 inversions/pair at 1.95 deg M (#529); dataset step ~1.9 deg E | **0 at dataset density**; 0.2-0.4 only at <=0.5 deg steps (viewer, #539) | exact | Medium | Yes: #529/#539 shelved for the bench; reopens only if the user names a dense-trajectory workload |
| T8 | Frustum-union candidate set per view cluster | (b)/(c) | Compact the Gaussian set to the union of a cluster's frusta, pfwc reads fewer | #644: 32% fewer reads -> -0.03 ms with even deal M | <=0.07 E | exact | Medium | Yes (chunk-cull t479/t493 shelved): no new evidence; not reopened |
| T9 | 2 views share one chip (spatial split) | (e) | Half the cores per view, 2 views in flight | imbalance-only gain E | <=0.2 E, overlaps R5 (stealing), does not stack | exact | Medium | No; not queued |
| T10 | Tile-bin reuse across views | (b) | Reuse pair lists of unchanged tiles | 17.5% of pairs kept at 1.95 deg, 0% identical tiles M (#529) | 0 | n/a | n/a | Rejected |
| T11 | Approximate (stale) order | (b) | Blend with the previous view's order | 42.34 dB at 1 deg M (#529) | up to sort cost | **breaks md5** | Low | Rejected unless the user relaxes the md5 gate |
| T12 | SH/colour reuse at SH0 | (b)/(c) | Cache view-dependent colour | SH0 colours already pinned per scene | 0 | exact | n/a | Rejected (done) |
| T13 | More per-scene invariants | (c) | cov3d, opacity, SH0 pin, opacity prune already resident; bf16 pfwc inputs = SP4 R7 | #529 option C 0 M; #644 reads hidden M | 0-0.3 (R7, already in SP4) | R7 needs PSNR check | — | No new task |
| T14 | Host/pipeline overlap | (d) | Hide host per-view work | host view_total 8.16 vs device 8.063 M; zero-host bound <=0.03 M (t614) | <=0.03 | exact | — | Rejected (device-bound) |

**Bottom line.** Only multi-chip data parallel (T1/T2) moves throughput by
whole milliseconds. Every cross-view reuse lever is worth 0-0.2 ms/view at
dataset camera density, because training views are ~1.9 deg apart and
almost nothing (order, tile bins) survives that step. The one new
single-chip candidate worth sizing is T4 (saturation-depth cull). The path
to 1.83 ms/view is: single-chip R1-R6 (already planned) times 4-5 chips.

Throughput arithmetic (E): chips x (1 / per-chip ms). 4 chips at today's
8.06 give ~2.0 ms/view; 5 chips ~1.6; 4 chips after a 1-chip 6.8 ms/view
(R1-R5 mid-range) give ~1.7. AFR scales ~1.99x per doubling (M at 2), with
host feed far below the per-chip device time (1.15 vs 8.06 ms/view/chip, M).

## 3. Device tasks for levers above 0.1 ms/view

All device task specs below restate these binding restrictions: device work
only under `ttp lock p100 -- ...` on the measurement reservation (or the
named 'p150-2board' / user-approved host), one run at a time; own git
worktree of ~/dev/gstt2, never edit ~/dev/gstt2; no force push, rebase,
branch or tag deletion, no PRs; no host scanning or exploratory logins,
never disable SSH host-key checking; GPU numbers are published, not
measured. Every A/B runs `python3 opt/ab_jit_check.py` on both arms' JIT
cache dirs (a JIT-HASH FAIL voids the A/B, rule #656), 4 rotated rounds
plus a same-day A/A; a delta under ~0.1 ms/view is noise. Every kept
iteration needs a device hero.png, a diff image and PSNR against
benchmarks/reference_v2/hero.png, plus a human-eye check for tile
artifacts. Gates: md5 golden 077caaba (30 views), hero PSNR >= 42.4 dB.

### D1. 2-chip AFR re-measure at best-iter-229, b2b, orbit and training views (T2)

- Resource: 'p150-2board' (a 2-board p150 host other than bh-30, via the
  yyz-ird jump host; `ird reserve ... --team bgd` on tt_bgd). Release when
  done. Never touch the measurement reservation or the viewer box.
- Build best-iter-229 there. Arms: A = 1 chip, B = 2-chip AFR
  (existing AFR driver from afr-t581 / l2-2chip), same binary.
- Bench 1: render/run.py --back-to-back over the 30 bicycle views (md5
  077caaba per view on both chips).
- Bench 2: once #667 has the bicycle training cameras,
  scripts/training_cameras.py `cameras_train_shuffled` and `cameras_train_nn`
  at 1237x822, every view once.
- Report ms/view b2b for each arm and bench, per-chip skew, and the scaling
  factor; hero.png from each chip with PSNR and diff. Label as "2 chips",
  next to the 1-chip headline.
- Depends on #661 (if it holds the 2-board box) and on #667 for bench 2.

### D2. User decision: a 4-board p150 host for 4-chip AFR (T1)

- Not a device task until the user names a host. IRD offers multi-card
  Blackhole models (QuietBox/LoudBox with 4-8 p150, e.g. `--model lb`).
  The charter names only the p150 measurement box, bh-30 (viewer) and one
  2-board p150 host.
- Ask: may the project reserve one 4-board (or larger) p150 host via the
  yyz-ird jump host, only for the measurement, released afterwards?
- If approved: same as D1 with 4 chips (and 5+ if the box has them);
  check host feed (one host thread per chip; phase B ~1.15 ms/view/chip
  M) and PCIe contention on the readback. Kill line: if 4-chip scaling is
  below 3.6x, profile the host side before adding chips.

### D3. Saturation-depth cull, step 0 on CPU, then device A/B (T4)

- Step 0 (CPU, no device): with the CPU reference pipeline over the 30
  orbit views and a ~1.9 deg proxy tour (cameras_v2 densified, or the
  training cameras once #667 has them), measure per tile:
  (1) the share of K2 pairs behind the depth where every pixel reached
  T < 1/255 (the blend's own cutoff),
  (2) the capture rate when that depth comes from the previous view,
  reprojected with margin m (try 0, 2, 5% depth and 1 tile of screen
  slack),
  (3) the fallback rate (tiles where the predicted cut was too early).
  Kill line: net (captured pairs x blend cost per pair, minus fallback
  reruns) below 0.15 ms/view b2b at the measured pair cost -> shelve.
- Step 1 (device, only if step 0 clears): arm A = best-iter-229; arm B =
  K2 emit drops pairs past the predicted per-tile depth, blend writes a
  per-tile "unsaturated pixel" flag, host or a small kernel reruns flagged
  tiles with the full list. Bench: render/run.py --back-to-back, 30 views,
  4 rotated rounds + A/A, ab_jit_check on both arms, md5 077caaba must
  hold (the cut is exact), hero.png + diff + PSNR + visual check.
- The first view of a pass has no prediction: it renders in full (cost
  charged to the pass).

### D4. K-view batched dispatch, step 0 trace sizing (T5, optional)

- Step 0 (device, trace only): on best-iter-229 b2b, capture a device
  timeline (b2b-timeline tooling) and sum the gaps and exposed reads
  between views that a K-view program would remove. Kill line: below
  0.15 ms/view -> shelve for SH0; keep only as part of SH3 (#672, T6).
- Step 1 only if it clears: one program instance renders K views
  (persistent kernel or multi-view runtime args within the kcfg limit,
  #579). Same A/B protocol as D3.

### D5. Dense-trajectory order reuse (T7, conditional)

- Only if the user names a workload with steps <= 0.5 deg (viewer, smooth
  broadcast camera path). The bench (training views, ~1.9 deg) gains 0.
- Then: warm-start the mover sort from view N-1's order with insertion
  repair; A/B on a 0.5 deg orbit; exact by construction (repair yields the
  same order); bound 0.2-0.4 (#539, viewer-only).

## 4. Excluded, with evidence

- Tile-bin reuse (T10): 0% identical tiles at 1.95 deg (#529).
- Stale order (T11): breaks md5, 42.34 dB at 1 deg (#529).
- SH0 colour reuse (T12): already pinned.
- Host overlap (T14): device-bound, <=0.03 (t614).
- Frustum-union compaction (T8): <=0.07 (#644); chunk-cull shelves
  (t479, t493: c48 +0.177 on p150) stand.
- View-set-fitted prune: nets zero under the honest bench rule (needs a
  full prior pass).
- B2 probe: +0.015 now (#666), shelved again.

## 5. Sources

- docs/training-view/ANALYSIS.md (ttp/t668-training-view-analysis @91bd2f55)
- docs/structural-plan-4/PLAN.md (ttp/t663-structural-plan-4 @149af7ac)
- docs/training-bench/RESULTS.md (ttp/t667-training-bench @7f314142)
- docs/sort-coherence/STUDY.md (ttp/529-sort-coherence)
- docs/b4-overlap/PLAN.md, docs/chunk-cull-t479, docs/chunk-cull-t493,
  docs/u64-gather-t644, docs/blend-waste-t148, docs/l2-2chip/RESULT.md,
  docs/b2b-floor-t614
- Task ledger #529, #539, #591, #615, #627, #644, #645, #659, #661, #666
- IRD multi-card Blackhole models: DevInfra Confluence "IRD - Interactive
  Run Docker"
