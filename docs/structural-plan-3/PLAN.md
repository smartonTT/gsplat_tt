# Structural plan 3: levers left after iter 227

Task #592, 2026-10-13 (PDT). No device use; desk ranking from committed results only.
Headline metric: `render/run.py --back-to-back`, 30 bicycle views, median ms/view (b2b).
Latency is secondary; latency with `--dump-views` is never a basis for keep or shelve.

Start point: best-iter-227 (445f9e40), **8.454 ms/view b2b** on bh-33 (p100, 11x10),
latency 8.647, hero PSNR 42.85 dB vs `benchmarks/reference_v2/hero.png` (gate 42.4).
Goldens since 227: `--dump-views` md5 **077caaba** (p100 / bh-33 11x10) and **84695855**
(120-core p150). 906e0435 is pre-227 and must not be used.
GPU reference: 10.75 ms/view, published (INRIA, RTX A6000, 1080p), not measured.
2 chips (AFR, t582, bh-50): 4.226 ms/view vs 8.395 on 1 chip (1.99x); reported next to the
single-chip headline, never instead of it.

## Where the 8.45 ms go (estimate)

No device profile of 227 exists yet. The newest is t224 (docs/profile-t224, best-iter-224,
9.131 ms): pfwc 1.654, k2 0.647, sort 1.459, mat+blend 5.330, gaps 0.040. Moving that by the
three landings since (B3 prune -0.133 b2b, mostly pfwc; B4a fold hides k2 in blend's BRISC,
-0.38 net; EXP1 -0.14 on blend SFPU) gives this **estimate**:

| stage | ms/view (est.) | bound by | evidence |
|---|---|---|---|
| pfwc | ~1.54 | TRISC SFPU, 72% fixed (all N = 5.56M after prune) | t224 fit 1.195 + 0.274·M; t484 TRISC split vis+precull 0.62, cov2d 0.46, projection 0.34, cov_cam 0.26 of 1.68; DRAM ~70% of 420 GB/s |
| sort | ~1.43 | scalar movers (bin count / emit scatter) | t224 fit 0.269 + 0.455·P; DRAM <= 50%; B1 (coalesced emit) NO-GO: runs too short (t534) |
| mat + blend (+ K2 fold) | ~5.44 | blend SFPU ~3.4-3.5, mat TRISC1 ~1.4 mean / 2.0 max, serial per core | t224 blend 3.667 mean / 3.938 max; mat TRISC1 idle 0.33 mean, 1.17 worst |
| gaps + host | ~0.04 | host is off the critical path | t224 all-core idle 0.040; publish_host 0.305 overlaps device work |

FPU use is ~0 in every stage. End-of-program core spread is ~0.34 (0.105 + 0.092 + 0.146).

**F0 (first queued device task, cheap):** device profile of best-iter-227 on bh-33 under
`ttp lock p100`, same tooling as t224 (program windows, per-RISC busy, mat fill zones via
`opt/profiler/fill_zones.py`). It replaces the estimate above and is step 0 of specs B and C.

## Ranked levers (>= 0.1 ms/view b2b)

Ranked by realistic gain, image risk as tie-breaker. All bounds are model or profile
bounds, not measured speedups.

| # | lever | bound ms/view b2b | realistic | evidence | risk | L1 / kcfg / BRISC-stack constraints |
|---|---|---|---|---|---|---|
| 1 | **B4 cross-frame overlap** (queued as #591; ranked, not specced here) | 0.6-1.2 (ideal period ≈ pfwc + mat+blend ≈ 7.0) | unknown after B4' failed | t539 rebound; t224 idle RISC slots | high: B4' sort-in-blend measured 11.505 vs 8.670 (t573) | L4 persistent program is 249,904 B code vs 105,472 B kcfg (t579); matblend_brisc stack margin 1104 B |
| 2 | **A. pfwc survivor compaction in L1**: cheap frustum/depth gate per chunk, then cov_cam / cov2d / conic / radii only on dense survivor tiles; order kept, bit-exact | 0.40 (cov_cam + cov2d ≈ 0.65 TRISC ms over all N, x ~0.6 culled) | 0.15-0.35 | t484 TRISC split; per-view visible M 1.76M of N 5.56M (32%); pfwc is 72% fixed cost over all N (t224) | medium: repack cost on a mover that already back-pressures on high-survivor cores (~0.15 ms, t484); DRAM floor 334 MB / 420 GB/s ≈ 0.80 ms | pfwc is its own program: own L1 budget; dense staging ~10 fp32 streams x 4 KB x 2 ≈ 80 KB must fit next to today's CBs; kernel code under kcfg; writer stack check if BRISC changes |
| 3 | **B. mat fill jobs + big-tile split + mover-cap rebalance** (t335 / t508 lever) | 0.18-0.26 ceiling (t508, p150, best-iter-223, pre early-cull) | 0.06-0.15 | t508 bound508.py: split alone <= 0.023, fill jobs alone <= 0.13; t224 TRISC1 idle in mat 0.33 mean, fill bound 0.29 | medium-high: three coupled changes (mover, blend reader flags per part, host worklist); measured levers realized well under ceilings | static CBs at the L1 ceiling in matblend; matblend_brisc stack 1104 B, tile_assign_scatter_seg 432 B; per-part flag pages are host/DRAM, not L1 |
| 4 | **C. mat band cull on the FPU** (128-record coefficient tile x microblock-centre tile matmul with a conservative margin), gated on attribution | <= band_batch share of mat TRISC1 (unknown; mat TRISC1 ~1.4 mean) | 0-0.3, unknown until F0 | t224 mat TRISC1 1.420 / 2.039; mat_cull_compute.cpp MATH runs band_batch on SFPU; plan-1 FPU kill (t111/t511) covered blend, not mat | high: precision (tf32-class FPU products, needs tile-local coordinates and a margin); movers may become the bound (TRISC1 already idles 0.33) | kernel code size of matblend near kcfg; needs one extra constant tile CB (4 KB) in a full L1 |
| 5 | pfwc byte cut (fp16 cov3d, packed rgb/opacity, 60 -> ~30 B/gaussian) | <= 0.3, only once pfwc is DRAM-bound | 0 today | pfwc TRISC-bound (t484); DRAM 70% | low-medium (cov precision) | follow-on to A only |
| 6 | p150 120-core box as headline | sign unproven | - | t577: bh-30 ETH 12x10 fold-on 8.119; t493 p150 rebaseline was slower than p100 | not a code lever | bh-30 is the viewer box; only measurable via the viewer exception or another p150 |

Interaction: if #591 hides sort and/or pfwc of view N+1 under blend of view N, levers on
those stages (A, 5) shrink to the part not hidden. mat/blend levers (B, C) add to B4.

Not ranked for b2b: **split frame across 2 chips** (screen split). It is a latency lever:
2.3-3.0 ms latency bound, per-chip 5.6-6.3 ms (t578 L3), 2-3 weeks. For throughput it loses to
AFR (4.23 ms/view measured). Worth doing only if the viewer needs lower latency, not faster b2b.

## Dropped, with reasons

| lever | reason |
|---|---|
| Excluded by spec (no new evidence): B1 coalesced emit, B2 ellipse pair test, B3 (landed 225), B4' sort-in-blend, L4 persistent program, alpha floor 2/255, T_EPS 1/128, coarser T periods, L6 Newton (landed), degree-1 exp (landed 227), B4a fold (landed 226), 2-chip AFR (measured) | earlier results stand: t534, t538, t573, t579, t580, t582 |
| Blend lane packing (plan-1 lever 5) | shares B2's ellipse math; B2 found only 6.8% dead pairs, net <= 0.09 (t538), and band cull already culls per microblock. Below 0.1 without new evidence; reopen only if a CPU count of dead (record, microblock) dispatches at 227 shows >= 0.1 ms (82 cycles per dispatch, t189) |
| mat TRISC1 idle fill alone (plan-1 lever 6) | folded into B; alone <= 0.13 (t508) |
| NoC / DRAM traffic in blend and sort | not the bound: blend DRAM 4%, sort <= 50% (t224); B1 showed emit runs are too short to coalesce |
| Host work in the b2b loop | 0.04 ms all-core idle (t224); publish_host 0.305 runs under device work; nothing to win |
| FPU in blend (quadratic form, dense GEMM, colour matmul) | measured net loss (t111, t511, t148); colour matmul dropped in t578 |
| cov_cam on the FPU | <= 0.26 TRISC ms, precision risk on cov2d determinant; lever A removes ~60% of that work anyway |
| VIS_FAST, PFWC_DEAL lpt, ply-order chunk cull | measured below gate or zero (t488/t494 -0.05/-0.07, t439, t466) |
| Less work per pixel under the quality bar | only 0.45 dB of PSNR margin left (42.85 vs 42.4); teps128 0.055-0.095, floor2 37.7 dB, coarser T periods slower (t580). No candidate >= 0.1 |
| Sort reuse across views | 0 on the 30-view bench (t529) |
| Scene pre-filter (60% of N never visible) | gaming the bench |
| Disjoint core groups | superseded by B4 |

**What remains above 0.1 ms/view b2b:** B4 (#591), A (pfwc survivor compaction) and B (mat fill
+ split) on bounds; C only if F0 shows band_batch is a large share of mat TRISC1. Outside B4, the
realistic sum is ~0.2-0.5 ms (2-6%). After these, the single-chip pipeline is near its
SFPU floor with the levers we know; further big steps need B4 or more chips.

## A/B specs

Rules common to every spec below (restated from the charter; binding):

- Own worktree of `~/dev/gstt2` at **best-iter-227 (445f9e40)** on a new `ttp/<task>` branch.
  Never edit `~/dev/gstt2` itself. Git: no PRs, no force push, no rebase/amend of pushed
  commits, no tag or branch deletion or move; main moves forward only.
- Device: only the existing **bh-33 measurement reservation** (p100). Never `ird reserve`,
  extend or release; never use bh-30 or the viewer reservation. Run
  `"$TTP_PROJECT/harness/bin/ssh-preflight" bh-33` before any sync; strict host-key checking
  on every ssh/rsync (`-o StrictHostKeyChecking=yes`; the user's ssh config sets `no` for
  bh-*, so override it; never edit that file or known_hosts by hand). No scanning, no
  exploratory logins.
- Every sync + build + run sequence is wrapped in **`ttp lock p100 -- ...`**. One device run
  at a time. Long runs go through `ttp detach`; no polling.
- Lever behind an env flag, default off in the A/B build. Protocol: one build, warm-up round,
  then 3 alternating rounds of `render/run.py --back-to-back` (1 check + 1 warm-up + 20
  measured passes, 30 bicycle views 1024x1024). Headline = median b2b ms/view; latency
  without dumps is secondary only.
- Correctness: `--dump-views` md5 **077caaba** on 30/30 views on bh-33 (84695855 if a
  120-core p150 run is ever made). These levers are bit-exact by design: a mismatch is a bug
  to fix, not a quality trade.
- Screenshot step (every arm that could be kept): device `hero.png` at that commit and config
  (rendered on bh-33, never the CPU reference), diff image and PSNR vs
  `benchmarks/reference_v2/hero.png` (>= 42.4 dB), look at hero and diff by eye for tile
  seams or artifacts, then add the iteration to `opt/ttw/iters.jsonl` and rebuild
  `REPORT.html` (validator must pass). GPU numbers are published, labeled published.
- Keep: >= 0.1 ms/view b2b median gain, same sign in all 3 rounds, md5 and screenshot gates
  pass. Landing: tag `best-iter-<N>`, push the tag, fast-forward main, run
  `python3 tt-project/harness/watchers/verify_main.py <N> --ff` and report "main
  fast-forwarded" only on its OK line; queue the viewer deploy + page-FPS task.

### Spec A: pfwc survivor compaction in L1 (`GSPLAT_TT_PFWC_COMPACT`)

Bound 0.40, realistic 0.15-0.35 ms/view b2b.

0. **CPU gate (no device, <= half a day).** From the CPU pipeline over the 30 views, per
   1024-gaussian chunk in pfwc deal order: fraction passing a cheap gate (depth > near and a
   conservative screen-rect test with a static per-gaussian radius 3 x max scale x f / z,
   plus the precull margin). Report mean and max survival per core per view, and check the
   gate never drops a gaussian the full pfwc keeps (must be 0). **Kill if mean survival >
   0.6** (saving < 0.25 TRISC ms).
1. Build: pfwc_compute splits each chunk into phase 1 (means, depth, cheap gate -> mask
   tile) and phase 2 (cov_cam, cov2d, conic fold, radii, precull) on dense survivor tiles.
   A mover repacks survivors' input words from the chunk's L1 pages into dense tiles and
   carries the original lane index so the writer emits survivors in today's order. Use the
   reader (NCRISC) for the repack: t484 shows the writer back-pressures on high-survivor
   cores. Partial dense tiles flush at the end of each core's slice. Per-lane math must stay
   the same ops in the same order (bit-exact).
2. Constraints to report: pfwc static CB total vs L1 after adding staging (~80 KB),
   pfwc kernel code size vs kcfg 105,472 B, BRISC stack margin if the writer changes.
3. Device A/B per the common rules, plus a pfwc program window and per-RISC busy capture
   (t224 tooling) for both arms.
4. **Kill rule:** b2b gain < 0.1 ms/view, or md5 not fixable. If the pfwc window drops
   >= 0.15 ms but b2b does not, profile once (TRISC vs reader/writer busy) before shelving.
   If the pfwc window drops < 0.15 ms, shelve with the profile.

### Spec B: mat fill jobs + big-tile split + mover-cap rebalance (`GSPLAT_TT_MAT_FILLSPLIT`)

Ceiling 0.18-0.26 (t508, pre early-cull), realistic 0.06-0.15 ms/view b2b.

0. **Rebound on 227 (uses F0).** Re-run `docs/mat-split-bound-t508/bound508.py` (or its
   equivalent on F0's bh-33 capture) for best-iter-227 with MAT_EARLY_CULL on. **Kill if
   the combined ceiling < 0.15 ms/view** (realized levers land well under ceilings).
1. Build in this order, measuring after each step (b2b A/B per the common rules):
   (a) mover-cap rebalance: raise or replace `kMatMover0Cap` (6144) and split mid-size items
   between movers in `build_mat_worklist` (no fusion change); (b) 512-record fill jobs with
   rank-ordered big gather on the permute path; (c) big-tile split with per-part ready flags
   in the blend reader (`reader_alpha_blend_mb_devcull.cpp`), part counts in the worklist,
   cut placed on key reads and select (shared pre-pass), not only the gather.
2. Constraints: matblend static CBs are at the L1 ceiling (no new L1 buffers; flags are
   host/DRAM pages); matblend_brisc stack margin 1104 B and tile_assign_scatter_seg 432 B must
   stay >= 256 B (report both); kernel code under kcfg.
3. **Kill rule:** stop after (a) if (a) + the rebound say (b)+(c) cannot add >= 0.1 more;
   shelve the whole lever if the cumulative b2b gain < 0.1 ms/view after (c) or md5 parity
   is not reached.

### Spec C: mat band cull on the FPU (`GSPLAT_TT_MAT_FPU_CULL`)

Bound unknown, <= band_batch share of mat TRISC1.

0. **Attribution gate (uses F0 fill zones).** From F0's `fz_*` markers: band_batch MATH
   cycles per core, and mover-side (NCRISC/BRISC sort + gather) time on the critical cores.
   Saving bound = min(band_batch on the critical core, critical-core TRISC1 time minus mover
   time). **Kill if the bound < 0.2 ms/view** (an FPU path rarely realizes more than half).
1. CPU model: evaluate the band test as a matmul of each 128-record coefficient tile with a
   constant microblock-centre tile, using tile-local coordinates (|x|, |y| <= 32) and tf32-like
   products; find the smallest margin that never drops a (record, microblock) pair today's
   SFPU test keeps over 30 views. **Kill if** that margin keeps > 5% more pairs (blend cost
   rises more than mat saves; 82 cycles per dispatch, t189).
2. Build: MATH does `matmul_tiles` instead of the SFPU band_batch, PACK compares with the
   margin and writes the same keep mask. Output must be a superset of today's mask; blend
   output stays bit-exact (extra pairs contribute alpha < 1/255 and are skipped).
3. Constraints: one extra constant tile CB (4 KB) in a full matblend L1 (find it or kill),
   kernel code under kcfg.
4. **Kill rule:** b2b gain < 0.1 ms/view, or any md5 mismatch that is not a superset-mask bug.

## Sources

docs/profile-t224/PROFILE.md (377b0ab7), docs/structural-plan/PLAN.md (c94c13b0),
docs/structural-plan-2/PLAN.md (ttp/t578 @bf864320), docs/precision-model (ttp/t580 @d285d66e),
docs/persistent-step0/RESULT.md (39bdd814), docs/l2-2chip/RESULT.md, docs/b4a-t554/REVIEW.md,
docs/b4a-t577, docs/coalesced-emit-tB/RESULT.md (t534), docs/b4-rebound/RESULT.md (t539),
docs/mat-split-bound-t508 (cd33efcd), docs/exp1-t584, opt/ttw/iters.jsonl (225-227).
