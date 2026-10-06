# Lever A: blend lane during mat, a model with no device run (task #303)

All numbers in this doc are **model results**, not measurements. The GPU numbers are
published, not measured (G1 10.75 ms, G2 5.44 ms).

Question: in the fused mat+blend program (iter 206), TRISC idles ~2.4 ms per core during the mat
phase while the movers work. If TRISC blends small, already-ready tiles in that idle time (a
"lane"), how much does ms/view drop? The gate (docs/next-levers-after-205.md, section A step 1)
is a modeled gain of ≥0.4 ms at a 5 µs switch cost.

## Inputs

- `docs/profile-postl1-t297.md` and `docs/profile-postl1-t297/out/tracy-t297-ns-percore.csv`
  (iter 206, 30 bench views, 110 cores). Per-core mat end, BRISC/NCRISC busy time, cull
  batches and TRISC busy time in mat. Measured program end 6.247 ms (traced), blend span after
  mat 3.655 ms mean, mat end mean 2.513 / max 2.989 ms.
- The raw t297 Tracy CSV (NCRISC `rd_l1_bulk` zones and blend tile spacing), used by
  `docs/lane-model-t303/fit_today.py` to fit:
  - lane tile load on a mover: 2.4 µs fixed (claim, tile id, meta, dir, ready reads) +
    0.5 + 0.0008·n µs payload (~37 GB/s);
  - today's blend time per tile (TRISC-bound spacing, median per size bin): 74 µs at 128-256
    records, 164 at 512-1K, 277 at 1-2K, 429 at 2-3K, 571 at 3-4K, 872 at 6-8K. This has a
    larger fixed part than the older t147 fit (87 + 0.184·n µs), so small tiles cost more than
    the t147 fit says. The model uses these bins (`--fit today`), scaled so the per-core blend
    span equals the measured 3.655 ms.
- `docs/mat-split-sort-model/out/dump.txt.gz`: per-tile record counts for the 30 views
  (1024 tiles, ~2.64 M records per view; 617 tiles per view have ≤2048 records).
- `docs/matblend-ready-t273/model.py`: the mat worklist replay (build_mat_worklist order, t144
  mover fits) and the ready-flag blend simulation. That model predicted the fused result within
  ~0.1 ms (#289).

## Method (`docs/lane-model-t303/model.py`)

Per view:

1. Replay the mat worklist (LPT over 220 mover slots, big items on mover 0) and compute when
   each tile's ready flag is set. Calibrate per core to the t297 CSV: cores ranked by modeled
   mat end get the measured, sorted mat ends, and their ready times scale with them.
2. Baseline: today's blend. Each core starts at its own mat end and claims tiles in descending
   count order from the global counter. This reproduces the measured program end
   (6.24 ms vs 6.247 measured).
3. Lane: while a core is in its mat phase, it takes the smallest ready, unclaimed tile of at
   most R/32 records (R = ring size, 32 B records), from the small end of the list (a second,
   reverse claim counter). Costs charged per lane tile:
   - the mover load (NCRISC by default, `--mover brisc` uses BRISC's ~0.1 ms slack instead).
     NCRISC has no slack in mat, so every load extends that core's mat phase;
   - 1 µs of BRISC writer time for the output;
   - the switch cost (`--switch_us`, 2/5/10 µs);
   - the cull's share of TRISC (measured duty) and a yield cost per cull batch that arrives
     while the lane tile runs (`--preempt_us`, see "Why the lane must yield");
   - with `--l1_cost`: the ring plus ~25 KB of lane-only CBs come out of CB4 (512 KB). That
     lowers bucket_fit and ov_cap, so more tiles take the subchunk path in mat (replayed) and
     in blend (+87 µs per extra subchunk).
   A lane tile can only start once its data is in L1 (no prefetch). The lane may run past the
   core's mat end; the core then joins the main claim when its lane tile finishes.
4. Main blend: the same claim simulation over the tiles the lane did not take. Each core
   starts at max(mat end incl. extension, lane end).
5. Saving = baseline program end − lane program end, per view, then mean/min/max over 30 views.
   A safety check confirms the main claim never reaches a lane tile before the lane took it.

## Results

Program end of the fused mat+blend program, traced calibration (base 6.275 ms = measured 6.247).
Saving = base − lane, mean over 30 bench views (min, max). "mat ext" = how much the lane's mover
loads lengthen each core's mat phase (mean / max over cores). "lane work" = lane TRISC time per core.

| config | ring | switch | yield/batch | loads | L1 carve | mat order | saving ms (min, max) | lane tiles/view | lane work ms/core | mat ext mean/max ms | lane past mat max ms | lane first start ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r64_sw5 (spec case) | 64 KB | 5 µs | 0 | NCRISC ×1 | no | desc | **0.500** (0.26, 0.94) | 377 | 0.561 | 0.012 / 0.027 | 0.138 | 1.98 |
| r64_sw2 | 64 | 2 | 0 | NCRISC | no | desc | 0.503 (0.26, 0.94) | 383 | 0.568 | 0.012 / 0.028 | 0.135 | 1.99 |
| r64_sw10 | 64 | 10 | 0 | NCRISC | no | desc | 0.486 (0.26, 0.94) | 366 | 0.549 | 0.011 / 0.027 | 0.142 | 1.98 |
| r64_sw5_brisc | 64 | 5 | 0 | BRISC | no | desc | 0.501 (0.27, 0.94) | 366 | 0.552 | 0.000 / 0.011 | 0.142 | 1.98 |
| r64_sw5_load2 | 64 | 5 | 0 | NCRISC ×2 | no | desc | 0.489 (0.26, 0.93) | 380 | 0.562 | 0.024 / 0.055 | 0.138 | 1.99 |
| r64_sw5_pre2 | 64 | 5 | 2 µs | NCRISC | no | desc | 0.444 (0.25, 0.83) | 332 | 0.510 | 0.010 / 0.024 | 0.166 | 1.98 |
| r64_sw5_pre5 | 64 | 5 | 5 µs | NCRISC | no | desc | 0.383 (0.18, 0.77) | 281 | 0.453 | 0.009 / 0.022 | 0.227 | 1.97 |
| r64_sw5_l1 | 64 | 5 | 0 | NCRISC | yes | desc | 0.344 (0.06, 0.80) | 462 | 0.688 | 0.015 / 0.029 | 0.252 | 2.09 |
| r64_sw5_t147 (old blend fit) | 64 | 5 | 0 | NCRISC | no | desc | 0.508 (0.26, 0.95) | 372 | 0.561 | 0.012 / 0.026 | 0.129 | 1.99 |
| r64_sw5_t273 (t273 ready cal.) | 64 | 5 | 0 | NCRISC | no | desc | 0.327 (0.24, 0.54) | – | 0.396 | 0.007 / 0.024 | 0.198 | 2.19 |
| r32_sw5 | 32 | 5 | 0 | NCRISC | no | desc | 0.243 (0.06, 0.51) | 304 | 0.308 | 0.009 / 0.027 | 0.147 | 2.24 |
| r96_sw5 | 96 | 5 | 0 | NCRISC | no | desc | 0.722 (0.54, 1.06) | – | 0.791 | 0.014 / 0.031 | 0.138 | 1.75 |
| r96_sw5_l1 | 96 | 5 | 0 | NCRISC | yes | desc | 0.470 (0.06, 0.75) | – | 1.234 | 0.022 / 0.042 | 0.302 | 1.78 |
| r128_sw5 | 128 | 5 | 0 | NCRISC | no | desc | 1.054 (0.82, 1.30) | 479 | 1.179 | 0.018 / 0.036 | 0.152 | 1.34 |
| r64_sw5_small | 64 | 5 | 0 | NCRISC | no | small-first | 0.784 (0.57, 1.17) | 617 | 0.953 | 0.020 / 0.029 | 0.000 | 0.28 |
| **real64** (realistic) | 64 | 5 | 2 µs | NCRISC ×2 | yes | desc | **0.267** (−0.09, 0.67) | 418 | 0.619 | 0.026 / 0.054 | 0.252 | 2.08 |
| real96 | 96 | 5 | 2 µs | NCRISC ×2 | yes | desc | 0.350 (0.02, 0.56) | – | 1.090 | 0.040 / 0.080 | 0.313 | 1.79 |
| real128 | 128 | 5 | 2 µs | NCRISC ×2 | yes | desc | 0.525 (−0.35, 0.76) | – | 1.643 | 0.053 / 0.086 | 0.421 | 1.37 |
| **real64_small** | 64 | 5 | 2 µs | NCRISC ×2 | yes | small-first | **0.543** (0.19, 0.92) | – | 0.953 | 0.039 / 0.057 | 0.004 | 0.30 |
| real96_small | 96 | 5 | 2 µs | NCRISC ×2 | yes | small-first | 0.576 (0.06, 0.95) | – | 1.406 | 0.049 / 0.067 | 0.156 | 0.48 |
| pess64 (pessimistic) | 64 | 10 | 5 µs | NCRISC ×2 | yes | desc | 0.192 (−0.10, 0.60) | 353 | 0.529 | 0.022 / 0.047 | 0.306 | 2.08 |

"small-first" reorders each mover slot so lane-eligible (≤ ring) tiles are materialized first; the
slot totals do not change, and the baseline with that order is 6.286 ms (+0.011 vs today's order),
so its savings above are against its own baseline (vs today's 6.275 baseline: real64_small 0.532).
A pessimistic small-first run (10 µs switch, 5 µs yield) did not finish within the run's time limit
and is not reported.

Breakdown of the spec case (r64_sw5) → realistic (real64_small), mean ms/view, traced:
- 0.500 lane gain at 64 KB, 5 µs switch, NCRISC loads (mat ext 0.012 mean / 0.027 max per core:
  NCRISC contention is small because lane loads are small tiles, ~2.4 µs + payload each).
- −0.056 yield cost (2 µs per cull batch while a lane tile runs).
- −0.011 doubled mover load cost.
- −0.16 L1 carve (ring + 25 KB lane CBs out of CB4 push more tiles onto the subchunk path).
- +0.28 small-first mat order (small tiles become ready at ~0.3 ms instead of ~2.0 ms, so the lane
  starts early and finishes inside the mat phase: lane past mat max 0.004 ms).
  The parts do not add exactly; each was measured against the step before in the sweep.
- The worst view in real64_small still gains 0.19 ms; no view regresses. In real64 and real128
  some views regress (lane runs past mat end and delays the core's main claim).

## Why the lane must yield (and what that costs)

The cull runs with CULL_DEPTH=2 (`GSPLAT_TT_MATCULL_DEPTH`): each mover can push only two
128-record coefficient tiles before it waits for a mask. Batches arrive every ~12.9 µs per core.
One lane tile takes 74-570 µs of TRISC time. If the lane ran a tile to completion, the movers
would stall for that long and the mat phase would grow by the lane work, which cancels the
gain. So the lane must yield to the cull between record groups.

Yielding is not free. The cull uses DST tiles 0-4 (`microblock_band_cull_compute.cpp`); the
blend keeps its R/G/B/T accumulators in DST tiles 0-3 across the whole tile, plus X/Y/S in 4-6
(`alpha_blend_compute_mb.cpp`). With fp32 DST there is no room for both, so every yield has to
pack the 4 accumulator tiles to L1, run the cull, then unpack them back, plus the unpacker and
SFPU re-init. Estimated 2-5 µs per yield; the model charges it as `--preempt_us`.

## Costs the model does not charge

- Implementation risk in the fused kernel: TRISC0 decode-ahead, TRISC1 SFPU and TRISC2 pack
  all have to interleave two programs through the existing stream picker (`pick_stream` in
  `mat_cull_compute.cpp`), with DST save/restore.
- Mover-side lane servicing is modeled as serial load time; the mover loops have no natural
  poll point today.
- Traced vs untraced: the model is calibrated to a Tracy capture (11.42 ms traced vs 11.10
  untraced). #176 found untraced mat-side gains ~0.5× the traced ones; L1 came in at ~1×.

## Verdict

**Build, staged, behind a kill switch.** The gate is met, but only just in the realistic case.

- Spec case (5 µs switch, NCRISC loads, per-core CSV): 0.50 ms modeled. Passes the 0.4 gate.
  Mover contention is not the problem: lane loads lengthen each core's mat phase by 0.012 ms mean,
  0.027 ms max. BRISC as lane loader removes even that but gains nothing (0.501).
- With the costs the design cannot avoid (DST spill per yield, L1 carve, slower loads) and today's
  mat order: 0.27 ms. Fails. The lane starts too late (~2.0 ms into a ~2.5 ms mat phase), because
  today's worklist materializes small tiles last.
- Same realistic costs plus a small-first mat order: 0.54 ms (worst view 0.19, no regressions).
  Passes. Small-first costs the baseline ~0.01 ms in the model.
- Untraced: earlier mat-side levers came in at 0.5-1× their traced gain (#176 ~0.5×, L1 ~1×), so
  the realistic small-first case is 0.27-0.54 ms untraced. The pessimistic case (10 µs switch,
  5 µs yield, today's order) is 0.19 ms traced.

Given ~0.3-0.5 ms expected on a ~11.1 ms view (3-5%), with real implementation risk, build in stages
with cheap go/no-go points:

Stage 0 (small, device): small-first mat order alone (`build_mat_worklist`, `render/host/sort_device.cpp` / `sort_mover_split.h`:
sort each mover slot's items small tiles first, knob `GSPLAT_TT_MATORDER_SMALLFIRST`). Check on
device that ms/view stays within noise (model: +0.01 ms) and, with Tracy, that small tiles' ready
flags move to ~0.3 ms. If ms/view gets worse by >0.1 ms, stop and shelve A.

Stage 1 (the lane), knob `GSPLAT_TT_MATBLEND_LANE` (unset = off until the A/B passes):
- Host (`render/host/matblend_fuse.h`, `render/host/sort_device.cpp`): carve a 64 KB lane ring + ~25 KB non-aliased lane
  CBs (MB_COUNTS 8, OUT 12, IMG_U8 3, scratch 2) + ~16 KB DST spill from CB4; re-run the L1 fit
  check in prod and with `KCFG_EXTRA_KB` (Tracy). Add a second NoC-atomic semaphore: a reverse
  claim counter starting at the small end of the descending tile list. The main claim stops when
  it meets the reverse counter (two counters meet in the middle); the lane only takes tiles with
  ≤2048 records whose ready flag is set.
- NCRISC mover (`render/kernels/dataflow/matblend_ncrisc.cpp`, reusing the blend-load code of
  `reader_alpha_blend_mb_devcull.cpp`):
  between mat items, poll the lane request; claim via the reverse counter, check the tile's ready
  flag, load its records into the lane ring (no prefetch needed; loads are ~2.4 µs + payload).
- TRISCs (`render/kernels/compute/matblend_compute.cpp`; `pick_stream` in `mat_cull_compute.cpp`): add a MSG_LANE stream. The blend
  (`alpha_blend_compute_mb.cpp`) runs record groups from the ring and yields to the cull between
  groups; on yield pack R/G/B/T DST 0-3 to the spill buffer, run the cull batch (DST 0-4), unpack
  and re-init. This DST save/restore is the main risk and cost (2-5 µs per yield in the model).
- BRISC (`render/kernels/dataflow/matblend_brisc.cpp`): write lane output tiles through the non-aliased OUT/IMG_U8 CBs (~1 µs each).
- A/B under `ttp lock p100` on the bench (30 views), md5 906e0435 must hold, hero screenshot +
  diff + PSNR vs reference_v2, and a visual check for tile artifacts. Keep only if ≥0.3 ms untraced.

If Stage 1's DST spill turns out to cost more than ~5 µs per yield, the model says the gain falls
below 0.4 (r64_sw5_pre5: 0.38 even without L1/order effects); shelve then.

## Reproduce

```
python3 docs/lane-model-t303/model.py --fit today                       # 64 KB, 5 µs, NCRISC loads
python3 docs/lane-model-t303/model.py --fit today --preempt_us 2 --load_mult 2 --l1_cost
```

Each run takes ~1 min (30 views). Outputs of the sweep are in `docs/lane-model-t303/out/`.
