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

RESULTS_TABLE

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

VERDICT

## Reproduce

```
python3 docs/lane-model-t303/model.py --fit today                       # 64 KB, 5 µs, NCRISC loads
python3 docs/lane-model-t303/model.py --fit today --preempt_us 2 --load_mult 2 --l1_cost
```

Each run takes ~1 min (30 views). Outputs of the sweep are in `docs/lane-model-t303/out/`.
