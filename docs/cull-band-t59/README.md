# Task #59 — band-extent microblock cull (lever L2)

**Board: yyzo-bh-07 (Blackhole p100a), not a p150.** Bicycle, 30 views, 1024x1024.

## What changed

The tile-local microblock cull (`tile_mb_mask`, TRISC SFPU) was the second most
expensive stage of the frame: 15.0 ms/core, about as long as the blend itself
(iter-162 profile, docs/OPTIMIZATION-PLAN.md §0.5). It put one gaussian in each
SFPU vector (32 lanes = 32 microblocks) and ran a box-constrained Mahalanobis
minimum per lane. Per (gaussian, tile) pair that was ~140 SFPU instructions plus
~100 scalar TRISC instructions (runtime-immediate coefficient loads built by the
RISC, 128 noinline calls per batch, per-record decode), about 900 cycles/pair.

New kernel `render/kernels/compute/microblock_band_cull_compute.cpp`:

- **One gaussian per SFPU lane**, 32 gaussians per vector, 128 per coefficient
  tile. The NCRISC reader (idle under the old SFPU cull) transposes the slab
  records into an fp32 coefficient tile (`fill_coeff_tile`), so every per-pair
  value reaches the SFPU through `copy_tile`, and every constant in the SFPU
  code is a compile-time immediate. No per-pair scalar work on the TRISC.
- **Walk the 8 microblock rows (bands).** For a fixed row the ellipse
  `m2 <= t` spans `u = R v -/+ S sqrt(P - Q v^2)`. The right edge is concave in
  `v`, so the band's right-most extent is at the ellipse's right-most point
  clamped into the band; the left edge mirrors it. A column block is kept iff
  that extent meets its pixel-centre span; the test is done on squares
  (`n|n| + S^2 d >= 0`), so there is no sqrt in the per-band work. Geometrically
  this is exactly "the ellipse meets the pixel-centre box", the same test the
  box-min cull did, so the mask is unchanged except for pairs on the thr
  boundary (same `kThrMargin` = 0.05 m2 slack).
- Keep bits are accumulated as fp32 `2^23 + mask16`, so the writer builds the
  32-bit mask with integer ops only.
- **Writer**: patches word3 of the records in the reader's L1 slab slot and
  writes the slab back in 2 KB page writes (the shape `sort_subchunk_materialize`
  emits). The old per-32-record 64 B DRAM read-modify-write is gone.

Model: `opt/golden-check/cull_mask_check.py` `band_keep_f32`; tests:
`tests/spec/test_band_cull.py` (covers every live pixel with the SFPU log /
exp error bounds, matches the box-min mask off the thr boundary, drops
invisible splats).

## Evidence

- Mask check on the first frame (3,369,033 pairs, GSPLAT_TT_DUMP_CULL):
  box-min kernel kept 8,033,113 microblocks, band kernel 8,033,112; 1 bit only
  in the band mask and 2 only in the box-min mask, all three with the nearest
  pixel centre >= 0.047 m2 beyond the live threshold (peak alpha*255 < 1). All
  non-mask slab words byte-identical (the write-back preserves the record).
- 30-view `--dump-views` md5: all 30 identical to the base; hero_vs_ref 100.00 dB.
- Timing: see the ledger row (iters.jsonl) and the table below.

## Scripts

`remote_build_dump.sh` (build + one render with the cull dump and 30-view dump),
`remote_ab.sh` (3 interleaved A/B rounds), `remote_tracy.sh` (one 10-view Tracy
chunk + zone table). All run on the remote under
`ttp lock p100 -- devrun.sh --no-verify ...`.
