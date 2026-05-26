# iter-012 — Per-tile init fuse: 5 fill_tile acquires → 1

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-010 e-fuse-fpu (KEEP, 97.77 ms).

## Hypothesis

At the top of every `for (t = 0; t < num_tiles; t++)` iteration, the kernel
performs **5 independent `fill_tile` acquires** to initialize the per-tile
state CBs:

```
acquire { fill_tile(0, 0.0f); }  → pack CB_COLOR_R_STATE   // 1
acquire { fill_tile(0, 0.0f); }  → pack CB_COLOR_G_STATE   // 2
acquire { fill_tile(0, 0.0f); }  → pack CB_COLOR_B_STATE   // 3
acquire { fill_tile(0, 1.0f); }  → pack CB_T_STATE         // 4
acquire { fill_tile(0, 1.0f); }  → pack CB_SAT_MASK        // 5
```

DST has 4 fp32 slots (fp32_dest_acc_en). Fill 0.0 into slots 0/1/2 and 1.0
into slot 3 in a single acquire, then pack each slot to its corresponding
CB. CB_SAT_MASK reuses dst[3] (same 1.0 value as CB_T_STATE) — pack the
same slot twice.

```
acquire {                       // 4 dst slots, R/G/B/T
  fill_tile(0, 0.0f);           // R
  fill_tile(1, 0.0f);           // G
  fill_tile(2, 0.0f);           // B
  fill_tile(3, 1.0f);           // T
}
commit, wait;
pack 0 → CB_COLOR_R_STATE
pack 1 → CB_COLOR_G_STATE
pack 2 → CB_COLOR_B_STATE
pack 3 → CB_T_STATE
release;

acquire {                       // separate dst for sat_mask
  fill_tile(0, 1.0f);
}
commit, wait;
pack 0 → CB_SAT_MASK
release;
```

Saves **3 acquire/commit/wait/release roundtrips per screen tile** (5 → 2).
With ~16 tiles/core and 64 cores, that's ~3072 saved acquire cycles per
frame. Per-core saving ~2-3 μs at an estimated 30-50ns each.

**First attempt history (0592a23, reverted):** tried 1-acquire form
packing dst[3] to BOTH CB_T_STATE and CB_SAT_MASK. The kernel hung at
cycle 1 (daemon blocked in futex_wait_queue_me holding /dev/tenstorrent/0).
Packing the same dst slot to two different CBs in one acquire isn't
supported — pack_tile must consume the slot or pack hardware needs
reconfig between CBs.

## Why this is low-PSNR-risk

- Pure `fill_tile` op — no FPU/SFPU math, no DEST_TO_SRCA reuse, no
  bf16 truncation between ops. The bytes written to each CB are
  bit-identical to the 5-acquire form.
- pack_tile is read-only on the source dst slot, so packing dst[3]
  twice (once to CB_T_STATE, once to CB_SAT_MASK) is safe. This pattern
  is already used elsewhere in tt-metal (e.g., constant-tile broadcast).
- No new CB depth requirements: each CB is still at depth 1, still
  reserved/pushed exactly once.

## Validation gate

- PSNR per view: bit-identical to iter-010 (40.40 / 43.70 / 40.15 dB or
  within ±0.1 dB rounding noise from differing pack ordering, which
  shouldn't matter since the data is constant).
- All 8 visual checks must pass with diff10 dominated by the iter-010
  noise floor.
- kernel ms ≤ prev_best (97.77 ms). Expected win: small (0.2-0.5 ms),
  but pure overhead reduction with no PSNR cost.

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Replace lines 142-188 (5 acquire blocks for R/G/B/T/SAT_MASK init)
    with a single fused acquire that fills 4 dst slots and packs to 5 CBs.

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores iter-010 state.
