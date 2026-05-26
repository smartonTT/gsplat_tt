# iter-014 — Per-tile init fuse via multi-slot copy_tile from CB_CONST tiles

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: planned
- Predecessor: iter-010 e-fuse-fpu (KEEP, 97.77 ms / 40.40 / 43.70 / 40.15 dB).

## Hypothesis

At the top of every `for (t = 0; t < num_tiles; t++)` iteration, the kernel
performs **5 independent `fill_tile` acquires** to initialize the per-tile
state CBs (R/G/B = 0, T = 1, SAT_MASK = 1). This is the same fusion target
as iter-012, but iter-012 used multi-slot `fill_tile` in one acquire and
deadlocked.

iter-014 takes a **different path** that uses a proven kernel pattern:

1. Add a new `CB_CONST_ONE` startup constant alongside the existing
   `CB_CONST_ZERO` and `CB_CONST_099` (single fill_tile + pack at the top of
   kernel_main, never popped).
2. Replace the 5 per-tile init acquires with 2 acquires using
   **multi-slot `copy_tile`** (already proven safe at lines 595-602 and
   267-280 in the same kernel):

```cpp
// Acquire 1: 4 dst slots for R/G/B/T
tile_regs_acquire();
copy_tile_to_dst_init_short(CB_CONST_ZERO);
copy_tile(CB_CONST_ZERO, 0, 0);  // R
copy_tile(CB_CONST_ZERO, 0, 1);  // G
copy_tile(CB_CONST_ZERO, 0, 2);  // B
copy_tile_to_dst_init_short(CB_CONST_ONE);
copy_tile(CB_CONST_ONE, 0, 3);   // T
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(CB_COLOR_R_STATE, 1);
pack_tile(0, CB_COLOR_R_STATE);
cb_push_back(CB_COLOR_R_STATE, 1);
cb_reserve_back(CB_COLOR_G_STATE, 1);
pack_tile(1, CB_COLOR_G_STATE);
cb_push_back(CB_COLOR_G_STATE, 1);
cb_reserve_back(CB_COLOR_B_STATE, 1);
pack_tile(2, CB_COLOR_B_STATE);
cb_push_back(CB_COLOR_B_STATE, 1);
cb_reserve_back(CB_T_STATE, 1);
pack_tile(3, CB_T_STATE);
cb_push_back(CB_T_STATE, 1);
tile_regs_release();

// Acquire 2: sat_mask = 1 (separate acquire — DST has only 4 fp32 slots)
tile_regs_acquire();
copy_tile_to_dst_init_short(CB_CONST_ONE);
copy_tile(CB_CONST_ONE, 0, 0);
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(CB_SAT_MASK, 1);
pack_tile(0, CB_SAT_MASK);
cb_push_back(CB_SAT_MASK, 1);
tile_regs_release();
```

Saves **3 acquire/commit/wait/release roundtrips per screen tile** (5 → 2).

## Why this is different from iter-012

iter-012 tried multi-slot `fill_tile` in one acquire. The tt-metal SFPU fill
pipeline appears to require single-slot use — no working precedent existed
in tt-metal `programming_examples/` or `tests/`. Both variants hung.

**copy_tile multi-slot, by contrast, is already used in this very kernel**:

- Lines 595-602: 3 slots from 3 different CBs in one acquire, packed to one
  CB (per-tile finalize).
- Lines 267-280: 2 slots from 2 different CBs in one acquire, packed to 2
  different CBs (Stage B1).

So the pattern "load N slots via copy_tile, pack to N different CBs in one
acquire" is established to work.

## Why this is low-PSNR-risk

- Pure `copy_tile` from pre-populated constant CBs — no FPU/SFPU math, no
  `DEST_TO_SRCA` reuse, no bf16 truncation.
- Each CB still pushed/popped exactly once per tile, depth unchanged.
- The bytes written to each state CB are bit-identical to the 5-acquire form.

Expected PSNR: ∞ vs iter-010 reference (no FP differences at all).

## Perf impact

5 → 2 acquires per tile. With ~16 tiles per core and 64 cores active, that's
~3072 saved acquire/commit/wait/release roundtrips per frame. At an
estimated ~30-50 ns per roundtrip, total savings ~100-150 μs per frame, or
~0.1-0.15 ms on the 97.77 ms baseline.

Marginal but positive, and the new `CB_CONST_ONE` is a useful building
block for future iters.

## Validation gate

- PSNR per view: expected bit-identical to iter-010 (40.40 / 43.70 / 40.15
  dB or ∞ when compared against iter-010's render directly).
- All 8 visual checks must pass with diff10 dominated by the iter-010 noise
  floor (or be completely black if truly bit-identical).
- kernel ms ≤ prev_best (97.77 ms). Expected: 97.5-97.7 ms range.

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Add `constexpr uint32_t CB_CONST_ONE = 24;` near line 89.
  - Add startup `fill_tile(0, 1.0f) → CB_CONST_ONE` block after line 126
    (modeled on the existing CB_CONST_099 setup at lines 118-126).
  - Add `cb_wait_front(CB_CONST_ONE, 1);` after line 129.
  - Replace lines 142-188 (5 fill_tile acquires) with the 2-acquire form
    above.

Note: CB_CONST_ONE must also be declared at the host side. The compute
kernel uses CB indices declared in the host C++ binary. Need to check
where CBs are configured in `alpha_blend.cpp` and add a corresponding
buffer config.

## Rollback plan

`git revert <iter-014-commit>` restores iter-010 state.

## Why not iter-012's deadlocked form?

iter-012 used `fill_tile(idst=0..3)` in one acquire which hung. iter-014
uses `copy_tile` from pre-populated CBs (CB_CONST_ZERO, CB_CONST_ONE)
which is a different SFPU code path — copy_tile is fundamentally an
unpack-and-write-to-dst op, not a write-without-source op. Multi-slot
unpack-and-write is exercised elsewhere in the kernel.
