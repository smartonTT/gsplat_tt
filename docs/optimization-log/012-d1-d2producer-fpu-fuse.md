# Iter 012 — D1 + D2 producer fuse (FPU `mul_tiles` recompute)

- **Idea**: Fuse Stage D1 (contrib = α·T_state) and Stage D2 producer into one acquire block by recomputing α·T via 4 FPU `mul_tiles` calls (one per dst slot), avoiding iter-009's broken `copy_dest_values` SFPU path.
- **Hypothesis**: Save 1 acquire/commit/wait/release (~50–100 cycles) per Gaussian; FPU work approximately equal (4 mul_tiles vs 1 mul_tiles + 3 copy_tile) → ~0.9–1.5% kernel speedup (~55.7–56.0 ms).
- **Branch**: `opt/012-d1-d2producer-fpu-fuse`
- **Worker model**: composer-2.5-fast
- **Decision**: **reverted-no-win** (visual clean, perf flat/regression)

## Code diff (attempted, reverted)

Only `alpha_blend_compute.cpp` (reverted to `1318f3c`):

- Deleted standalone Stage D1 acquire block.
- Replaced Stage D2 producer with fused single-acquire block:
  - `mul_tiles(CB_ALPHA, CB_T_STATE, 0, 0, 0..3)` × 4 → dst[0..3] = contrib
  - `mul_unary_tile(1/2/3, color_r/g/b_bits)` → dst[1..3] = contrib · color
  - Pack dst[0] → CB_CONTRIB, dst[1..3] → CB_T_R/G/B
- No `copy_dest_values`, no host changes.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 56.54 | **56.90** | **+0.64%** | 44.01 dB | 0.9876 |

Daemon RT median: 122.79 ms. All 10 timed frames: stable 56.9 ms `device_kernel`.

Per-frame kernel ms: 56.9, 56.9, 56.9, 56.9, 56.9, 56.9, 56.9, 56.9, 56.9, 56.9.

## Visual gate

- **exit 0 (clean-keep)** vs original baseline: PSNR 44.01 dB, SSIM 0.9876; max-abs-diff R16/G19/B28; mean-abs-diff 1.07 LSB.
- Identical metrics to iter-011 baseline. No NaN/Inf.
- **4× FPU `mul_tiles` → 4-dst-slot pattern compiled and ran cleanly** (JIT, no compile errors).

## Screenshots

- Render: `docs/optimization-log/screenshots/012_stitch_hero_after.png`
- `docs/optimization-log/012-amplified-diff.png`

## Notes

- **No measurable win:** +0.36 ms (+0.64%) vs iter-011 baseline 56.54 ms — fails gating threshold (>0.5 ms or >1% improvement required). **REVERT** per no-win policy.
- **Correctness confirmed:** Unlike iter 009 (`copy_dest_values`), the FPU-only recompute path produces bit-identical visual output to baseline. The fusion mechanism works; it is just not faster.
- **Calibration observation:** Replacing 1 acquire + 3 `copy_tile` with 3 extra `mul_tiles` yielded a **small regression**, not the expected speedup. This suggests:
  1. The saved acquire overhead (~50–100 cycles estimated) is **smaller than the cost of 3 additional FPU `mul_tiles`** (~90 cycles each if not pipelined).
  2. The 4× recomputed `mul_tiles` did **not** pipeline for free — net FPU work dominated. If they had pipelined heavily, we would expect ~flat or faster; instead we got +0.36 ms.
  3. FPU `copy_tile` from CB_CONTRIB is likely **cheaper than a full `mul_tiles`** re-read of CB_ALPHA + CB_T_STATE, despite similar ~30-cycle estimates per op.
- **Net cycle accounting (revised):** Old = 2 acquires + 1 mul + 3 copy + 3 mul_unary. New = 1 acquire + 4 mul + 3 mul_unary. The 3 copy→mul swaps plus 1 extra mul cost more than 1 acquire saves.

## Next

Do not merge. D1+D2 producer fusion via FPU recompute is exhausted for this bottleneck (correct but not faster). Remaining acquire-fusion candidates should avoid increasing FPU tile-op count — e.g. dst-resident accumulators that eliminate CB round-trips entirely.
