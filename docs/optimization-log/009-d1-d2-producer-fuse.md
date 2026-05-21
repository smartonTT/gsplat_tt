# Iter 009 — Stage D1 + D2 producer fuse (`copy_dest_values`)

- **Idea**: Fuse Stage D1 (contrib = α·T_state) and Stage D2 producer (contrib·color_r/g/b) into one acquire block, replacing three `copy_tile(CB_CONTRIB, …)` with three `copy_dest_values` dst-slot copies after `mul_tiles`.
- **Hypothesis**: Save 1 acquire/release/wait (~50–100 cycles) plus 3 SFPU `copy_tile` CB reads (~60–90 cycles) → ~1–2% kernel speedup (~56.8–57.4 ms).
- **Branch**: `opt/009-d1-d2-producer-fuse`
- **Worker model**: composer-2.5-fast
- **Decision**: **reverted-hard-reject** (broken output + +0.52 ms perf regression)

## API discovery (Step 1)

Header: `tt_metal/hw/inc/api/compute/copy_dest_values.h`

```cpp
template <DataFormat DATA_FORMAT>
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out);
ALWI void copy_dest_values_init();
```

- **Init**: `copy_dest_values_init()` required once before calls (same acquire block).
- **Runtime args**: `(idst_in, idst_out)` — source/dest dst slot indices; must be in acquired state via `tile_regs_acquire`.
- **DataFormat used**: `DataFormat::Float16_b` (matches all bf16 tile CBs in `alpha_blend.cpp`).
- **Constraints**: Blocking SFPU math op (`llk_math_eltwise_binary_sfpu_copy_dest_values`). Not gated behind a build flag. Deprecated non-template overload exists but should not be used.
- **Canonical usage** (ttnn `compute_mpwi.cpp`): copies **index tiles** with `DataFormat::UInt16` or `UInt32` after `copy_tile` into dst — not used after FPU `mul_tiles` in existing codebase.

No hits for `copy_dest_values_to_buf`, `dest_zero`, or `dest_unary_*` under `hw/inc/api/compute/`. Other dst helper: `transpose_wh_dest.h`.

## Code diff (attempted, reverted)

Only `alpha_blend_compute.cpp` (reverted to `82c4d8a`):

- Added `#include "api/compute/copy_dest_values.h"`.
- Fused D1+D2 producer: `mul_tiles` → dst[0]; `copy_dest_values<Float16_b>(0,1/2/3)`; `mul_unary_tile(1/2/3, color_*_bits)`; pack dst[0..3] to CB_CONTRIB + CB_T_R/G/B in one acquire.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 57.97 | **58.49** | **+0.9%** | **15.22 dB** | **0.4207** |

Daemon RT median: 110.97 ms. All 10 timed frames: stable 58.5 ms `device_kernel` (except warmup JIT).

Baseline re-verify after revert: 57.97 ms median, PSNR 44.10 dB / SSIM 0.9885 vs same reference PNG.

## Visual gate

- **exit 1 (hard-reject)**: PSNR 15.22 dB, SSIM 0.4207 vs original baseline; mean-abs-diff 31.0 LSB; max-abs-diff R155/G158/B182.
- Output severely corrupted (not subtle bf16 drift). **REVERT** per gating policy (also fails no-win perf gate: +0.52 ms).

## Screenshots

- Render: `docs/optimization-log/screenshots/009_stitch_hero_after.png`
- `docs/optimization-log/009-amplified-diff.png`

## Notes

- **`copy_dest_values` is not a free register move after FPU ops.** Despite being labeled a "copy" rather than compute, calling it on dst[0] immediately after FPU `mul_tiles` produced garbage in dst[1..3] (and thus wrong R/G/B accumulators). dst[0] packed to CB_CONTRIB may remain valid, but scaled copies fed CB_T_R/G/B with corrupted data → catastrophic visual failure.
- **Likely root cause:** FPU dst accumulator layout vs SFPU `copy_dest_value<Float16_b>` format mismatch. Canonical tt-metal usage copies integer index tiles (UInt16/UInt32), not FPU bf16 multiply results. Iter 006 showed SFPU dst-to-dst *compute* (`add_binary_tile`) regressed perf; iter 009 shows SFPU dst-to-dst *copy* breaks correctness when bridging FPU→SFPU dst slots.
- **Perf also regressed +0.52 ms** even before considering correctness — the primitive is neither cheap enough nor safe for this fusion pattern.
- **Safe alternative for D1+D2 fusion** would need FPU-native replication (e.g. keep CB round-trip via `copy_tile(CB_CONTRIB)`) or dst-resident accumulators that avoid the producer split entirely — not `copy_dest_values` across FPU/SFPU boundary.

## Next

Do not merge. Mark `copy_dest_values` fusion exhausted for Stage D1+D2. Consider dst-resident R/G/B/T accumulators or explore FPU-only single-acquire patterns without SFPU dst manipulation.
