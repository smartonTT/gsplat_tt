# Iter 001 — block-early-term

- **Idea**: Block-wide early termination — when all 1024 pixels in a 32×32 tile have `T < 1e-4`, skip SFPU stages A–E for remaining Gaussians (still pop `CB_SCALARS` for reader flow control).
- **Hypothesis**: Many interior stitch tiles saturate before exhausting their sorted Gaussian list; skipping ~1500 SFPU cycles per skipped Gaussian should cut kernel ms materially on stitch_hero.
- **Branch**: `opt/001-block-early-term`
- **Worker model**: composer-2.5-fast
- **Decision**: clean-keep

## Code diff

After each Stage F `sat_mask` refresh (every 16 Gaussians, `g > 0`), run `PoolType::SUM` / `ReduceDim::REDUCE_SCALAR` with `enforce_fp32_accumulation=true` over `CB_SAT_MASK`. If the reduced scalar is exactly zero (`read_tile_value == 0`), set per-tile `block_dead` and only `cb_wait_front`/`cb_pop_front` on `CB_SCALARS` for the rest of the tile. Added `CB_CONST_ONE` (index 11) as a pre-filled 1.0 reduce-scaler tile; host allocates it in `alpha_blend.cpp`. Restores `binary_op_init_common` after each `reduce_uninit` so eltwise ops stay valid.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs prev | PSNR vs baseline | SSIM vs prev | SSIM vs baseline |
|---|---|---|---|---|---|---|---|---|
| stitch | hero | 68.92 | 69.91 | +1.4% | ∞ | ∞ | 1.0000 | 1.0000 |

Daemon RT median: 122.15 ms (baseline 121.95 ms). End-to-end fps: 3.14 (baseline 3.20).

## Screenshots

- Render: `/tmp/iter001_stitch_hero.png` on bh-30 (bit-identical to baseline)
- `docs/optimization-log/001-amplified-diff.png` (empty — PSNR ∞)

## Notes

- **No kernel speedup on stitch_hero**: `block_dead` never fired with correct detection (`sat_sum_bits == 0`). Tiles still have at least one active pixel at every Stage F checkpoint in this scene/view, so we pay a small overhead for the SUM-reduce + `binary_op_init_common` restore every 16 Gaussians (~+1% kernel ms).
- **False-positive trap (debugging)**: Comparing `read_tile_value` as `float < 0.5` treats bf16 `1.0` (`0x3c00` in the low 16 bits) as ~1e-4 and incorrectly marked every tile dead → bogus **33 ms** kernel time and PSNR 14 dB. Do not use float casts on raw `read_tile_value` for bf16 tiles.
- **Build on bh-30**: `ninja` CMake regen failed (permissions / VERSION). Rebuilt via manual `clang++` compile+link of `alpha_blend.cpp` only (kernel `.cpp` is JIT).
- **Next**: Reader-signaled early termination or per-Gaussian block check may be needed; compute-only scalar reduce at `g % 16` does not see fully saturated tiles on stitch_hero.
