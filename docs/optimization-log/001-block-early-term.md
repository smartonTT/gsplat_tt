# Iter 001 — block-early-term

- **Idea**: Block-wide early termination — when all 1024 pixels in a 32×32 tile have `T < 1e-4`, skip SFPU stages A–E for remaining Gaussians (still pop `CB_SCALARS` for reader flow control).
- **Hypothesis**: Many interior stitch tiles saturate before exhausting their sorted Gaussian list; skipping ~1500 SFPU cycles per skipped Gaussian should cut kernel ms materially on stitch_hero.
- **Branch**: `opt/001-block-early-term`
- **Worker model**: composer-2.5-fast
- **Decision**: **REVERTED — no perf win** (kernel +1.4%, no visual regression). Per the optimization-loop gating policy, a change with no speedup is reverted even when bit-identical, because it adds maintenance burden and SFPU overhead with no upside. The kernel code on `smarton/optimization` is unchanged from iter 000; only this markdown (and the empty amplified-diff PNG) are kept as a record of the negative result. The full implementation lives on branch `opt/001-block-early-term` (commit `013d36e`) for future reference.

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
  - **Why it doesn't fire**: most screen tiles in stitch_hero include some empty background pixels (T stays at 1.0 forever — no Gaussian ever touches them). `sat_mask` for those pixels is always 1, so the per-tile SUM is always > 0, so `block_dead` never gets set. Block-wide early termination only helps tiles that are entirely *inside* an opaque object — rare in stitch_hero (a furry plush against black background).
- **False-positive trap (debugging)**: Comparing `read_tile_value` as `float < 0.5` treats bf16 `1.0` (`0x3c00` in the low 16 bits) as ~1e-4 and incorrectly marked every tile dead → bogus **33 ms** kernel time and PSNR 14 dB. **Do not use float casts on raw `read_tile_value` for bf16 tiles** — read the raw bit pattern as uint32 and compare bit patterns, or unpack the bf16 high half explicitly.
- **Build on bh-30**: `ninja` CMake regen failed (permissions / VERSION). Rebuilt via manual `clang++` compile+link of `alpha_blend.cpp` only (kernel `.cpp` is JIT'd at daemon start, so the kernel changes still got exercised). Supervisor TODO: figure out a clean rebuild path on bh-30 so future iters don't hit this.

## Lessons for the loop

1. **Negative result, useful insight.** Block-wide early termination doesn't pay off when scenes have empty background tiles. Future iters that go after the "empty pixel" cost should target the **host-side tile assignment** (cull Gaussians that don't actually overlap a tile) — kill the work before it reaches the device.
2. **The per-iter test scene biases what we can detect.** stitch_hero may not be the right scene to validate block-wide early termination — try strawberry (which has dense opaque regions) on the next attempt at this idea, or revisit after the kernel has been speedup'd enough that the reduce overhead becomes negligible.
3. **bf16 packing matters.** Reads from CBs are not "fp32 floats" — they're raw 32-bit words containing 1 or 2 packed bf16 values. Future iters that read from CBs must remember this.

## Next

Move to a different optimization. Top candidates per supervisor analysis:
- Tighten per-Gaussian-per-tile assignment on the host side so background tiles get 0 Gaussians (zero kernel cost for empty tiles).
- Reduce SFPU spill/reload by fusing more stages into single acquire blocks (Dst is large enough on Blackhole to hold more state).
- Replace per-pixel `sat_mask` multiplications with an SFPU predicate / cmov so saturated pixels skip the per-Gaussian update entirely (no zero-multiply work).
