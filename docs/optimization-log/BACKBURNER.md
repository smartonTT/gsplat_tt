# BACKBURNER

Parked experiments — REJECT or NEEDS_REVIEW iters that the user may want to promote.

_None yet._

## iter-001-dst-resident-state — REJECT 

- Class: `kernel-algebra`
- kernel ms: median 90.91 / p99 94.93
- PSNR per view: hero 3.5 / side 3.9 / top 4.5
- Validator reasoning: Layer A Dst-resident refactor produced a 9% kernel-ms win (99.95 → 90.91 ms) but PSNR collapsed from infinity to 3.5-4.5 dB across all views. The renders show classic accumulator-corruption tile-grid artifacts: 32x32 colored blocks (magenta, blue, saturated white) tiling the image. The hypothesis stated this should be bit-identical to iter-0; instead the storage relocation altered the math. Likely cause: Dst[0..3] is being reused across Gaussians without the proper unpacker-side init that the prior CB round-trip provided, so subsequent Gaussians compute against stale or uninitialized accumulator state. Kernel-algebra class requires >100 dB; observed 3.5 dB. REJECT — revert kernel edit, BACKBURNER with note that Dst-resident state requires a different init pattern than the assumed equivalence.
- Thumbnails: ![hero](screenshots/iter-001-dst-resident-state/hero.png) ![diff10](screenshots/iter-001-dst-resident-state/hero_diff10.png)


## iter-002-basis-form-tile-local — REJECT 

- Class: `kernel-algebra`
- kernel ms: median 106.16 / p99 112.65
- PSNR per view: hero 47.4 / side 51.0 / top 44.7
- Validator reasoning: Layer B basis-form refactor rendered visually correct (no tile seams, no NaN, no geometry shift), but PSNR landed at 44-51 dB across views — far below the kernel-algebra 100 dB floor. Worker's own root-cause analysis: basis tiles stored/fetched as bf16; the 6-step mul_tiles FPU chain introduces ~4-7 bits of accumulated rounding error in the polynomial evaluation that did not exist in the original SFPU sub_unary path. Also slower than baseline (106.17 vs 99.96 ms, +6.2%) — CB overhead of the 6 mul_tiles + Q accumulation outweighed the SFPU→FPU substitution gain. Per spec §2: if basis-form drifts below 100 dB after the tile-local fix, BACKBURNER, never the kernel. REJECT — revert. Tile-local recentering itself looked numerically fine (no 14 dB cliff like iter-057), so the basis-form approach is recoverable if the basis tiles can be stored fp32 end-to-end or if the accumulation can be reorganized to keep the dominant terms in fp32 SFPU.
- Thumbnails: ![hero](screenshots/iter-002-basis-form-tile-local/hero.png) ![diff10](screenshots/iter-002-basis-form-tile-local/hero_diff10.png)

