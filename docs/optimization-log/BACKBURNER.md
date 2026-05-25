# BACKBURNER

Parked experiments — REJECT or NEEDS_REVIEW iters that the user may want to promote.

_None yet._

## iter-001-dst-resident-state — REJECT 

- Class: `kernel-algebra`
- kernel ms: median 90.91 / p99 94.93
- PSNR per view: hero 3.5 / side 3.9 / top 4.5
- Validator reasoning: Layer A Dst-resident refactor produced a 9% kernel-ms win (99.95 → 90.91 ms) but PSNR collapsed from infinity to 3.5-4.5 dB across all views. The renders show classic accumulator-corruption tile-grid artifacts: 32x32 colored blocks (magenta, blue, saturated white) tiling the image. The hypothesis stated this should be bit-identical to iter-0; instead the storage relocation altered the math. Likely cause: Dst[0..3] is being reused across Gaussians without the proper unpacker-side init that the prior CB round-trip provided, so subsequent Gaussians compute against stale or uninitialized accumulator state. Kernel-algebra class requires >100 dB; observed 3.5 dB. REJECT — revert kernel edit, BACKBURNER with note that Dst-resident state requires a different init pattern than the assumed equivalence.
- Thumbnails: ![hero](screenshots/iter-001-dst-resident-state/hero.png) ![diff10](screenshots/iter-001-dst-resident-state/hero_diff10.png)

