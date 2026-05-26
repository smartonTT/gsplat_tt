# BACKBURNER

Parked experiments — REJECT or NEEDS_REVIEW iters that the user may want to promote.

## iter-001-dst-resident-state — REJECT

- Class: `kernel-algebra`
- kernel ms: median 90.91 / p99 94.93
- PSNR per view: hero 3.5 / side 3.9 / top 4.5
- Validator reasoning: Layer A Dst-resident refactor produced a 9% kernel-ms win (99.95 → 90.91 ms) but PSNR collapsed to 3.5–4.5 dB with classic accumulator-corruption tile-grid artifacts. The hypothesis stated this should be bit-identical; the worker's implementation introduced a coding bug (Dst slot reuse without proper re-init between Gaussians). REJECTED for catastrophic visual failure, not the architectural idea — the proper retry is **M3** (DST-resident state, **on top of** the FPU-heavy basis-form foundation from M1+M2), which has a clear DST slot layout that doesn't repeat the previous mistake.
- Thumbnails: ![hero](screenshots/iter-001-dst-resident-state/hero.png) ![diff10](screenshots/iter-001-dst-resident-state/hero_diff10.png)

## iter-002-basis-form-tile-local — REJECTED ON SECOND LOOK

- Class: `kernel-algebra`
- kernel ms: median 106.16 / p99 112.65 (slightly slower than baseline)
- PSNR per view: hero 47.4 / side 51.0 / top 44.7
- 2026-05-25 reclassified KEEP under 40 dB perceptual floor, with iter-2 declared "structurally required for the FPU-heavy 16×16-face end state."
- **2026-05-26 user correction:** the cross-hatch in diff10 is *actually visible* in iter-003's hero at the ear bottoms (basis-form propagated to M1). Visible artifacts at perceptual scale override the 40 dB numeric floor. Basis-form is NOT a safe building block as-is.
- Thumbnails: ![hero](screenshots/iter-002-basis-form-tile-local/hero.png) ![diff10](screenshots/iter-002-basis-form-tile-local/hero_diff10.png)

## iter-003-m1-basis-form-fpu-q — REJECTED ON SECOND LOOK (was committed, then reverted)

- Class: `kernel-algebra`
- kernel ms: median 97.55 / p99 102.64 (was a -2.4% speedup vs baseline)
- PSNR per view: hero 43.9 / side 46.9 / top 41.8
- Original verdict 2026-05-26 02:18 UTC: KEEP / commit (3d5b162).
- **2026-05-26 user correction (03:30 UTC):** visible tile-grid seams at ear bottoms in hero.png. Reverted in 044f398. Building-block status retracted.
- Root cause: basis-form Q sums 6 fp32 terms of magnitude ~10 to a small Q ~0.01 (≈17 bits of cancellation). The residual is then bf16-quantized when packing R/G/B/T state, and adjacent tiles diverge along tile boundaries → cross-hatch.
- Lesson: a building-block iter that passes the 40 dB floor but visibly damages renders is not actually a building block. Visual gate must hold without exception.
- Thumbnails: ![hero](screenshots/iter-003-m1-basis-form-fpu-q/hero.png) ![diff10](screenshots/iter-003-m1-basis-form-fpu-q/hero_diff10.png)

## iter-004-fuse-inner-acquires — REVERTED with iter-003

- Class: `kernel-algebra`
- kernel ms: median 99.67 (regressed from prev_best 97.55)
- PSNR per view: hero 43.9 / side 46.9 / top 41.8 (inherited from iter-003 M1)
- Built on iter-003 M1 (basis-form), so it inherited the tile-seam artifact and was reverted in the same rollback. The SFPU DST-DST fusion in D1+D2+E was also slower than the prior FPU mul_tiles approach, so even on its own merits it would not have landed.
- Thumbnails: ![hero](screenshots/iter-004-fuse-inner-acquires/hero.png) ![diff10](screenshots/iter-004-fuse-inner-acquires/hero_diff10.png)

## iter-009-b2-fuse-fpu — REJECT 

- Class: `kernel-algebra`
- kernel ms: median 100.69 / p99 106.17
- PSNR per view: hero 59.5 / side 62.3 / top 60.0
- Validator reasoning: Algebra-preserving fusion: PSNR per-view is bit-identical to iter-007 (59.518/62.260/59.995). Visual checks all pass with the same diff10 noise floor. However the kernel ms regressed 99.34 → 100.695 (+1.37%), which falls within the 2% break-even band but offers no improvement, while making the kernel less explicit (three live products in one acquire). Same class of negative result as iter-006: collapsing acquire cycles for FPU ops appears to remove overlap between pack and the next stage's setup, so theoretical 'saves 2 acquire cycles' becomes net-zero or slightly negative. REJECT with action=revert; lesson is that the 3-mul_tiles single-acquire pattern is not a perf win on this kernel and the explicit 3-acquire form is preferable.
- Thumbnails: ![hero](screenshots/iter-009-b2-fuse-fpu/hero.png) ![diff10](screenshots/iter-009-b2-fuse-fpu/hero_diff10.png)


## iter-011-b3b-fuse-fpu — NEEDS_REVIEW 

- Class: `kernel-algebra`
- kernel ms: median 97.90 / p99 102.94
- PSNR per view: hero 39.7 / side 43.1 / top 38.9
- Validator reasoning: Visual gate passes on all 8 checks: no tile seams, no uniform-fill blocks, no missing-splat holes, no clipping bands, no ringing, no NaN/Inf, no geometry shift, and diff×10 structure is uniform speckle consistent with bfloat16 quantization noise from the fused Q-summation change. However, two of three views fall below the 40 dB kernel-algebra PSNR floor: hero at 39.69 dB and top at 38.87 dB. Per spec, any view below the 40 dB floor for class kernel-algebra triggers NEEDS_REVIEW (not auto-REJECT since visuals are clean). Timing is essentially break-even: median 97.9 ms vs prev_best 97.77 ms (+0.13%), within the 1.02× tolerance. Per-view PSNR delta is only 4.20 dB (well under 20 dB threshold) and per-view ms ratio is 1.16× (well under 2×). The NEEDS_REVIEW verdict is driven solely by two views sub-floor — the iteration produces structurally clean output but the fused Q-summation path is introducing enough bfloat16 accumulation error to push hero and top below 40 dB, which warrants human review before promotion.
- Thumbnails: ![hero](screenshots/iter-011-b3b-fuse-fpu/hero.png) ![diff10](screenshots/iter-011-b3b-fuse-fpu/hero_diff10.png)

