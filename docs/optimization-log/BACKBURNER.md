# BACKBURNER

Parked experiments — REJECT or NEEDS_REVIEW iters that the user may want to promote.

## iter-001-dst-resident-state — REJECT

- Class: `kernel-algebra`
- kernel ms: median 90.91 / p99 94.93
- PSNR per view: hero 3.5 / side 3.9 / top 4.5
- Validator reasoning: Layer A Dst-resident refactor produced a 9% kernel-ms win (99.95 → 90.91 ms) but PSNR collapsed to 3.5–4.5 dB with classic accumulator-corruption tile-grid artifacts. The hypothesis stated this should be bit-identical; the worker's implementation introduced a coding bug (Dst slot reuse without proper re-init between Gaussians). REJECTED for catastrophic visual failure, not the architectural idea — the proper retry is **M3** (DST-resident state, **on top of** the FPU-heavy basis-form foundation from M1+M2), which has a clear DST slot layout that doesn't repeat the previous mistake.
- Thumbnails: ![hero](screenshots/iter-001-dst-resident-state/hero.png) ![diff10](screenshots/iter-001-dst-resident-state/hero_diff10.png)

## iter-002-basis-form-tile-local — RECLASSIFIED → BUILDING BLOCK (no longer in backburner)

- Class: `kernel-algebra`
- kernel ms: median 106.16 / p99 112.65 (slightly slower than baseline)
- PSNR per view: hero 47.4 / side 51.0 / top 44.7 (perceptually clean)
- **2026-05-25 reclassification:** Under the new 40 dB perceptual PSNR floor (see `docs/superpowers/specs/2026-05-25-fpu-heavy-architecture.md`), iter-2 passes its visual + numeric gates. It was previously rejected only because of the now-retired 100 dB kernel-algebra floor. iter-2 is structurally required for the FPU-heavy 16×16-face end state: basis-form rewrites the Gaussian quadratic into FPU-friendly `mul_tiles(basis_cb, gauss_cb, …)` FMAs.
- Action: KEEP but `no_commit_valid_but_not_faster` — kernel stays at iter-0 for "current best ms" purposes, but the basis-form approach is the foundation for milestones M1-M4 of the FPU-heavy roadmap.
- Thumbnails: ![hero](screenshots/iter-002-basis-form-tile-local/hero.png) ![diff10](screenshots/iter-002-basis-form-tile-local/hero_diff10.png)
