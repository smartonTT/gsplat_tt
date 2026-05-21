# Iter 010 — B3b1 fold (add_binary_tile in B2+B3a)

- **Idea**: Eliminate standalone Stage B3b1 by pre-summing `a·dx² + c·dy²` inside Stage B2+B3a via a single SFPU `add_binary_tile` after FPU muls, then pack 2 tiles to CB_Q instead of 3.
- **Hypothesis**: Save one acquire block + CB_POWER traffic + one FPU `add_tiles` + one pack per Gaussian; net +50–200 cycles/Gaussian after paying one SFPU add → ~1–3% kernel speedup.
- **Branch**: `opt/010-b3b1-fold`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Only `alpha_blend_compute.cpp`:

- Added `#include "api/compute/eltwise_binary_sfpu.h"`.
- **Stage B2+B3a**: after three FPU mul+scale ops in dst[0..2], chain `add_binary_tile_init()` + `add_binary_tile(0, 1, 0)` to fold partial Q sum into dst[0]; pack dst[0] and dst[2] only (2 tiles) to CB_Q.
- **Deleted Stage B3b1** entirely (no CB_POWER traffic).
- **Stage B3b2+C**: opening add changed from `add_tiles(CB_POWER, CB_Q, 0, 2, 0)` to `add_tiles(CB_Q, CB_Q, 0, 1, 0)`; cleanup pop `cb_pop_front(CB_Q, 2)` (was CB_POWER+3×CB_Q).

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 57.97 | **57.31** | **−1.14%** | 44.00 dB | 0.9876 |

Daemon RT median: 110.12 ms. All 10 timed frames: stable 57.31 ms `device_kernel`.

## Visual gate

- **exit 0 (clean-keep)** vs original baseline: PSNR 44.00 dB, SSIM 0.9876; max-abs-diff R16/G19/B28; mean-abs-diff 1.07 LSB.
- No NaN/Inf.

## Screenshots

- Render: `docs/optimization-log/screenshots/010_stitch_hero_after.png`
- `docs/optimization-log/010-amplified-diff.png`

## Notes

- **Measurable win:** −0.66 ms (−1.14%) vs iter-008 baseline 57.97 ms — exceeds gating threshold (>0.5 ms and >1%).
- **Safe SFPU pattern confirmed:** Single `add_binary_tile` chained after FPU output in the same acquire, then immediate pack — no dst-slot replication, no subsequent FPU ops on the SFPU-touched slot within the block. Contrasts with iter 009 (`copy_dest_values` garbage) and iter 006 (3 SFPU adds/Gaussian net loss).
- **CB_Q traffic reduced:** 3 tiles → 2 tiles per Gaussian; CB_POWER unused but host allocation unchanged.

### add_binary_tile calibration (fused-with-FPU context)

Frame-level (832,049 sorted entries, stitch hero):

| Metric | Value |
|---|---|
| Net kernel Δ | −0.66 ms |
| Iter-006 scale (3 ops → +1.33 ms) | ~+0.44 ms per `add_binary_tile`/frame |
| Implied B3b1 elimination gross save | ~1.10 ms/frame (0.66 + 0.44) |
| Per-call `add_binary_tile` (scaled from 006) | **~630 cycles** @ 1.2 GHz / entry |

**Interpretation:** One `add_binary_tile` in this fused context costs roughly the same as in iter 006's D2 attempt (~640 cycles/op when scaled). It is **worth it** when it eliminates a full acquire block + CB round-trip + FPU add (~1.1 ms/frame here). It is **not worth it** when added per-channel on top of existing FPU paths (iter 006: 3× cost with no acquire savings → +1.33 ms).

## Next

Consider further B-stage fusions that eliminate acquire blocks (not per-channel SFPU adds). CB_POWER may become removable from host in a later iter once confirmed unused elsewhere.
