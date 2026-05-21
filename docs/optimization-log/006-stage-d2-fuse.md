# Iter 006 — Stage D2 fuse (add_binary_tile)

- **Idea**: Collapse Stage D2 color accumulation from 6 acquire blocks (2 per R/G/B channel) to 3 acquire blocks (1 per channel) by fusing producer+adder in-register via SFPU `add_binary_tile`.
- **Hypothesis**: Eliminate 3 acquires + 3 CB_T_TMP push/pop cycles per Gaussian → ~5–7% kernel speedup (~56–58 ms).
- **Branch**: `opt/006-stage-d2-fuse`
- **Worker model**: composer-2.5-fast
- **Decision**: **reverted-no-win** (visual clean-keep, kernel regressed +1.33 ms)

## Variant chosen

**Variant A** — `add_binary_tile(idst0, idst1, odst)` exists in `eltwise_binary_sfpu.h`. Fused each channel into one acquire: load `color_c_state` → dst[0], load `contrib` → dst[1], `mul_unary_tile(1, color_c_bits)`, `add_binary_tile(0, 1, 0)`, spill back. Zero CB_T_TMP traffic in D2.

## API discovery (Step 1)

Grep paths: `hw/inc`, `include` (missing), `api`.

### dst-to-dst binary ops (`eltwise_binary_sfpu.h`)

```cpp
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst);
ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst);
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst);
ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst);
ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst);
// + power/eq/ne/lt/gt/le/ge variants; each has matching *_init()
```

Note: iter-004 log said `mul_binary_tile` absent (grep over full tree); it **is** present under `hw/inc/api/compute/eltwise_binary_sfpu.h` (SFPU path, not FPU `eltwise_binary.h`).

### dst-to-dst copy (`copy_dest_values.h`)

```cpp
template <DataFormat DATA_FORMAT>
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out);
ALWI void copy_dest_values_init();
```

### FMA / accumulate-from-CB

- `add_tiles_to_dst`: **not found**
- `dest_acc`: **not found**

### Other relevant

```cpp
template <DataFormat data_format>
ALWI void addcmul_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst, uint32_t value);
// odst = idst0 + (value * idst1 * idst2) — not applicable to color_c · contrib without squaring contrib
```

No hits for: `binary_op_dst_to_dst`, `dst_add_tile`, `dst_mul_tile`, `copy_dst_to_dst`, `copy_dest_to_dest`.

## Code diff (attempted, reverted)

Only `alpha_blend_compute.cpp` (reverted to `64bc9e8`):

- Added `#include "api/compute/eltwise_binary_sfpu.h"`.
- Stage D2: per channel, single acquire with `copy_tile` state+contrib, `mul_unary_tile`, `add_binary_tile_init()` + `add_binary_tile(0,1,0)`, in-place spill. Removed CB_T_TMP producer/adder split.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 61.03 | **62.36** | **+2.2%** | 48.54 dB | 0.9892 |

Daemon RT median: 114.80 ms. All 10 timed frames: 62.3–62.6 ms `device_kernel` (stable regression vs 61.0–61.2 ms baseline).

## Visual gate

- **exit 0 (clean-keep)**: PSNR 48.54 dB, SSIM 0.9892 vs original baseline; max-abs-diff R16/G19/B29; mean-abs-diff 0.53 LSB.
- Correctness preserved; perf gate failed → **REVERT**.

## Screenshots

- Render: `/tmp/iter006_stitch_hero.png` on bh-30
- `docs/optimization-log/006-amplified-diff.png`

## Notes

- **No measurable win:** +1.33 ms (+2.2%) vs iter-005 baseline 61.03 ms. Gating policy requires revert despite clean visual.
- **`add_binary_tile` compiled cleanly** (JIT, no build errors). Mixing FPU `mul_unary_tile` + SFPU `add_binary_tile` in one acquire block is valid (same pattern as Stage B3b2+C with `binary_min_tile`).
- **Likely regression cause:** SFPU dst-to-dst add may be slower than FPU `add_tiles` from CB; per-channel `add_binary_tile_init()` adds overhead; loading two CB tiles into Dst (state + contrib) vs one (contrib-only producer) increases copy cost within the fused block. CB_T_TMP elimination did not compensate.
- **Surprise:** iter-004 noted `mul_binary_tile` absent — it exists in SFPU header; prior grep may have missed `hw/inc/api/compute/` path.
- **For iter 007+:** Consider `addcmul_tile` if a 3-tile dst layout can express `state + scalar*contrib` without SFPU binary add; or dst-resident accumulators to avoid per-channel CB reload; or FPU-only fusion (Variant B: bundle 3 producers, keep FPU adders).

## Next

Do not merge. Supervisor should pick alternate iter 006 strategy or skip to dst-resident / addcmul exploration.
