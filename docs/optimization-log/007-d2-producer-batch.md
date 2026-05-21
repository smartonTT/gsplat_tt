# Iter 007 — Stage D2 producer batch

- **Idea**: Collapse Stage D2's three per-channel producer acquire blocks into one batched acquire that writes contrib·color_r/g/b to three dedicated scratch CBs (CB_DX2/DY2/DXDY aliases), keeping FPU `add_tiles` adders unchanged.
- **Hypothesis**: Save 2 tile_regs acquire/commit/wait/release cycles and 2 CB_T_TMP push/pop pairs per Gaussian → ~1.5–3% kernel speedup with no new SFPU ops.
- **Branch**: `opt/007-d2-producer-batch`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Only `alpha_blend_compute.cpp`:

- Added kernel-only aliases `CB_T_R = CB_DX2`, `CB_T_G = CB_DY2`, `CB_T_B = CB_DXDY`.
- Stage D2: one batched producer (`copy_tile` contrib → dst[0..2], three `mul_unary_tile`, pack to CB_T_R/G/B).
- Three FPU adder blocks unchanged except read/pop from CB_T_R/G/B instead of shared CB_T_TMP.

## Host CB verification

From `alpha_blend.cpp`:

```cpp
cb_tile(CB_DX2, 2);
cb_tile(CB_DY2, 2);
cb_tile(CB_DXDY, 2);
```

- **Depth**: 2 (≥ 1) per scratch CB.
- **Format**: `Float16_b` (bf16), page size = `TILE_BYTES_BF16` (1 tile).
- **Dataflow kernels**: no reads/writes of CB_DX2/DY2/DXDY (grep over `kernels/dataflow/` — zero hits).
- **Compute kernel**: CB_DX2/DY2/DXDY unused since iter 003 (Stage B2+B3a collapse); CB_T_TMP still used by Stage F.

No host change required.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 61.03 | **59.10** | **−3.2%** | 44.10 dB | 0.9885 |

Daemon RT median: 111.67 ms (vs ~114 ms baseline). All 10 timed frames: stable 59.1 ms `device_kernel`.

## Visual gate

- **exit 0 (clean-keep)**: PSNR 44.10 dB, SSIM 0.9885 vs original baseline; max-abs-diff R17/G19/B29; mean-abs-diff 1.05 LSB.
- No NaN/Inf.

## Screenshots

- Render: `docs/optimization-log/screenshots/007_stitch_hero_after.png`
- `docs/optimization-log/007-amplified-diff.png`

## Notes

- **Measurable win:** −1.93 ms (−3.2%) vs iter-005 baseline 61.03 ms — exceeds gating threshold (>0.5 ms and >1%).
- **Contrast with iter 006:** SFPU `add_binary_tile` fusion regressed +1.33 ms; this iter only batches FPU/SFPU producer ops inside one acquire — no new SFPU binary ops, adders stay FPU `add_tiles`.
- **3× `copy_tile` observation:** Three sequential `copy_tile(CB_CONTRIB, 0, k)` into dst[0..2] share one `copy_tile_to_dst_init_short` and likely serialize on the same unpack path (same CB tile, three dst slots). Total copy work is unchanged vs three separate acquire blocks; the win comes from fewer acquire/release cycles and dedicated scratch CBs eliminating CB_T_TMP reuse (2 fewer push/pop pairs on the shared scratch CB).
- **Surprise:** Win landed at the high end of the 1.5–3% estimate despite conservative expectations.

## Next

Consider batching the three FPU adders (future iter) or dst-resident color accumulators if D2 adder acquires remain visible in profile.
