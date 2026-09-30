# Soft-float removal in the cull/blend TRISC compute kernels (task #39, 2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views,
1024x1024, `python3 render/run.py --no-ref`. Base = `smarton/tt-project-opt` cc9399c.
Also covers queued task #36 (integer record decode in cull/blend): same code.

The TRISCs' scalar code has no FPU, so every fp32 op there is a libgcc call
(~85-90 cycles for add/sub/mul, `docs/hw-ceilings.md`). The SFPU math is not
affected; only the scalar record decode that feeds it.

## Offenders (JIT ELFs, `riscv-tt-elf-objdump -d`, static call sites)

| kernel (thread) | base | where | after |
|---|---|---|---|
| `microblock_cull_compute` TRISC1 | `__subsf3` 129, `__addsf3` 3, `__floatunsisf` 4, `__mulsf3` 2, `__divsf3` 2, `__gesf2` 2 | per record: `mean + tile_origin` (2 adds) and `op = u16 * (1/65535)` (convert + mul) in the decode loop; `mx - txf`, `my - tyf` in each of `cull_face_x<V>`/`cull_face_y<V>` (2 x 2 x 32 unrolled = 128 subs) | `__addsf3` 9, `__subsf3` 9 (cold fallback only), `__floatunsisf` 3 / `__divsf3` 2 / `__gesf2` 2 (once per tile or core) |
| `alpha_blend_compute_mb` TRISC1 | `__floatunsisf` 5, `__mulsf3` 5, `__gesf2` 4 | per record: 4 x UNORM16 -> float (op, r, g, b); T-readback max/threshold compares (1024 per readback) | none |
| TRISC0 / TRISC2 of both | none | | none |

So per record the cull paid 8 soft-float calls (2 add + 4 sub + convert + mul) and
the blend 8 (4 x convert + mul), all on the thread that issues the SFPU work.

## Change

- `dm_fp32.h`: `unorm16_to_f(q)` = bits of `fl((float)q * (1.0f/65535.0f))`.
  `fl(1/65535) = 0x800080 * 2^-39`, so the exact product is `q * 65537 * 2^-32`
  (32-bit) rounded to 24 bits RNE. `add_sub_roundtrip(a, kb)` = bits of
  `fl(fl(a + k) - k)` via two `dm_fp32::add`; float fallback outside the fast range.
- cull: the decode computes `mlx = (mean + origin) - origin` once per record (it was
  recomputed in both faces) and passes it to the SFPU faces; decode runs on the
  MATH thread only. Same values reach the SFPU, so the keep masks are unchanged.
- blend: UNORM decode via `unorm16_to_f`, and only for records with a live mask;
  T-readback max/threshold on bits (`mbmax` stays a non-negative non-NaN float).

Check: `tests/unit/test_trisc_fp32.cpp` (build line in the file): all 65536 UNORM
codes, every fp32 pattern against 5 origins, dense and random sweeps: 26.8e9
comparisons against native fp32, **0 mismatches**.

## Result

3 interleaved rounds per build in one device session:

| metric | base (cc9399c) | t39 | delta |
|---|---:|---:|---:|
| `avg_frame_ms` | 121.78 (121.70 / 121.94 / 121.69) | **107.83** (107.83 / 107.80 / 107.86) | **-13.95 (-11.5%)** |
| FPS | 8.21 | 9.27 | |
| `stage_blend` (cull + blend programs) | 63.60 | 49.63 | -13.97 |
| other stages | unchanged | | |

`hero_vs_ref = 100.00 dB`, hero md5 `e3fefb116d860f99d92bba1ef51d820c`, all 30 dumped
views byte-identical to the base build. No Tracy capture in this task, so the split
of the 14 ms between the cull (`tile_mb_mask`) and blend (`tile_blend_sfpu`)
programs is not measured.

## Reproduce

Base tree `gstt2-t39base` (cc9399c) and new tree `gstt2-t39` on the box, each with
its own `TT_METAL_CACHE_RENDER`; scripts were `/tmp/t39/s1.sh` (build + dumps) and
`/tmp/t39/s2.sh` (A/B). Logs: `~/dev/gstt2/.ttw/logs/t39-*.log`.
