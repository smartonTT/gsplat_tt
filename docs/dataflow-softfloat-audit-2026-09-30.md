# Soft-float audit of the critical-path dataflow kernels (task #30, 2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views,
1024x1024, `python3 render/run.py --no-ref`. Base = `smarton/tt-project-opt` 7df44de.

BRISC/NCRISC (and the TRISCs' scalar code) have no FPU: every fp32 op is a libgcc
call, ~85-90 cycles for `__addsf3`/`__subsf3`/`__mulsf3` (`docs/hw-ceilings.md` p9).
Task #20 removed them from `sort_bin`'s emit; task #27 owns the rest of `sort_bin`.

## Inventory (JIT ELFs, `riscv-tt-elf-objdump -d`, static call sites)

| kernel (RISC) | stage | base call sites | where | after |
|---|---|---|---|---|
| `gather_visible_scatter` (BRISC) | project (count + scatter passes) | `__addsf3` 4, `__subsf3` 4, `__gesf2` 16, `__lesf2` 4 | visibility test, **every gaussian, twice per frame** | `__subsf3` 2, `__lesf2` 2 (cold fallback only) |
| `tile_assign_bbox` (NCRISC) | tile_assign | `__addsf3` 2, `__subsf3` 2, `__mulsf3` 4, `__fixsfsi` 4, `__divsf3` 1, `__floatunsisf` 1 | AABB tile coords, every visible gaussian | `__addsf3` 1, `__mulsf3` 1, `__fixsfsi` 1 (fallback), `__divsf3`/`__floatunsisf` 1 each (once per program) |
| `tile_assign_scatter` (NCRISC) | tile_assign | `__addsf3` 1, `__subsf3` 2, `__mulsf3` 3, `__fixsfsi` 3, `__divsf3` 1, `__floatunsisf` 1 | AABB tile coords, every gaussian a core visits | same as bbox |
| `sort_subchunk_materialize` (NCRISC) | blend CQ | `__subsf3` 4, `__floatunsisf` 2 | tile-local mean, overflow records only | `__subsf3` 2 (fallback), `__floatunsisf` 2 (per tile) |
| `reader_pfwc`, `writer_pfwc`, `gather_scan_bases`, `tile_assign_scan_*`, `sort_radix_tile`, `sort_publish`, `reader_tile_l1_cull`, `writer_tile_l1_mask`, `reader_alpha_blend_mb_devcull`, `writer_alpha_blend` | all | none | | |
| `sort_bin` (BRISC+NCRISC) | sort | `__addsf3` 4, `__mulsf3` 4, `__subsf3` 2, compares 8, converts 6 | emit pack | out of scope (task #27) |
| `microblock_cull_compute` TRISC1 | cull | `__subsf3` 128, `__addsf3` 2, `__divsf3` 1, ... | scalar code on the math RISC | not audited (compute kernel, see follow-up) |

Ranking by measured cost (below): gather visibility test >> tile_assign AABB >> materialize.

## Change

- `render/kernels/dataflow/dm_fp32.h` (new): fp32 on bit patterns with integer ops.
  IEEE compares (`lt`/`le`: NaN false, -0 == +0), RNE `add` (6 guard bits + sticky,
  fast path when both operands and the result are normal), `trunc_shr` (= `(int)(x * 2^-s)`).
  Outside the fast range the helpers fall back to the original float expression, so
  the result is the same for every input.
- `gather_visible_scatter`: the visibility test runs on bits; `mx + rx > 0` is the
  exact identity `-rx < mx` (a rounded sum is > 0 iff the exact sum is); cheap
  compares first so the two subtractions are skipped for rejected gaussians.
- `tile_assign_bbox` / `tile_assign_scatter`: `(int)((p ± r) * inv_tsf)` via
  `add_mul_pow2_to_int` (tile size 32 = 2^5, so the multiply is an exact shift).
- `sort_subchunk_materialize`: `mx - tile_x` via the existing `sort_bin_fp32::sub_int`.

Check: `tests/unit/test_dm_fp32.cpp` (standalone, build line in the file):
11.8e9 comparisons against native fp32 (all 2^32 patterns for `trunc_shr` at s = 0/4/5,
full sweeps of `a` against 20 fixed `b` incl. ±0, subnormals, ±inf, NaN, max;
1e9 random pairs incl. near-cancellation and pixel-scale means/radii): **0 mismatches**.

## Result

3 interleaved rounds per build in one device session:

| metric | base (7df44de) | t30 | delta |
|---|---:|---:|---:|
| `avg_frame_ms` | 145.47 (145.5 / 145.6 / 145.3) | **121.73** (121.7 / 121.8 / 121.7) | **-23.73 (-16.3%)** |
| FPS | 6.87 | 8.21 | |
| `stage_project` | 36.97 | 20.41 | -16.56 |
| `stage_tile_assign` | 21.77 | 14.75 | -7.02 |
| `stage_sort` | 18.77 | 18.76 | 0 |
| `stage_blend` | 63.58 | 63.60 | 0 (materialize change below noise) |

`hero_vs_ref = 100.00 dB`, hero md5 `e3fefb116d860f99d92bba1ef51d820c`, all 30 dumped
views byte-identical to the base build.

## Follow-ups

- `microblock_cull_compute` TRISC1 has 128 `__subsf3` call sites (plus add/div/mul);
  the cull runs 20+ ms per view on the TRISCs. Check whether any sit in a per-pair
  or per-microblock loop; same integer treatment if so.
- `alpha_blend_compute_mb` TRISC1: `__mulsf3` 4, `__floatunsisf` 4, `__gesf2` 3 (likely per tile).
- `project` is now 20.4 ms, of which `proj_count` + `proj_scatter` are two full passes
  over all gaussians; with the test now cheap, the passes are read/NoC-bound candidates
  for the dual-mover split (task #33) or a single fused count+scatter pass.

## Reproduce

```
# base tree and new tree built on the box, each with its own TT_METAL_CACHE_RENDER
for r in 1 2 3; do for v in t30base t30; do cd /localdev/smarton/gstt2-$v;
  TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-$v python3 render/run.py --iter-dir t30-ab-$v-$r --no-ref; done; done
```
Logs: `~/dev/gstt2/.ttw/logs/t30-*.log`.
