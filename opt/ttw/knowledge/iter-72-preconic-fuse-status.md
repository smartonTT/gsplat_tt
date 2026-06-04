# Iter 72 (ttw-072) — reader PRECONIC + wired MB_FUSE_TILE_L1_CULL

**Hypothesis:** Move conic (cov→A,B,C) from blend compute SFPU to reader emit + per-g scalar preconic on L1 bulk; shrink `blend_one_gaussian_math` LRA so fused `tile_l1_cull_sfpu` links. First l1_bulk subchunk runs fuse cull before tile_regs init (no DEST stash); later subchunks still load DRAM masks from standalone `tile_l1_cull` unless fuse env skips that pass.

**Env:** `MB_FUSE_TILE_L1_CULL=1` at program build (recreate blend context after toggling).

**Code:**
- `render/kernels/common/mb_cov_preconic.hpp` — scalar cov→ABC
- Reader: preconic coeff emit; box ramps CB 14/15; skip DRAM mask load when `!MB_FLAG_CONTINUE`
- Compute: PRECONIC blend; `fuse_l1_cull_subchunk` + kernel_main reorder
- Host: fuse CBs, box DRAM upload, skip `tile_l1_cull::process_frame` when fuse

## Device (yyzo-bh-07) — 2026-06-04 worker verify

**Tree:** Replaced broken `gstt2` → `proj_sw` symlink with real `/localdev/smarton/gstt2`; full rsync; `.venv` + pybind11 3.0.4; `.git` → `gsplat_tt/.git`. Host build fixes on device: GCC-safe `getenv`, `mb::cull::make_box_ramp` fwd-decl, `../common/mb_cov_preconic.hpp`, `#if MB_FUSE` around fuse call.

**Build:** `render_clean.so` rebuilt; `bin=cab7c12a51d5ffd5` (artifact + `render/kernels`).

| Attempt | MB_FUSE | JIT hash (blend compute) | TRISC0 seg[1] | Gate / outcome |
|---------|---------|--------------------------|---------------|----------------|
| 1 | 0 | `13186597283002392372` | **0x4** (≤0x720) | **reject** — `hero_vs_ref=63.39`, `ms_view=328.72` (<63.6; iter71=63.63) |
| 2 | 1 | `8755149977830927907` | **0x4** (≤0x720) | **blocked** — program config **109776 > 70656** at finalize (fuse CBs/args); JIT linked before host program size fail |

**LRA reference (cached):** iter-69 fuse hash `2850661745767925136` seg[1]=**0x768** (overflow). New wired-fuse hash `8755149977830927907` seg[1]=**0x4** — preconic freed LRA; new ceiling is **kernel-config buffer**, not TRISC0 local.

**Decision:** **reject** iter 72 (quality regression + fuse path blocked on program size). No Tracy (gate fail). Do not `iterate.sh --keep`.
