# Iter 60 Tracy vs perf anchor iter 51 (e6d43d4)

**Captured:** 2026-06-04 on `yyzo-bh-03` via `devrun.sh --no-verify` + `render/profiler/capture_tracy_clean.sh` @ `c946556` (iter 60 keep, tile-local L1 cull). Artifacts: `opt/profiler/ttw-060-tracy/{profile_log_device.csv,render.tracy}`.

**Baseline:** `opt/profiler/ttw-051-tracy/profile_log_device.csv` (iter 51 anchor; no `sort_subchunk_*` zones — confirms pre–step-A capture).

## Host timing (30-view bench, no Tracy)

| Iter | commit   | ms/view (gate) | Δ vs 51 |
|------|----------|----------------|---------|
| 51   | e6d43d4  | 242.1          | —       |
| 55   | iter-55  | 270.7          | +28.6   |
| 60   | c946556  | 295.6          | +53.5   |

Tracy run on iter 60: **319.8 ms/view** (`avg_frame_ms` in capture log) — ~24 ms profiler/mid-run dump tax vs gate.

## Zone inventory (device START counts / 30 views)

| Zone | iter 51 | iter 60 |
|------|---------|---------|
| `cull_global_mb` | 10230 | — |
| `tile_l1_cull_rd` | — | 3410 |
| `tile_mb_mask` | — | 10230 |
| `sort_subchunk_mat` | — | 3410 |
| `sort_subchunk_dir` | — | 31 |
| `rd_l1_bulk` / `rd_bk_emit` / `cp_inb` / `tile_blend_sfpu` | unchanged counts | same |

Blend reader/compute zones flat (≤0.3 ms/view makespan delta).

## Top ms/view regressions (device makespan)

Method: pair ZONE_START/END; split timeline into **30 views** using `mcam` START markers (10230/30 = 341 per view); per view per zone **max over cores of summed span time** (stage makespan). Compare mean across views.

| Zone / bucket | 051 ms/view | 060 ms/view | Δ |
|---------------|------------|------------|---|
| **`sort_subchunk_mat`** | 0 | **27.2** | **+27.2** |
| **`tile_mb_mask`** (SFPU cull, was `cull_global_mb`) | **246.2** | **324.4** | **+78.2** |
| **`tile_l1_cull_rd`** (L1 bulk + sort for cull) | 0 | **108.1** | **+108.1** |
| `sort_subchunk_dir` | 0 | 2.4 | +2.4 |
| `cull_global_mb` | 246.2 | 0 | −246.2 |
| proj / ta / sort / `rd_*` / `cp_*` | — | — | ≈0 |

**Cull stage critical path (approx):** iter 51 `cull_global_mb` **246 ms** → iter 60 **max(`tile_mb_mask`, `tile_l1_cull_rd`) ≈ 324 ms** (+**~78 ms** SFPU), plus **+108 ms** reader if not overlapped with compute (3-kernel cull is reader→compute→writer; reader likely overlaps partially — host sees only **~+26 ms** 55→60 because mat already paid).

**Full 51→60 (+54 ms gate)** decomposes roughly as:

1. **+27 ms** — `sort_subchunk_mat` (step A/B materialize; absent in 051 trace, present from iter 54+).
2. **+26 ms** — step D tile-local L1 cull vs iter-55 reader (extra `tile_l1_cull_rd` pass + heavier `tile_mb_mask` vs monolithic `cull_global_mb`).
3. Blend path unchanged in Tracy.

Step C (unified payload reader / mat-before-blend) remains **blocked** — do not bundle with iter 61.

## Iter 61 proposal (one small step)

**Fuse `tile_l1_cull_rd` L1 bulk load into the blend reader** (in-budget `rd_bk_emit` / overflow `rd_l1_bulk`): one PACK2 subchunk DMA + L1 radix per subchunk, then SFPU `tile_mb_mask` on the **same** L1 residency before coeff emit — drop the standalone 3-kernel `tile_l1_cull` program.

- **Why:** `reader_tile_l1_cull.cpp` duplicates the bulk L1 load/sort path the blend reader already runs (`rd_l1_bulk` / bucket path); Tracy shows **+108 ms/view** reader + **+78 ms/view** SFPU vs old single-zone cull, while `rd_l1_bulk` did not shrink.
- **Risk:** low if gated behind existing `MB_TILE_L1_MASKS`-style L1 mask handoff (iter 60 Try2 path); no mat-before-blend / no Step C payload unification.
- **Gate:** 63.84+ dB; ms/view vs iter 60 ≤ 296 (target recover ~20–30 ms of the 55→60 regression).

## Artifacts

- `opt/profiler/ttw-060-tracy/profile_log_device.csv` — 919427 data rows (~30× 1-view)
- `opt/profiler/compare_tracy_zones.py` — mcam-partitioned compare helper
