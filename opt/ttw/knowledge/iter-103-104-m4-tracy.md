# M4 — dead-code strip + ALU-bound diagnostic (iter 103 keep)

**Device:** `yyzo-bh-07` · **Branch:** `smarton/stage2-hostfree-l1` · **Keep:** iter 103 `cpp#59` `bin=81e2917b5cb8f378` (M4a cleanup, bit-identical to anchor).

## Gate (iter 103, no profiler)

| Metric | Value |
|--------|------:|
| `hero_vs_ref` | **63.85 dB** (= anchor; ≥ 63.6 gate) |
| `avg_frame_ms` | **98.4 ms/view** (30-view, no Tracy) — **below** the ~104 ms pre-sprint anchor region (net ms ≤ pre-sprint ✓) |

Pure dead-code removal ⇒ byte-identical render ⇒ exact anchor 63.85 dB, build-delta moved (`df835eea… → 81e2917b…`).

## Part A — what was deleted (no ABI/runtime-arg change)

All removals are localized (no host↔kernel runtime-arg renumbering, no CT-arg-chain edits), so each is bisectable and the slot-aligned bulk-CB invariant (M1b lesson) is untouched.

- **Blend reader** (`reader_alpha_blend_mb_devcull.cpp`): removed uncalled helpers `read_soa_u32`, `issue_chunk_reads_aos` (+`GATHER_FIELDS/GATHER_SLOT_BYTES`), `load_ids_chunk`, `load_mask_page`, `pack_fp32_unorm16`, `pack_blendrec_to_l1`, `l1_splat_words`, `f_to_bits`, `rd_row_suppress`; the dead per-tile `bucket_meta` read block (computed `rec_start`/`Lb`, never used — a per-tile NoC read + barrier now gone); dead CB consts `CB_MB_COEFF`/`CB_BUCKET`/`CB_BSORT`/`CB_BMASK`; the `dprint.h` debug include guard.
- **Cull reader** (`reader_tile_l1_cull.cpp`): dead `CB_BSORT` const.
- **Blend compute** (`alpha_blend_compute_mb.cpp`): dead `CB_MB_COEFF`/`CB_BUCKET`/`CB_BMASK` consts.
- **Host** (`blend_device.cpp`): dead `cb_cfg` for `CB_MB_COEFF` (coeff stream) and `CB_BSORT` (radix scratch) in BOTH the blend and cull programs (frees L1); reader defines `MB_CULL_SPIN`, `MB_C1_PAYLOAD_DEBUG`, `MB_RD_ROW_SUPPRESS_DPRINT` (debug/measurement scaffolding).

### Left in place (deliberate, with reason)
- **`CB_BUCKET`** — NOT dead: M2 reuses it as the cull reader's slab bulk-load scratch (reserved-but-never-pushed). Kept in cull reader + cull-program `cb_cfg`.
- **`proj_m_blendrec` DRAM buffer + `sort_tile_recs`/`sort_bucket_meta`** — still LIVE upstream: `sort_subchunk_materialize`/`sort_bin` consume `proj_m_blendrec` to build the depth-sorted PACK2 slab; sort host still publishes `bucket_meta`. Only their now-dead *consumer wiring* in the cull/blend readers is void-cast. Removing the buffers/args would require renumbering the host↔kernel runtime-arg + TensorAccessor CT-arg contract (mid-list args 0–6/17–20 in the blend reader, 0/1/3/4 in the cull reader) — high-risk ABI churn for ~0 runtime cost (the void-cast accessors do no DMA). Deferred per the bounded-effort + "leave if unsure" mandate; logged as a follow-up.
- `CB_SCR_ATTR`/`CB_SCR_MASK` blend-program `cb_cfg` (≈18 KB dead L1) — not on the explicit target list; safe follow-up.

## Part B — 30-view Tracy DEVICE-zone diagnostic (the M4 evidence)

Capture: reconstructed `opt/profiler/capture_tracy.sh <iter-dir>` (+ `_capture_inner.sh`) — the proven `python -m tracy --dump-device-data-mid-run` wrapper (only path that streams DEVICE zones since gsplat never closes the device), full 30-view `render/run.py --no-ref`, same flags as `[run] verify_cmd`. Output: `opt/profiler/ttw-103/render.tracy` (11.5 MB) + device CSV **699,303 rows (~30×1-view ⇒ full 30 views)**. Analyzed with `opt/profiler/analyze_zones.py` (pairs ZONE_START/END per core+RISC; **per-view makespan = max-over-cores of summed span**, the iter-60/65 metric).

NOTE: fresh failover venv on `yyzo-bh-07` was missing tracy deps — installed `loguru pyyaml click tabulate pandas seaborn` into `/localdev/smarton/gstt2/.venv` (one-time; matplotlib/numpy already present).

### Per-stage DMA (dataflow RISC) vs SFPU (compute), per-view makespan (ms)

| Stage | DMA zone (RISC) | DMA ms | SFPU zone (RISC) | SFPU ms | DMA/SFPU | ALU-bound? |
|-------|------------------|-------:|-------------------|--------:|---------:|------------|
| **Cull** | `tile_l1_cull_rd` (NCRISC) | **26.71** | `tile_mb_mask` (TRISC) | **26.72** | **1.00** | **No — co-limited** |
| **Blend** | `tile_blend_load` (NCRISC), of which `rd_l1_bulk`=12.33 | **12.35** | `tile_blend_sfpu` (TRISC) | **13.08** | **0.94** | Marginal (≈ balanced, SFPU slightly leads) |

Tracy-run `avg_frame_ms` = 117.7 (≈ +19 ms profiler/mid-run-dump tax vs the 98.4 gate run).

### Verdict
**NOT yet strongly ALU-bound.** Blend is essentially balanced / marginally SFPU-bound. **Cull is exactly co-limited:** its slab bulk-load reader (`tile_l1_cull_rd`, 26.7 ms/view) equals the cull SFPU (`tile_mb_mask`, 26.7) and is the single **largest dataflow zone in the whole pipeline**.

### Dominant remaining DMA lever → M5
`tile_l1_cull_rd` (cull reader, 26.7 ms/view) — a full depth-sorted PACK2 slab bulk-DMA per subchunk that is NOT overlapped with the cull SFPU. **M5 double-buffer prefetch (subchunk N+1 while SFPU processes N)** is the lever to hide it behind `tile_mb_mask` and push cull into ALU-bound. The same pattern (smaller) sits in the blend reader `rd_l1_bulk` (12.3 ms/view). Secondary: `tile_mb_mask` SFPU itself (26.7 ms/view, the biggest compute zone) is the post-prefetch cull ceiling → microblock-saturation early-out / `kBucketFit` sweep (also M5).
