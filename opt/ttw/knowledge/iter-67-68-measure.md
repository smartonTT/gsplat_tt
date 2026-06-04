# Iter 67–68 measurement: row-suppress DPRINT + Tracy

**Device:** `yyzo-bh-07` · **Branch:** `smarton/stage2-hostfree-l1`

## Iter 67 (code, kept)

- Reader in-budget row suppress on `rd_bk_emit` (mask==0, `op_f <= contrib_floor`); two-pass count-first emit.
- **Gate:** `hero_vs_ref` **63.63 dB**, `ms_view` **296.30** (flat vs ~295 baseline).
- **Build:** `cpp#226` `bin=aaf8d3ae5b088c68`.

## Iter 68 (measurement)

### RDSUP DPRINT hit rate (3 views: hero, orbit_01, orbit_02)

Env: `MB_RD_ROW_SUPPRESS_DPRINT=1`, `TT_METAL_DPRINT_CORES=all`. Kernel define wired from host via `render/host/blend_device.cpp` when env is set (measurement-only; no effect when unset).

| Metric | Value |
|--------|------:|
| In-budget tiles logged (`RDSUP` lines) | 3,634 (3 views) |
| Candidate rows `L` (sum) | 8,127,107 |
| Emitted rows | 7,695,940 |
| Suppressed (`m0` + `op`) | 431,167 (**5.31%** of candidates) |
| By reason | mask0: 431,167 (5.31%); op_floor: 0 (0.00%) |

**Verdict:** Suppress hit rate **well below 10%** — row suppress cannot move `cp_inb` / blend makespan materially at 30-view scale. **Do not invest further in row-suppress tuning this sprint.**

### Tracy (30-view, `capture_tracy_clean.sh`, no DPRINT)

**Baseline:** `opt/profiler/ttw-066-tracy/profile_log_device.csv` (iter 66, compute early-outs).  
**Candidate:** `opt/profiler/ttw-068-tracy/profile_log_device.csv` (post–iter-67 reader suppress; host `.so` includes DPRINT hook only — `bin=d3754d74d3a307b3` after hook; kernels unchanged when DPRINT unset).

Tracy wall: **363.9 ms/view** (~68 ms profiler tax vs gate **296.6 ms/view**).

#### Zone makespan mean ms/view (iter 66 → iter 68)

| Zone | iter 66 | iter 68 | Δ ms | Δ % |
|------|--------:|--------:|-----:|----:|
| `tile_mb_mask` | 324.19 | 324.22 | +0.03 | 0.0% |
| `tile_l1_cull_rd` | 108.06 | 108.07 | +0.01 | 0.0% |
| `tile_blend_sfpu` | 127.68 | 129.35 | +1.67 | +1.3% |
| **`cp_inb`** | **66.78** | **66.65** | **−0.13** | **−0.2%** |
| `rd_bk_emit` | 25.93 | 25.13 | −0.81 | −3.1% |
| `cp_l1_blend` | 24.84 | 24.84 | 0.00 | 0.0% |
| `rd_l1_bulk` | 24.46 | 23.28 | −1.17 | −4.8% |

**Rolled `blend_cp`** (`tile_blend_sfpu` + `cp_inb` + `cp_l1_blend` + `cp_ovf`): **219.29 → 220.84** (+1.55 ms/view, **+0.7%**). **Flat** — no zone ≥5% win on `cp_inb` or `tile_mb_mask`; `rd_bk_emit` slightly down, SFPU makespan slightly up (noise / overlap).

### Gate (30-view, no DPRINT)

| Metric | Value |
|--------|------:|
| `hero_vs_ref` | **63.63 dB** (≥ 63.6) |
| `ms_view` | **296.62** |

## Recommendations

| Track | Decision |
|-------|----------|
| **Row suppress (iter 67) as perf lever** | **Reject** for further perf work — 5.3% suppress rate + flat Tracy; keep iter 67 for correctness / architectural alignment only. |
| **Fuse `tile_l1_cull_rd` (iter 61 hang)** | **Defer** this iter (per &lt;10% suppress rule). |
| **Iter 69 hypothesis (pick one)** | (1) **Fuse `tile_l1_cull_rd` into blend reader** (~108 ms/view bucket, iter-60 proposal) — only after separate hang bisect; **or** (2) **Reduce `tile_mb_mask` traffic** (~324 ms/view dominant bucket). |

**NO Step C monolith.**
