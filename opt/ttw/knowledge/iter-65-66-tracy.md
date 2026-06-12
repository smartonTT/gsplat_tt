# Iter 65–66 Tracy: compute early-out zone proof

**Device:** `yyzo-bh-07` · **Branch:** `smarton/stage2-hostfree-l1` · **Build:** `cpp#223` `bin=1a2a2641dbf7e117` (iter 65 keep)

**Method:** `compare_tracy_zones.py` — pair ZONE_START/END, partition timeline into 30 views (gap heuristic), per view per zone **max over cores of summed span** (stage makespan), mean across views. Same as iter-60/64 notes.

## Gate (iter 66 verify, no Tracy)

| Metric | Value |
|--------|------:|
| `hero_vs_ref` | **63.63 dB** (≥ 63.6) |
| `ms_view` | **295.43** ms/frame (30-view bench, no profiler) |

## Artifacts

| Role | Path on device |
|------|----------------|
| Iter 64 baseline CSV | `opt/profiler/iter-064-baseline.csv` (copy of pre-iter66 capture @ 2026-06-04 11:24) |
| Iter 66 candidate CSV | `opt/profiler/wrap_out_render-clean/.logs/profile_log_device.csv` |
| Iter 66 archive | `opt/profiler/ttw-066-tracy/{profile_log_device.csv,render.tracy}` |
| Tracy UI | `opt/profiler/render-clean/render.tracy` (~15 MB) |

Tracy wall (30-view, mid-run dump): **363.8 ms/view** — ~68 ms profiler tax vs gate.

## Zone makespan mean ms/view (iter 64 → iter 66)

| Zone | iter 64 | iter 66 | Δ ms | Δ % |
|------|--------:|--------:|-----:|----:|
| `tile_mb_mask` | 324.18 | 324.19 | +0.01 | **0.0%** |
| `tile_l1_cull_rd` | 108.05 | 108.06 | +0.01 | 0.0% |
| `tile_blend_sfpu` | 128.78 | 127.68 | −1.11 | −0.9% |
| **`cp_inb`** | **64.62** | **66.78** | **+2.16** | **+3.3%** |
| `cp_l1_blend` | 25.29 | 24.84 | −0.45 | −1.8% |
| `rd_bk_emit` | 25.44 | 25.93 | +0.50 | +2.0% |
| `rd_l1_bulk` | 24.46 | 24.46 | 0.00 | 0.0% |
| `sort_subchunk_mat` | 27.17 | 27.17 | 0.00 | 0.0% |

**Rolled blend_cp** (`tile_blend_sfpu` + `cp_inb` + `cp_l1_blend` + `cp_ovf`): **218.69 → 219.29** (+0.6 ms/view, **+0.3%**).

**Verdict:** **Flat** — no zone ≥5% drop on `tile_mb_mask` or `cp_inb`; `cp_inb` slightly **up** in makespan. Iter 65 compute early-outs (`mask==0` blend skip, `thr<0` SFPU skip) do **not** show up as measurable stage makespan wins at 30-view Tracy resolution (likely low hit rate and/or overlap hides skipped work; compute still pays mailbox/coeff path for in-budget rows).

**Parser fix:** `compare_tracy_zones.py` `zone_stats` had wrong tuple unpack (`_, z, _, _` → `_, _, z, _`) — fixed in this session; prior empty tables were a tooling bug, not missing zones.

## Iter 65 keep (unchanged)

Quality/perf gate already logged iter 65 (`ms_view` 295.6, `hero_vs_ref` 63.63). **No new iter-66 ledger keep** — Tracy did not justify a measurement-only milestone.

## Iter 67 recommendation (one hypothesis)

**Reader-side row suppress (in-budget `rd_bk_emit` path):** before pushing each gaussian row into `CB_MB_COEFF`, skip rows whose tile microblock mask word is **0** (and/or whose cull `thr` is the `<0` sentinel). Iter 65 only elides `dispatch_blend_guarded` on compute; the reader still does L1 read + mailbox sync per row — Tracy shows **`cp_inb` co-limited (~65 ms/view)** with no drop after compute-only skip.

- **Why now:** compute early-out is banked; Tracy proves the bottleneck remains row traffic into `cp_inb`, not SFPU dispatch count alone.
- **Not Step C:** no unified payload reader / mat-before-blend monolith.
- **Gate:** 63.6+ dB; optional `DPRINT` hit counters first (measure-only lap) if unsure about mask-zero frequency.
- **Stretch (larger):** fuse `tile_l1_cull_rd` into blend reader (~108 ms/view reader bucket, iter-60 proposal) — defer unless row-suppress is also flat.
