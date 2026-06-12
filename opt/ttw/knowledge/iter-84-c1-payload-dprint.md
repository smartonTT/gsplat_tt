# Iter 84 — C1 payload DPRINT (yyzo-bh-07)

**Branch:** `smarton/stage2-hostfree-l1` · **Baseline keep:** iter 83 `bin=ff7e2433fe5af4f1`

## DPRINT setup

Env: `MB_C1_PAYLOAD_DEBUG=1`, `TT_METAL_DPRINT_CORES=all` (define wired in `blend_device.cpp`).

Overflow subchunk (`num_subchunks>1`, `L_sub>0`): compare first **16 PACK2 words** (2 splats) — DRAM `sort_subchunk_payload` page `payload_page` vs fresh `pack_blendrec_to_l1` gather for `sort_sorted_ids` k=0,1.

## Findings (representative, all overflow tiles logged)

| Check | Result |
|--------|--------|
| `payload_page == dir_pg` (`pg_eq`) | **1** always — reader page index matches dir word0 |
| `dir_L` vs reader `L_sub` | **Match** (e.g. 8192 for sc0) |
| `dir_fl` vs reader flags | **Match** (sc0: 0; sc≥1: 2\|1) |
| `pay0` vs `gat0` (cov_a) | **Equal** on sampled lines |
| `mism` (first 16 words) | **4** consistently |
| First mismatch | `fk=0 fw=3` → splat word **3** (depth key), also word **7** (cg\|cb pack) per splat |
| `pay7` vs `gat7` | Payload often `0xFFFF…` garbage vs sane gather (e.g. `pay7=4294930527 gat7=1808035935`) |

Example:

```
C1DBG t=663 sc=0 L=8192 pay_pg=1440369 dir_pg=1440369 dir_L=8192 dir_fl=0 pg_eq=1 mism=4 fk=0 fw=3
C1DBG pay0=1074086573 gat0=1074086573 pay7=4294930527 gat7=1808035935
```

**Conclusion:** Not an off-by-one **page** / dir stride bug. Payload **bytes** at the correct page do not match gather PACK2 for the same sorted ids (depth + color words wrong; cov_a matches). `use_payload` DMA would load this DRAM — explains ~18.4 dB when enabled.

Mat is enqueued (`[SUBCHUNK] payload_pages≈1.7M`, piped `materialize_ms` ≈ enqueue only); content still wrong at blend read time.

## Gate (attempt 1, `use_payload=false`)

| Metric | Value |
|--------|------|
| `hero_vs_ref` | **63.63 dB** |
| `ms_view` | **297.0** |
| Decision | **blocked** on payload; **no regression** vs iter 83 |

## Build

Debug run (host + kernel DPRINT hooks, `use_payload` still false): **`bin=497be364272dfde7`** (not stamped in `.ttw/buildid.cpp` — iter 83 stamp still `ff7e2433fe5af4f1`).

## Next (not one-line)

- Trace why `sort_subchunk_payload` PACK2 ≠ gather at words 3/7 (mat write path vs L1_RECORD layout, piped CQ visibility, or buffer init).
- Do **not** enable `use_payload` until payload matches gather on DPRINT (`mism=0`) for overflow sc0/sc≥1.
