# Task #83 — the "blend reader/handshake floor" at GSPLAT_TT_BLEND_ABL=4: attribution

**Board: yyzo-bh-07 (Blackhole p100a), not a p150.** Bicycle, 1024x1024. Code: d8afd09
(= 8fc8c31 kernels) plus default-off profiling zones (3808ef2). Result: **no fix; the
floor is not the blend reader.** Nothing to gain above the 3 ms gate in the blend program's
reader/handshake/emit path.

## What the 5.6 ms really is

The host `blend` stage timer is the one `Finish` that drains the command queue after the
blend enqueue. Three programs are still in flight at that point, back to back with no idle
between them: `sort_subchunk_mat`, the SFPU cull, and the blend itself. With no per-record
TRISC work (ABL=4) the blend program is only 0.76 ms of the 5.6 ms.

Per-program windows (new `opt/profiler/zone_windows.py`, first start to last end over all
cores, mean of 10 traced views, Tracy chunk 0:10):

| program | ABL=4 ms | ABL=0 ms |
|---|---:|---:|
| sort_subchunk_mat | 3.53 | 3.52 |
| SFPU cull (tile_mb_mask + tile_l1_cull_rd) | 1.32 | 1.32 |
| blend (reader + compute + writer) | **0.76** | 7.69 |
| sum (traced) | 5.61 | 12.53 |
| host `blend` stage, untraced (1 round) | 5.58 | 12.39 |

## Inside the blend program (mean busy per core, ms/view, traced)

New zones behind `GSPLAT_TT_BLEND_PROF=1` (compiled out by default).

| RISC | zone | what | ABL=4 | ABL=0 |
|---|---|---|---:|---:|
| NCRISC | tile_blend_load | whole reader kernel | 0.49 | 6.67 |
| NCRISC | rd_claim | NoC atomic tile claim (spin on return) | 0.00 | 0.00 |
| NCRISC | rd_tid | tile id DRAM read | 0.04 | 0.02 |
| NCRISC | rd_meta | tile range + subchunk meta reads | 0.07 | 0.03 |
| NCRISC | rd_dir | subchunk dir read | 0.04 | 0.02 |
| NCRISC | rd_l1_bulk | slab bulk NoC read incl. slot wait | 0.28 | 6.56 |
| NCRISC | rd_bulk_wait | of which: waiting for compute to free the slot | 0.04 | **6.48** |
| TRISC | tile_blend_sfpu | whole compute kernel | 0.59 | **7.00** |
| TRISC | cmp_wait_cnt | waiting for the reader's counts page | 0.01 | 0.00 |
| TRISC | cmp_bulk_wait | waiting for the slab | 0.00 | 0.00 |
| TRISC | cmp_init | DEST acquire, fill R/G/B/T, copy ramps | 0.10 | 0.00 |
| TRISC | cmp_emit | commit + pack R/G/B (PACK waits for MATH here) | 0.07 | 0.32 |
| BRISC | wr_wait_out | waiting for the R/G/B tiles | 0.22 | 6.42 |
| BRISC | wr_pack | u8 pack + row writes | 0.44 | 0.61 |

T readbacks run inside the per-record loop, so ABL=4 skips them. The t80 ablation bounds
them: a3 - a4 = 1.0 ms for the whole loop (mask read, loop, T readbacks).

## Reading

- At ABL=0 the blend is compute-bound. Compute never waits for data (`cmp_bulk_wait` and
  `cmp_wait_cnt` are 0.00). The reader spends 6.48 of its 6.67 ms waiting for compute to
  free a slab slot, and the writer spends 6.42 of its time waiting for tiles. The reader,
  the claim, the metadata reads and the writer are all hidden behind the SFPU.
- The most any reader/handshake/emit change could save is the non-compute part of the
  blend window: 7.69 - 7.00 = 0.69 ms/view (launch ramp and tail imbalance, not the
  reader). That is under the 3 ms gate, so no fix was implemented.
- The real 4.85 ms under the ABL=4 number is `sort_subchunk_mat` (3.52 ms window, 2.72 ms
  mean per core, so ~0.8 ms is tail imbalance) and the SFPU cull (1.32 ms). Both are booked
  under "blend" by the stage timer. They are the next levers in this window.
- Traced runs also show 5.3-5.6 ms/view of all-core idle between programs, with the
  ta_bucket_scatter -> sort_bin_hist gap at 2.5 ms (it was 1.29 ms in the task #31 doc).
  This should be re-checked untraced.

## Measured (untraced, 30 views)

| run | avg ms/view | blend stage ms | output |
|---|---:|---:|---|
| tip, default build (verify) | 44.6 | 12.30 | 30/30 md5-identical to 8fc8c31 (t82 run), hero_vs_ref 100 dB |
| round 1, ABL=0 | 44.5 | 12.39 | |
| round 1, ABL=4 | 38.1 | 5.58 | wrong by design (timing only) |

No ms/view change: no fix landed, so no ledger iter and no report regeneration.

## Files

`remote_*.sh` (copied to `/localdev/smarton/t83_scripts`, tree `/localdev/smarton/gstt2-t83`),
`tracy-a0.txt` / `tracy-a4.txt` (zone table, program gaps, zone windows), `ab.log`.
Remote captures: `/localdev/smarton/gstt2-t83/opt/profiler/t83-a{0,4}/chunks/0-10/`.
