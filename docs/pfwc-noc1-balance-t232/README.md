# t232: balance the pfwc writer NoCs, then re-gate P2

Board yyzo-bh-07 (Blackhole p100a), bicycle 30 views 1024x1024, md5 reference
`md5-r82new.txt` (46a725ab).

**Decision: keep.** P2 (`GSPLAT_TT_PFWC_COV2D_SFPU=1`, t228) plus BRISC / NoC0 reading pfwc
input tiles 0-3 on every core (`GSPLAT_TT_PFWC_RD_REST=0x0F`) are both on by default now
(=0 turns each off). Paired gate against the old default: 12.605 -> 12.275 ms/view
(-0.330, gate 0.3), md5-identical on every arm.

## 1. Problem

t228 measured P2 alone at +0.256 ms/view. With the writer split (t221) BRISC / NoC0 writes the
even chunks, NCRISC / NoC1 writes the odd chunks and also reads all 10 input tiles per chunk
(mx my mz c00 c01 c02 c11 c12 c22 opacity). Once P2 cut the TRISC time, the NCRISC on grid
columns x=13-15 became the bound (NCRISC record loop 1.24-1.40 ms).

## 2. Change

- `writer_pfwc_split.cpp` (`PFWC_RD_COLS`): either RISC can read any subset of the 10 input
  tiles. Runtime args 39 (physical NoC0 column mask), 40 (BRISC tile set on those columns),
  41 (BRISC tile set elsewhere); NCRISC reads the complement.
- Knobs (`env_config.h`): `GSPLAT_TT_PFWC_RD_BRISC` (column mask, default 0),
  `GSPLAT_TT_PFWC_RD_SET` (default 0x3FF), `GSPLAT_TT_PFWC_RD_REST` (default now 0x0F).
- Kernel config buffer: with the BRISC reader the program is 96288 B with P2 (98000 B
  without), so the device auto-opens +32 KB instead of +24 (`device_state.cpp`).
- STEPCYC writer record `pfwc_ws` gains `fl` (all time in `flush_rec` / `flush_pg`: write
  issue plus `noc_async_writes_flushed`) and `m` (records written).
- `tests/unit/test_pfwc_wsplit.cpp` models both readers (NCRISC only, BRISC only, both).

## 3. What limits the writers

`fl` showed that the slow cores stall in NoC write issue, not in compute. With BRISC reading
5 tiles on every core, BRISC `fl` was 0.30-0.48 ms on rows y=2-4 vs 0.07-0.14 elsewhere:
input reads on a NoC slow down that NoC's record and page writes. NoC1 reads hurt the
right-hand columns, NoC0 reads hurt rows 2-4. Splitting the reads 4 / 6 balances the two.
About 14.3k records per core per launch (max 17.4k); rec - fl costs 109-125 cycles per record.

## 4. Tracy sweep (STEPCYC=1, KCFG_EXTRA_KB=32, views 0-4)

Mean pfwc span per launch (`span.py`, `out/percol-*.txt`):

| arm | config | mean span ms |
|---|---|---:|
| pb | old default (no P2, NCRISC reads all) | 2.323 |
| p0 | P2 alone | 2.565 |
| pf | P2 + BRISC reads all 10 on x=12-15 (mask F000) | 2.066-2.075 |
| pr1F | P2 + BRISC tiles 0-4 everywhere | 2.04-2.06 |
| pr3F | P2 + BRISC tiles 0-5 everywhere | 2.203 |
| **pr0F** | **P2 + BRISC tiles 0-3 everywhere (new default)** | **1.935** |
| pr07 | P2 + BRISC tiles 0-2 everywhere | 1.985 |
| ph1F_07 | P2 + tiles 0-4 on E000, 0-2 elsewhere | 1.943 |
| ph1F_0F | P2 + tiles 0-4 on E000, 0-3 elsewhere | 1.962 |
| ph3F_07 | P2 + tiles 0-5 on E000, 0-2 elsewhere | 1.923 |

pr0F per column (ms, mean over cores and launches; core wall p50 1.766, p90 1.852, max 1.903):

| x | wall mean | wall max | BRISC rec | BRISC rd | BRISC fl | NCRISC rec | NCRISC rd | NCRISC fl | TRISC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.764 | 1.841 | 0.795 | 0.336 | 0.124 | 0.778 | 0.429 | 0.100 | 1.723 |
| 2 | 1.765 | 1.852 | 0.812 | 0.327 | 0.134 | 0.773 | 0.430 | 0.102 | 1.727 |
| 3 | 1.772 | 1.874 | 0.809 | 0.334 | 0.144 | 0.736 | 0.463 | 0.101 | 1.735 |
| 4 | 1.765 | 1.876 | 0.811 | 0.331 | 0.148 | 0.723 | 0.465 | 0.104 | 1.728 |
| 5 | 1.767 | 1.836 | 0.814 | 0.333 | 0.152 | 0.744 | 0.449 | 0.113 | 1.729 |
| 6 | 1.765 | 1.862 | 0.787 | 0.352 | 0.153 | 0.780 | 0.418 | 0.123 | 1.730 |
| 11 | 1.758 | 1.826 | 0.758 | 0.363 | 0.124 | 0.776 | 0.424 | 0.128 | 1.717 |
| 12 | 1.772 | 1.845 | 0.797 | 0.342 | 0.132 | 0.788 | 0.419 | 0.140 | 1.728 |
| 13 | 1.804 | 1.886 | 0.829 | 0.345 | 0.139 | 0.826 | 0.406 | 0.171 | 1.755 |
| 14 | 1.822 | 1.890 | 0.814 | 0.372 | 0.135 | 0.866 | 0.385 | 0.198 | 1.778 |
| 15 | 1.815 | 1.903 | 0.813 | 0.369 | 0.137 | 0.869 | 0.381 | 0.183 | 1.774 |

For comparison, p0 (P2 alone) at x=14: NCRISC rec 1.402, core wall 2.336. With pr0F the
column means differ by at most 0.07 ms and the right-hand NCRISC record loop is down to
0.83-0.87 ms.

## 5. Paired untraced rounds (c074fae, 30 views, view_total ms/view)

fix = P2 + RD_REST=0x0F; alt = P2 + RD_BRISC=0xE000, RD_SET=0x3F, RD_REST=0x07. Rotated
order per round; `out/summary.txt`.

| round | base | fix | fix delta | alt | alt delta |
|---|---:|---:|---:|---:|---:|
| r1 | 12.607 | 12.261 | -0.346 | 12.353 | -0.254 |
| r2 | 12.609 | 12.257 | -0.352 | 12.367 | -0.242 |
| r3 | 12.626 | 12.281 | -0.345 | 12.283 | -0.343 |
| r4 | 12.579 | 12.301 | -0.278 | 12.342 | -0.237 |
| mean | 12.605 | 12.275 | **-0.330** | 12.336 | -0.269 |

Project stage 3.350 -> 3.034 (fix), 3.046 (alt). md5 46a725ab on all 12 runs; the fix
smoke at the default kernel config open was md5-identical too. The simpler fix wins, so it
is the new default. An earlier mask-only fix (P2 + BRISC reads all 10 on x=12-15) averaged
-0.18 (`out/mask-f000/`).

Not built: (b) a 2:1 BRISC:NCRISC chunk deal and (c) NoC0 for NCRISC record writes. The
read split alone passed the gate and leaves both NoCs with similar write stalls (`fl`).

## 6. Verify on the merged tip (new default vs off)

Pending: `drive.sh 3ec6008 "sync v1 v2 tr"` (rounds v1 / v2: base = new default, off =
`GSPLAT_TT_PFWC_COV2D_SFPU=0,GSPLAT_TT_PFWC_RD_REST=0`; tr = 30-view Tracy of the default).

## 7. Follow-ups

- The remaining writer cost is NoC write-issue stalls (`fl` 0.10-0.20 ms per RISC); a
  non-blocking or ring-buffered flush could hide them once TRISC drops further.
- Per-core load is uneven (records per core p50 ~14.3k, max ~17.4k).
- Choosing the NoC per chunk by DRAM bank column could cut the write stalls further.

## Files

`dev/drive.sh` (Mac driver, one `ttp lock p100` per step), `dev/remote_time.sh`,
`dev/remote_prof.sh`, `dev/remote_tracy.sh`, `dev/percol.py` (per column), `dev/span.py`
(per launch span), `dev/grid.py` / `dev/cores.py` (per core), `dev/summ.py` (paired tables),
`dev/out/` (logs, md5, heroes, Tracy CSVs).
