# Hardware ceilings — RISC / NoC microbenchmarks (T-A)

Measured 2026-09-30 on **`yyzo-bh-07`, Blackhole p100a** (board_id `000004323191b005`),
tt-metal `e77780fe`. Not a p150: no p150 was free (see project decision log); the
p150 re-run is the same one command (below). Every number here is `[M]` on p100a
unless marked `[P]` (published) or `[D]` (derived).

Tool: `render/bench/` — host driver `risc_microbench.cpp` (own CMake project, does not
build or link `render_clean`) + one dataflow kernel per probe in `render/bench/kernels/`.
Each probe runs N = 2²⁰ ops per RISC (fewer where noted) inside one
`DeviceZoneScopedN`, timed with the RISC-V wall clock, once on core (0,0) and once
on all 110 cores. JIT warm-up run discarded; the timed run is the second. Whole suite
≈ 40 s of device time. Four full runs: every probe repeated to 3–4 significant digits
(the scatter probes p4–p6 were re-run after fixing a page stride that pinned each core to
one DRAM bank; the tables use the fixed run).

```bash
export TT_METAL_HOME=/localdev/smarton/tt-metal
cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release
cmake --build render/bench/build
TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME render/bench/build/risc_microbench | grep '^\[MB\]'
```

## Clock and grid

| item | value | how |
|---|---|---|
| tick rate (RISC wall clock) | **1350.0 MHz** | `[M]` 2³⁰ − 2²⁸ tick spin vs host `steady_clock` (0.5966 s / 805 M ticks) |
| RISC core clock during run | 1.35 GHz | `[M]` p1 loop is 20 instructions (disassembled) and takes 20.0 ticks; a single-issue core cannot beat 1 IPC ⇒ core clock = tick rate |
| AICLK reported by `tt-smi -s` | 800 MHz | `[M]` idle clock, read before device open / after close; tt-metal raises it to 1.35 GHz while open |
| `compute_with_storage_grid_size()` | **11 × 10 = 110** | `[M]` |
| DRAM channels | 7 | `[M]` `num_dram_channels()` |

Below, **1 tick = 1 cycle = 0.741 ns**.

## Results (ticks per op, median over cores; NCRISC unless noted)

| probe | what one op is | 1 core | 110 cores | notes |
|---|---|---|---|---|
| p0 loop overhead | loop iter / 8 (index + address math only) | 0.625 | 0.625 | 5 cycles per 8-op iteration; included in p1/p2 |
| **p1 L1 store, volatile u32** | one `volatile uint32_t` store | **2.50** | 2.50 | BRISC identical. Loop = 8 `sw` + 7 `xori` + 5 addr/loop = 20 instr in 20 cycles: **the store itself issues in 1 cycle, no stall** |
| p2 32 B block, `__builtin_memcpy(void*,…)` | per 4 B word | **12.38** | 12.38 | GCC emits `jal memcpy` (checked in ELF): **99 cycles/record, 5× slower than p1** |
| p2 32 B block, typed non-volatile `u32*` | per 4 B word | 9.00 | 9.00 | record spilled to stack then copied |
| p2 32 B block, memcpy + `__builtin_assume_aligned(dst,16)` | per 4 B word | **2.75** | 2.75 | inlined 8×`sw`; still ≥ p1 |
| p3 L1 load, independent | per load (8 per iter, summed) | 4.07 | 4.07 | |
| **p3 L1 load, dependent chain** | load-to-use latency | **8.26** | 8.26 | every load whose result is used next costs ~8 cycles |
| p4 `noc_async_write` 32 B → DRAM, precomputed addr | issue only | 40.6 | 40.6 | 16 per batch, barrier excluded; pages stepped by 5 so every core touches every bank |
| p4 same, barrier share | barrier ÷ 16 | 19.6 | 20.8 | |
| **p4 `noc_async_write` + `get_noc_addr(page, TensorAccessor)`** | issue only | **55.6** | 55.6 | accessor address math = **15 cycles** |
| p4 same, barrier share | barrier ÷ 16 | 19.6 | 19.7 | ≈ 313 cycles per batch-of-16 barrier |
| p4 total per write (production shape) | issue + barrier/16 | **77.7** | 77.8 | plan §2.2 assumed 50 |
| **p5 `noc_async_read` 64 B DRAM, 1 outstanding + barrier** | exposed round trip | **476** | 478 | = 353 ns, incl. 15-cycle `get_noc_addr`; sort_bin's blendrec read today |
| p5 same, 8 outstanding + 1 barrier | per read | **105** | 274 | 4.5× better on 1 core; at 110 cores the chip caps at ~0.54 G 64 B reads/s (35 GB/s) |
| p8 `noc_async_write_barrier()`, nothing outstanding | per call | 16 | 16 | |
| p8 `noc_async_read_barrier()`, nothing outstanding | per call | 16 | 16 | |
| p9 fp32 `a − (float)u32` | per op | **84** | 88 | **soft-float** (`__floatunsisf` + `__subsf3`): data movers have no FPU |
| p9 fp32 `a × b` | per op | **90** | 90 | soft-float `__mulsf3` |
| p9 u32 ÷ runtime divisor | per op | 20.4 | 20.4 | hardware `divu` |

### p6 — dual data mover (R2's scaling factor, measured)

Loop = the `sort_bucket_emit` shape: pack a 32 B record with 8 volatile stores, scatter
16 records as 32 B `noc_async_write`s with `get_noc_addr` per write, one barrier per 16.
`_rd` adds one 64 B DRAM read + immediate barrier per record (the blendrec 1-deep read),
so the loop is latency-bound instead of NoC-write-bound. Makespan = slowest RISC.

| loop | movers | 1 core makespan | scaling | 110 cores makespan | scaling |
|---|---|---|---|---|---|
| store+scatter (2²⁰ rec) | NCRISC only | 70.53 ms (90.8 cyc/rec) | 1.00 | 70.75 ms | 1.00 |
| store+scatter | BRISC only | 70.20 ms (90.4 cyc/rec) | — | 98.99 ms (slow tail, max 127 cyc/rec) | — |
| store+scatter | BRISC + NCRISC, N/2 each | 35.36 ms | **1.99×** | 64.93 ms | **1.09×** |
| store+read+scatter (2¹⁸ rec) | NCRISC only | 109.14 ms (562 cyc/rec) | 1.00 | 113.03 ms | 1.00 |
| store+read+scatter | BRISC + NCRISC, N/2 each | 54.56 ms | **2.00×** | 61.65 ms | **1.83×** |

- No measurable L1-port contention between the two movers (1-core split is exactly 2.00×;
  p1/p2 are identical on BRISC and NCRISC).
- At 110 cores a loop that does nothing but 32 B DRAM scatter saturates the chip at
  **~1.8–2.3 G scattered 32 B writes/s** (p4_pre on NOC1 alone 2.27 G/s, p4_acc 1.91 G/s;
  p6 split over NOC0+NOC1 1.78 G/s). **BRISC's NOC0 path to DRAM is ~40 % slower at that
  load** (1.16 G/s alone, slow tail to 127 cyc/rec) and limits the split.
- A latency-bound loop scales 1.83× at 110 cores.

### p7 — DRAM bandwidth (8 KiB pages, 8 outstanding)

| direction | movers | 1 core | 110 cores aggregate |
|---|---|---|---|
| read | NCRISC | 56.7 GB/s | **214.9 GB/s** |
| write | NCRISC | 61.0 GB/s | 259.7 GB/s |
| read | BRISC + NCRISC | 108.5 GB/s | **421.0 GB/s** |

Published p100a peak is 448 GB/s (7 × 64 GB/s GDDR6) `[P]`; p150 is 512 GB/s `[P]`.
Both NoCs together reach **94 %** of peak; **one NoC tops out at ~48 %** — a stage that
streams DRAM on a single mover can never exceed ~215 GB/s on this board.

## What this means for `sort_bucket_emit` (1 904 cycles/pair `[D]`, plan §2.3)

Measured cost of the three stall suspects named in the plan, per pair (K = 1.71 pairs per
gaussian):

| suspect | measured cost / pair | share of 1 904 |
|---|---|---|
| 8 volatile `pack_rec` stores | 8 × 2.5 = **20** | 1.1 % |
| 1-deep `blendrec` read + barrier | 476–478 ÷ 1.71 = **~279** | 15 % |
| 32 B scatter write + barrier share | **~78** | 4 % |
| `ksp`/`isp` (+`curp`) volatile stores | 3 × 2.5 = 7.5 | 0.4 % |
| **sum of the three named suspects** | **~385** | **~20 %** |

Not named in the plan but measured here: **soft-float**. The data movers have no FPU;
`pack_rec` does 2 × (u32→float + fsub) per pair = 2 × 84 = **168 cycles (9 %)**, and the
once-per-gaussian `pack_invariants` does 4 × `to_unorm` (2 compares, fmul, fadd, f→u each;
at ~85–90 cycles per soft-float op that is on the order of 1 000 cycles/gaussian ≈
**~550 cycles/pair `[D]`, ~30 %**). That is the most likely owner of the unexplained
~60–80 %, and it matches the iter-129 ablation: nulling the 8 stores made `pack_rec`
and its float inputs dead code (−57 %) — the stores themselves are 1 %.

**Is `sort_bucket_emit` within 2× of any measured hardware ceiling? No.**
- Instruction/latency ceiling for its exact dataflow (p6_rd with K = 1.71 records per
  read): (476 + 1.71 × 91) ÷ 1.71 ≈ **370 cycles/pair** — production is **5.1×** above it.
- Pure store+scatter ceiling: 91 cycles/record — production is 21× above.
- NoC write rate: production ≈ 2.4 M pairs ÷ 30.6 ms = 78 M writes/s vs a
  1.8–2.3 G/s chip cap — **>20× headroom**; blendrec reads ≈ 1.4 M ÷ 30.6 ms = 46 M/s
  vs a ~0.54 G/s cap — 12× headroom. DRAM bandwidth: ~1 % used.

## Implications for R3 (de-stall the emit loop)

1. **Change 1 (register-built record + one `__builtin_memcpy`) — expected gain ≈ 0, and
   the form written in the plan is a regression.** Volatile stores cost 2.5 cycles; an
   aligned inlined block costs 2.75. A plain `__builtin_memcpy((void*)…, rec, 32)` compiles
   to a `memcpy` call: +79 cycles/pair ≈ **+4 % (~+1.3 ms)**. Skip it, or if kept for
   readability use `__builtin_assume_aligned(dst, 16)` and expect neutral.
2. **Change 2 (8-deep `blendrec` prefetch) — the real stall lever of the three.** 476 → 105
   cycles per gaussian ⇒ −217 cycles/pair ≈ **−11 % of emit ≈ −3.5 ms** `[D]`. The 110-core
   p5_d8 figure (274) is the chip's ~0.54 G reads/s cap; production issues 12× fewer, so
   the 1-core figure applies.
3. **Change 3 (batch `ksp`/`isp`) — ≤ 5 cycles/pair (0.3 %). Skip.**
4. **New, larger lever: remove soft-float from the per-pair and per-gaussian path**
   (~40 % of emit `[D]`). Options, bit-identical ones first: integer-only `to_unorm`
   (exhaustively checkable on host over all fp32 in (0,1)); precompute per gaussian
   what can be, compute `(float)(txi*32)` via a 32-entry table (8 cycles) instead of
   `__floatunsisf`; or move the tile-origin subtraction to the blend side (format change).
   Confirm with one nested zone around `pack_invariants` before committing (T-B/E3).
   The same cost applies to **every NCRISC/BRISC stage that does float math per element**
   (projection, cull, gather): ~85–90 cycles per float op vs 1 cycle per integer op. This
   strongly supports R9 (move scalar float work to the SFPU) and an audit of float ops in
   all data-mover kernels.

## Implications for R2 (split single-mover stages over BRISC + NCRISC)

- Expected scaling for `sort_bucket_emit`: **1.83–2.0×** of the zone `[M]` (its NoC write
  and read rates are 12–20× below the caps where scaling stops) ⇒ 30.6 → ~16–17 ms,
  **−13 to −15 ms/view** `[D]`, before R3. With R3 change 2 + soft-float removal landed
  first the absolute gain shrinks proportionally, but the factor holds.
- A 50/50 split is right for latency/compute-bound stages. For a stage that is itself
  NoC-write-saturating at 110 cores, dual-mover gives almost nothing (1.09×) — check a
  stage's aggregate small-write rate against ~1.8 G/s before splitting it.
- DRAM-streaming stages on one mover are capped at ~215 GB/s; splitting them across both
  NoCs doubles that (420 GB/s). Irrelevant to the current ~1 %-of-DRAM workload, but it
  matters for R7 if it moves bulk data.
- BRISC (NOC0) is ~1.4× slower than NCRISC (NOC1) for saturating 110-core DRAM scatter
  (and 1.83× rather than 2× scaling for the latency-bound loop); give BRISC ≤ 50 % of a
  write-heavy stage, or have both movers issue on NOC1.

## Plan corrections

| plan assumption | measured |
|---|---|
| §2.2 NoC transaction issue ≈ 50 cycles | 41 issue + 15 accessor + ~20 barrier share = **~78** (≈ 5.2 ms/core at 91 k transactions — 3 % of the frame, not 2 %) |
| §2.3 "volatile stores are a stall source" | **refuted**: a volatile L1 store issues in 1 cycle (IPC = 1 in p1); 8 of them ≈ 1 % of a pair |
| T-B change 1 expected gain = 8·p1 − p2 | 8 × 2.5 − 22 ≈ **−2 cycles** (neutral); plain memcpy +79 (regression) |
| AICLK 1.35 GHz `[P]` | 1350.0 MHz `[M]` (tick rate vs host clock) |
| 512 GB/s DRAM `[P]` | p100a: 448 `[P]`, **420 achieved** with both NoCs, 214 with one |

## Raw `[MB]` log (final run, `yyzo-bh-07`, p100a)

```
[MB] grid=11x10 cores=110 dram_channels=7
[MB] clock tick_mhz=1350.0 (ticks 1073741830 vs 268435456, host 0.7955 s vs 0.1990 s)
[MB] probe=p0_loop_overhead scope=1core cores=1 risc=n n=1048576 tpo_med=0.625 tpo_min=0.625 tpo_max=0.625 a_med=0.000 b_med=0.000 makespan_ticks=655393 makespan_ms=0.485 host_ms=0.576
[MB] probe=p0_loop_overhead scope=all cores=110 risc=n n=1048576 tpo_med=0.625 tpo_min=0.625 tpo_max=0.625 a_med=0.000 b_med=0.000 makespan_ticks=655397 makespan_ms=0.485 host_ms=0.657
[MB] probe=p1_l1_store_volatile scope=1core cores=1 risc=n n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621482 makespan_ms=1.942 host_ms=2.054
[MB] probe=p1_l1_store_volatile scope=all cores=110 risc=n n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621486 makespan_ms=1.942 host_ms=2.122
[MB] probe=p1_l1_store_volatile scope=1core cores=1 risc=b n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621482 makespan_ms=1.942 host_ms=2.036
[MB] probe=p1_l1_store_volatile scope=all cores=110 risc=b n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621486 makespan_ms=1.942 host_ms=2.131
[MB] probe=p2_l1_store_block_memcpy scope=1core cores=1 risc=n n=1048576 tpo_med=12.375 tpo_min=12.375 tpo_max=12.375 a_med=0.000 b_med=0.000 makespan_ticks=12976178 makespan_ms=9.612 host_ms=9.708
[MB] probe=p2_l1_store_block_memcpy scope=all cores=110 risc=n n=1048576 tpo_med=12.375 tpo_min=12.375 tpo_max=12.375 a_med=0.000 b_med=0.000 makespan_ticks=12976178 makespan_ms=9.612 host_ms=9.777
[MB] probe=p2_l1_store_block_typed scope=1core cores=1 risc=n n=1048576 tpo_med=9.000 tpo_min=9.000 tpo_max=9.000 a_med=0.000 b_med=0.000 makespan_ticks=9437218 makespan_ms=6.990 host_ms=7.087
[MB] probe=p2_l1_store_block_typed scope=all cores=110 risc=n n=1048576 tpo_med=9.000 tpo_min=9.000 tpo_max=9.000 a_med=0.000 b_med=0.000 makespan_ticks=9437218 makespan_ms=6.990 host_ms=7.161
[MB] probe=p2_l1_store_block_aligned scope=1core cores=1 risc=n n=1048576 tpo_med=2.750 tpo_min=2.750 tpo_max=2.750 a_med=0.000 b_med=0.000 makespan_ticks=2883618 makespan_ms=2.136 host_ms=2.234
[MB] probe=p2_l1_store_block_aligned scope=all cores=110 risc=n n=1048576 tpo_med=2.750 tpo_min=2.750 tpo_max=2.750 a_med=0.000 b_med=0.000 makespan_ticks=2883618 makespan_ms=2.136 host_ms=2.297
[MB] probe=p3_l1_load scope=1core cores=1 risc=n n=1048576 tpo_med=12.334 tpo_min=12.334 tpo_max=12.334 a_med=4.071 b_med=8.263 makespan_ticks=12933656 makespan_ms=9.580 host_ms=9.677
[MB] probe=p3_l1_load scope=all cores=110 risc=n n=1048576 tpo_med=12.334 tpo_min=12.334 tpo_max=12.335 a_med=4.071 b_med=8.263 makespan_ticks=12933666 makespan_ms=9.580 host_ms=9.756
[MB] probe=p4_noc_write_issue_pre scope=1core cores=1 risc=n n=1048576 tpo_med=62.563 tpo_min=62.563 tpo_max=62.563 a_med=40.625 b_med=19.562 makespan_ticks=65601586 makespan_ms=48.592 host_ms=48.708
[MB] probe=p4_noc_write_issue_pre scope=all cores=110 risc=n n=1048576 tpo_med=63.777 tpo_min=60.761 tpo_max=65.317 a_med=40.632 b_med=20.772 makespan_ticks=68489477 makespan_ms=50.731 host_ms=51.015
[MB] probe=p4_noc_write_issue_acc scope=1core cores=1 risc=n n=1048576 tpo_med=77.688 tpo_min=77.688 tpo_max=77.688 a_med=55.562 b_med=19.562 makespan_ticks=81461322 makespan_ms=60.340 host_ms=60.473
[MB] probe=p4_noc_write_issue_acc scope=all cores=110 risc=n n=1048576 tpo_med=77.782 tpo_min=76.808 tpo_max=77.794 a_med=55.563 b_med=19.656 makespan_ticks=81572695 makespan_ms=60.422 host_ms=60.623
[MB] probe=p5_noc_read_rt_d1 scope=1core cores=1 risc=n n=1048576 tpo_med=475.836 tpo_min=475.836 tpo_max=475.836 a_med=0.000 b_med=0.000 makespan_ticks=498950005 makespan_ms=369.581 host_ms=369.736
[MB] probe=p5_noc_read_rt_d1 scope=all cores=110 risc=n n=1048576 tpo_med=477.670 tpo_min=447.666 tpo_max=478.071 a_med=0.000 b_med=0.000 makespan_ticks=501293957 makespan_ms=371.317 host_ms=371.586
[MB] probe=p5_noc_read_rt_d8 scope=1core cores=1 risc=n n=1048576 tpo_med=105.335 tpo_min=105.335 tpo_max=105.335 a_med=0.000 b_med=0.000 makespan_ticks=110452051 makespan_ms=81.814 host_ms=81.929
[MB] probe=p5_noc_read_rt_d8 scope=all cores=110 risc=n n=1048576 tpo_med=273.610 tpo_min=254.217 tpo_max=290.637 a_med=0.000 b_med=0.000 makespan_ticks=304754563 makespan_ms=225.737 host_ms=225.975
[MB] probe=p6_dual_mover scope=1core cores=1 risc=n n=1048576 tpo_med=90.813 tpo_min=90.813 tpo_max=90.813 a_med=0.000 b_med=0.000 makespan_ticks=95223880 makespan_ms=70.534 host_ms=70.652
[MB] probe=p6_dual_mover scope=all cores=110 risc=n n=1048576 tpo_med=91.085 tpo_min=90.185 tpo_max=91.086 a_med=0.000 b_med=0.000 makespan_ticks=95510760 makespan_ms=70.746 host_ms=70.959
[MB] probe=p6_dual_mover scope=1core cores=1 risc=b n=1048576 tpo_med=90.379 tpo_min=90.379 tpo_max=90.379 a_med=0.000 b_med=0.000 makespan_ticks=94769232 makespan_ms=70.197 host_ms=70.336
[MB] probe=p6_dual_mover scope=all cores=110 risc=b n=1048576 tpo_med=91.389 tpo_min=90.920 tpo_max=127.454 a_med=0.000 b_med=0.000 makespan_ticks=133645149 makespan_ms=98.993 host_ms=99.179
[MB] probe=p6_dual_mover scope=1core cores=1 risc=b+n n=524288 tpo_med=91.063 tpo_min=90.611 tpo_max=91.063 a_med=0.000 b_med=0.000 makespan_ticks=47743354 makespan_ms=35.364 host_ms=35.466
[MB] probe=p6_dual_mover scope=all cores=110 risc=b+n n=524288 tpo_med=117.398 tpo_min=98.163 tpo_max=167.183 a_med=0.000 b_med=0.000 makespan_ticks=87651822 makespan_ms=64.925 host_ms=65.125
[MB] probe=p6_dual_mover_rd scope=1core cores=1 risc=n n=262144 tpo_med=562.087 tpo_min=562.087 tpo_max=562.087 a_med=0.000 b_med=0.000 makespan_ticks=147347624 makespan_ms=109.143 host_ms=109.265
[MB] probe=p6_dual_mover_rd scope=all cores=110 risc=n n=262144 tpo_med=580.998 tpo_min=550.634 tpo_max=582.107 a_med=0.000 b_med=0.000 makespan_ticks=152595804 makespan_ms=113.030 host_ms=113.255
[MB] probe=p6_dual_mover_rd scope=1core cores=1 risc=b+n n=131072 tpo_med=561.970 tpo_min=548.564 tpo_max=561.970 a_med=0.000 b_med=0.000 makespan_ticks=73658512 makespan_ms=54.560 host_ms=54.697
[MB] probe=p6_dual_mover_rd scope=all cores=110 risc=b+n n=131072 tpo_med=630.654 tpo_min=601.755 tpo_max=635.042 a_med=0.000 b_med=0.000 makespan_ticks=83236172 makespan_ms=61.654 host_ms=61.860
[MB] probe=p7_dram_bw_read scope=1core cores=1 risc=n n=4096 tpo_med=195.196 tpo_min=195.196 tpo_max=195.196 a_med=0.000 b_med=0.000 makespan_ticks=799524 makespan_ms=0.592 host_ms=0.690 per_core_GBps_med=56.66 agg_GBps=56.7
[MB] probe=p7_dram_bw_read scope=all cores=110 risc=n n=4096 tpo_med=5656.811 tpo_min=2115.344 tpo_max=5660.047 a_med=0.000 b_med=0.000 makespan_ticks=23183553 makespan_ms=17.172 host_ms=17.377 per_core_GBps_med=1.96 agg_GBps=214.9
[MB] probe=p7_dram_bw_write scope=1core cores=1 risc=n n=4096 tpo_med=181.212 tpo_min=181.212 tpo_max=181.212 a_med=0.000 b_med=0.000 makespan_ticks=742245 makespan_ms=0.550 host_ms=0.618 per_core_GBps_med=61.03 agg_GBps=61.0
[MB] probe=p7_dram_bw_write scope=all cores=110 risc=n n=4096 tpo_med=2751.148 tpo_min=397.038 tpo_max=4684.556 a_med=0.000 b_med=0.000 makespan_ticks=19187943 makespan_ms=14.213 host_ms=14.392 per_core_GBps_med=4.02 agg_GBps=259.7
[MB] probe=p7_dram_bw_read scope=1core cores=1 risc=b+n n=2048 tpo_med=203.881 tpo_min=203.853 tpo_max=203.881 a_med=0.000 b_med=0.000 makespan_ticks=417548 makespan_ms=0.309 host_ms=0.410 per_core_GBps_med=54.25 agg_GBps=108.5
[MB] probe=p7_dram_bw_read scope=all cores=110 risc=b+n n=2048 tpo_med=4976.424 tpo_min=1779.118 tpo_max=5779.750 a_med=0.000 b_med=0.000 makespan_ticks=11836927 makespan_ms=8.768 host_ms=8.954 per_core_GBps_med=2.22 agg_GBps=421.0
[MB] probe=p8_barrier_empty scope=1core cores=1 risc=n n=1048576 tpo_med=32.000 tpo_min=32.000 tpo_max=32.000 a_med=16.000 b_med=16.000 makespan_ticks=33554497 makespan_ms=24.854 host_ms=24.966
[MB] probe=p8_barrier_empty scope=all cores=110 risc=n n=1048576 tpo_med=32.000 tpo_min=32.000 tpo_max=32.000 a_med=16.000 b_med=16.000 makespan_ticks=33554500 makespan_ms=24.854 host_ms=25.034
[MB] probe=p9_fsub_i2f scope=1core cores=1 risc=n n=262144 tpo_med=84.002 tpo_min=84.002 tpo_max=84.002 a_med=0.000 b_med=0.000 makespan_ticks=22020660 makespan_ms=16.311 host_ms=16.424
[MB] probe=p9_fsub_i2f scope=all cores=110 risc=n n=262144 tpo_med=88.003 tpo_min=84.002 tpo_max=88.004 a_med=0.000 b_med=0.000 makespan_ticks=23069664 makespan_ms=17.088 host_ms=17.270
[MB] probe=p9_fmul scope=1core cores=1 risc=n n=262144 tpo_med=90.000 tpo_min=90.000 tpo_max=90.000 a_med=0.000 b_med=0.000 makespan_ticks=23593057 makespan_ms=17.476 host_ms=17.581
[MB] probe=p9_fmul scope=all cores=110 risc=n n=262144 tpo_med=90.000 tpo_min=90.000 tpo_max=90.000 a_med=0.000 b_med=0.000 makespan_ticks=23593051 makespan_ms=17.476 host_ms=17.652
[MB] probe=p9_udiv scope=1core cores=1 risc=n n=262144 tpo_med=20.419 tpo_min=20.419 tpo_max=20.419 a_med=0.000 b_med=0.000 makespan_ticks=5352770 makespan_ms=3.965 host_ms=4.067
[MB] probe=p9_udiv scope=all cores=110 risc=n n=262144 tpo_med=20.418 tpo_min=20.417 tpo_max=20.419 a_med=0.000 b_med=0.000 makespan_ticks=5352750 makespan_ms=3.965 host_ms=4.136
[MB] smi_after board_type=p100a board_id=000004323191b005 aiclk= 800
```
