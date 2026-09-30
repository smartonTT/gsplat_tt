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
≈ 40 s of device time. Three full runs agreed to 3–4 significant digits.

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
| p4 `noc_async_write` 32 B → DRAM, precomputed addr | issue only | 40.6 | 41.5 | 16 per batch, barrier excluded |
| p4 same, barrier share | barrier ÷ 16 | 19.6 | 58.5 | 110 cores all hit the same 16 pages ⇒ bank hot-spot; see _acc row |
| **p4 `noc_async_write` + `get_noc_addr(page, TensorAccessor)`** | issue only | **55.6** | 55.6 | accessor address math = **15 cycles** |
| p4 same, barrier share | barrier ÷ 16 | 19.6 | 19.9 | ≈ 313 cycles per batch-of-16 barrier |
| p4 total per write (production shape) | issue + barrier/16 | **78.1** | 78.4 | plan §2.2 assumed 50 |
| **p5 `noc_async_read` 64 B DRAM, 1 outstanding + barrier** | exposed round trip | **499** | 478 | = 370 ns; sort_bin's blendrec read today |
| p5 same, 8 outstanding + 1 barrier | per read | **104** | 291 | 4.8× better on 1 core; 110 cores saturate at 1.4 G req/s (see below) |
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
| store+scatter (2²⁰ rec) | NCRISC only | 70.54 ms (90.8 cyc/rec) | 1.00 | 70.82 ms | 1.00 |
| store+scatter | BRISC only | 69.81 ms (89.9 cyc/rec) | — | 94.64 ms (slow tail, max 122 cyc/rec) | — |
| store+scatter | BRISC + NCRISC, N/2 each | 35.29 ms | **2.00×** | 73.28 ms | **0.97×** |
| store+read+scatter (2¹⁸ rec) | NCRISC only | 109.14 ms (562 cyc/rec) | 1.00 | 114.32 ms | 1.00 |
| store+read+scatter | BRISC + NCRISC, N/2 each | 54.57 ms | **2.00×** | 64.55 ms | **1.77×** |

- No measurable L1-port contention between the two movers (1-core split is exactly 2.00×;
  p1/p2 are identical on BRISC and NCRISC).
- At 110 cores a loop that does nothing but 32 B DRAM scatter saturates the chip:
  **~1.6–1.9 G scattered 32 B writes/s** (p4_acc on NOC1 alone 1.88 G/s; split over
  NOC0+NOC1 1.57 G/s). BRISC's NOC0 path shows a slow tail at that load.
- A latency-bound loop scales 1.77× at 110 cores.

### p7 — DRAM bandwidth (8 KiB pages, 8 outstanding)

| direction | movers | 1 core | 110 cores aggregate |
|---|---|---|---|
| read | NCRISC | 56.6 GB/s | **214.5 GB/s** |
| write | NCRISC | 61.0 GB/s | 259.6 GB/s |
| read | BRISC + NCRISC | 108.4 GB/s | **420.3 GB/s** |

Published p100a peak is 448 GB/s (7 × 64 GB/s GDDR6) `[P]`; p150 is 512 GB/s `[P]`.
Both NoCs together reach **94 %** of peak; **one NoC tops out at ~48 %** — a stage that
streams DRAM on a single mover can never exceed ~215 GB/s on this board.

## What this means for `sort_bucket_emit` (1 904 cycles/pair `[D]`, plan §2.3)

Measured cost of the three stall suspects named in the plan, per pair (K = 1.71 pairs per
gaussian):

| suspect | measured cost / pair | share of 1 904 |
|---|---|---|
| 8 volatile `pack_rec` stores | 8 × 2.5 = **20** | 1.1 % |
| 1-deep `blendrec` read + barrier | 478–499 ÷ 1.71 = **280–292** | 15 % |
| 32 B scatter write + barrier share | **75–78** | 4 % |
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
  read): (499 + 1.71 × 91) ÷ 1.71 ≈ **383 cycles/pair** — production is **5.0×** above it.
- Pure store+scatter ceiling: 91 cycles/record — production is 21× above.
- NoC write rate: production ≈ 2.4 M pairs ÷ 30.6 ms = 78 M writes/s vs a
  1.6–1.9 G/s chip cap — **~20× headroom**. DRAM bandwidth: ~1 % used.

## Implications for R3 (de-stall the emit loop)

1. **Change 1 (register-built record + one `__builtin_memcpy`) — expected gain ≈ 0, and
   the form written in the plan is a regression.** Volatile stores cost 2.5 cycles; an
   aligned inlined block costs 2.75. A plain `__builtin_memcpy((void*)…, rec, 32)` compiles
   to a `memcpy` call: +79 cycles/pair ≈ **+4 % (~+1.3 ms)**. Skip it, or if kept for
   readability use `__builtin_assume_aligned(dst, 16)` and expect neutral.
2. **Change 2 (8-deep `blendrec` prefetch) — the real stall lever of the three.** 499 → 104
   cycles per gaussian ⇒ −231 cycles/pair ≈ **−12 % of emit ≈ −3.5 to −4 ms** `[D]`. The
   110-core p5_d8 slowdown (291) is a request-rate cap at 1.4 G reads/s; production issues
   ~30× fewer, so the 1-core figure applies.
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

- Expected scaling for `sort_bucket_emit`: **1.77–2.0×** of the zone `[M]` (its NoC write
  and read rates are 20–30× below the caps where scaling stops) ⇒ 30.6 → ~16–17 ms,
  **−13 to −15 ms/view** `[D]`, before R3. With R3 change 2 + soft-float removal landed
  first the absolute gain shrinks proportionally, but the factor holds.
- A 50/50 split is right for latency/compute-bound stages. For a stage that is itself
  NoC-write-saturating at 110 cores, dual-mover gives **nothing** (0.97×) — check a
  stage's aggregate small-write rate against ~1.6 G/s before splitting it.
- DRAM-streaming stages on one mover are capped at ~215 GB/s; splitting them across both
  NoCs doubles that (420 GB/s). Irrelevant to the current ~1 %-of-DRAM workload, but it
  matters for R7 if it moves bulk data.
- BRISC (NOC0) showed a 1.3× slow tail under saturating 110-core scatter; keep BRISC's
  share ≤ 50 % for write-heavy stages.

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
[MB] clock tick_mhz=1350.0 (ticks 1073741830 vs 268435456, host 0.7955 s vs 0.1989 s)
[MB] probe=p0_loop_overhead scope=1core cores=1 risc=n n=1048576 tpo_med=0.625 tpo_min=0.625 tpo_max=0.625 a_med=0.000 b_med=0.000 makespan_ticks=655394 makespan_ms=0.485 host_ms=0.558
[MB] probe=p0_loop_overhead scope=all cores=110 risc=n n=1048576 tpo_med=0.625 tpo_min=0.625 tpo_max=0.625 a_med=0.000 b_med=0.000 makespan_ticks=655397 makespan_ms=0.485 host_ms=0.638
[MB] probe=p1_l1_store_volatile scope=1core cores=1 risc=n n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621484 makespan_ms=1.942 host_ms=2.022
[MB] probe=p1_l1_store_volatile scope=all cores=110 risc=n n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621487 makespan_ms=1.942 host_ms=2.100
[MB] probe=p1_l1_store_volatile scope=1core cores=1 risc=b n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621482 makespan_ms=1.942 host_ms=2.022
[MB] probe=p1_l1_store_volatile scope=all cores=110 risc=b n=1048576 tpo_med=2.500 tpo_min=2.500 tpo_max=2.500 a_med=0.000 b_med=0.000 makespan_ticks=2621485 makespan_ms=1.942 host_ms=2.091
[MB] probe=p2_l1_store_block_memcpy scope=1core cores=1 risc=n n=1048576 tpo_med=12.375 tpo_min=12.375 tpo_max=12.375 a_med=0.000 b_med=0.000 makespan_ticks=12976176 makespan_ms=9.612 host_ms=9.695
[MB] probe=p2_l1_store_block_memcpy scope=all cores=110 risc=n n=1048576 tpo_med=12.375 tpo_min=12.375 tpo_max=12.375 a_med=0.000 b_med=0.000 makespan_ticks=12976179 makespan_ms=9.612 host_ms=9.770
[MB] probe=p2_l1_store_block_typed scope=1core cores=1 risc=n n=1048576 tpo_med=9.000 tpo_min=9.000 tpo_max=9.000 a_med=0.000 b_med=0.000 makespan_ticks=9437218 makespan_ms=6.991 host_ms=7.070
[MB] probe=p2_l1_store_block_typed scope=all cores=110 risc=n n=1048576 tpo_med=9.000 tpo_min=9.000 tpo_max=9.000 a_med=0.000 b_med=0.000 makespan_ticks=9437218 makespan_ms=6.991 host_ms=7.160
[MB] probe=p2_l1_store_block_aligned scope=1core cores=1 risc=n n=1048576 tpo_med=2.750 tpo_min=2.750 tpo_max=2.750 a_med=0.000 b_med=0.000 makespan_ticks=2883617 makespan_ms=2.136 host_ms=2.212
[MB] probe=p2_l1_store_block_aligned scope=all cores=110 risc=n n=1048576 tpo_med=2.750 tpo_min=2.750 tpo_max=2.750 a_med=0.000 b_med=0.000 makespan_ticks=2883618 makespan_ms=2.136 host_ms=2.291
[MB] probe=p3_l1_load scope=1core cores=1 risc=n n=1048576 tpo_med=12.334 tpo_min=12.334 tpo_max=12.334 a_med=4.071 b_med=8.263 makespan_ticks=12933659 makespan_ms=9.581 host_ms=9.660
[MB] probe=p3_l1_load scope=all cores=110 risc=n n=1048576 tpo_med=12.334 tpo_min=12.334 tpo_max=12.335 a_med=4.071 b_med=8.263 makespan_ticks=12933665 makespan_ms=9.581 host_ms=9.749
[MB] probe=p4_noc_write_issue_pre scope=1core cores=1 risc=n n=1048576 tpo_med=62.563 tpo_min=62.563 tpo_max=62.563 a_med=40.625 b_med=19.562 makespan_ticks=65601585 makespan_ms=48.594 host_ms=48.678
[MB] probe=p4_noc_write_issue_pre scope=all cores=110 risc=n n=1048576 tpo_med=105.977 tpo_min=71.083 tpo_max=152.356 a_med=41.508 b_med=58.440 makespan_ticks=159756564 makespan_ms=118.340 host_ms=118.507
[MB] probe=p4_noc_write_issue_acc scope=1core cores=1 risc=n n=1048576 tpo_med=78.063 tpo_min=78.063 tpo_max=78.063 a_med=55.562 b_med=19.562 makespan_ticks=81854542 makespan_ms=60.634 host_ms=60.739
[MB] probe=p4_noc_write_issue_acc scope=all cores=110 risc=n n=1048576 tpo_med=78.317 tpo_min=76.283 tpo_max=78.739 a_med=55.569 b_med=19.816 makespan_ticks=82563766 makespan_ms=61.159 host_ms=61.402
[MB] probe=p5_noc_read_rt_d1 scope=1core cores=1 risc=n n=1048576 tpo_med=499.179 tpo_min=499.179 tpo_max=499.179 a_med=0.000 b_med=0.000 makespan_ticks=523427277 makespan_ms=387.728 host_ms=387.840
[MB] probe=p5_noc_read_rt_d1 scope=all cores=110 risc=n n=1048576 tpo_med=477.864 tpo_min=447.084 tpo_max=478.449 a_med=0.000 b_med=0.000 makespan_ticks=501689637 makespan_ms=371.626 host_ms=371.878
[MB] probe=p5_noc_read_rt_d8 scope=1core cores=1 risc=n n=1048576 tpo_med=103.759 tpo_min=103.759 tpo_max=103.759 a_med=0.000 b_med=0.000 makespan_ticks=108799182 makespan_ms=80.593 host_ms=80.715
[MB] probe=p5_noc_read_rt_d8 scope=all cores=110 risc=n n=1048576 tpo_med=290.688 tpo_min=202.033 tpo_max=356.648 a_med=0.000 b_med=0.000 makespan_ticks=373972448 makespan_ms=277.020 host_ms=277.217
[MB] probe=p6_dual_mover scope=1core cores=1 risc=n n=1048576 tpo_med=90.813 tpo_min=90.813 tpo_max=90.813 a_med=0.000 b_med=0.000 makespan_ticks=95223878 makespan_ms=70.537 host_ms=70.623
[MB] probe=p6_dual_mover scope=all cores=110 risc=n n=1048576 tpo_med=91.149 tpo_min=89.104 tpo_max=91.177 a_med=0.000 b_med=0.000 makespan_ticks=95606295 makespan_ms=70.820 host_ms=70.991
[MB] probe=p6_dual_mover scope=1core cores=1 risc=b n=1048576 tpo_med=89.873 tpo_min=89.873 tpo_max=89.873 a_med=0.000 b_med=0.000 makespan_ticks=94238791 makespan_ms=69.807 host_ms=69.898
[MB] probe=p6_dual_mover scope=all cores=110 risc=b n=1048576 tpo_med=92.114 tpo_min=90.360 tpo_max=122.800 a_med=0.000 b_med=0.000 makespan_ticks=128765023 makespan_ms=95.383 host_ms=95.567
[MB] probe=p6_dual_mover scope=1core cores=1 risc=b+n n=524288 tpo_med=90.869 tpo_min=89.932 tpo_max=90.869 a_med=0.000 b_med=0.000 makespan_ticks=47641762 makespan_ms=35.291 host_ms=35.468
[MB] probe=p6_dual_mover scope=all cores=110 risc=b+n n=524288 tpo_med=129.480 tpo_min=97.498 tpo_max=188.514 a_med=0.000 b_med=0.000 makespan_ticks=98835432 makespan_ms=73.212 host_ms=73.414
[MB] probe=p6_dual_mover_rd scope=1core cores=1 risc=n n=262144 tpo_med=562.040 tpo_min=562.040 tpo_max=562.040 a_med=0.000 b_med=0.000 makespan_ticks=147335303 makespan_ms=109.139 host_ms=109.244
[MB] probe=p6_dual_mover_rd scope=all cores=110 risc=n n=262144 tpo_med=587.541 tpo_min=556.063 tpo_max=589.223 a_med=0.000 b_med=0.000 makespan_ticks=154461374 makespan_ms=114.417 host_ms=114.595
[MB] probe=p6_dual_mover_rd scope=1core cores=1 risc=b+n n=131072 tpo_med=562.083 tpo_min=548.704 tpo_max=562.083 a_med=0.000 b_med=0.000 makespan_ticks=73673296 makespan_ms=54.573 host_ms=54.677
[MB] probe=p6_dual_mover_rd scope=all cores=110 risc=b+n n=131072 tpo_med=651.652 tpo_min=620.570 tpo_max=664.500 a_med=0.000 b_med=0.000 makespan_ticks=87097351 makespan_ms=64.517 host_ms=64.726
[MB] probe=p7_dram_bw_read scope=1core cores=1 risc=n n=4096 tpo_med=195.303 tpo_min=195.303 tpo_max=195.303 a_med=0.000 b_med=0.000 makespan_ticks=799962 makespan_ms=0.593 host_ms=0.685 per_core_GBps_med=56.63 agg_GBps=56.6
[MB] probe=p7_dram_bw_read scope=all cores=110 risc=n n=4096 tpo_med=5669.665 tpo_min=2097.920 tpo_max=5672.425 a_med=0.000 b_med=0.000 makespan_ticks=23234251 makespan_ms=17.211 host_ms=17.379 per_core_GBps_med=1.95 agg_GBps=214.5
[MB] probe=p7_dram_bw_write scope=1core cores=1 risc=n n=4096 tpo_med=181.212 tpo_min=181.212 tpo_max=181.212 a_med=0.000 b_med=0.000 makespan_ticks=742245 makespan_ms=0.550 host_ms=0.654 per_core_GBps_med=61.03 agg_GBps=61.0
[MB] probe=p7_dram_bw_write scope=all cores=110 risc=n n=4096 tpo_med=2747.035 tpo_min=398.172 tpo_max=4685.517 a_med=0.000 b_med=0.000 makespan_ticks=19191877 makespan_ms=14.216 host_ms=14.426 per_core_GBps_med=4.03 agg_GBps=259.6
[MB] probe=p7_dram_bw_read scope=1core cores=1 risc=b+n n=2048 tpo_med=204.125 tpo_min=204.074 tpo_max=204.125 a_med=0.000 b_med=0.000 makespan_ticks=418047 makespan_ms=0.310 host_ms=0.417 per_core_GBps_med=54.18 agg_GBps=108.4
[MB] probe=p7_dram_bw_read scope=all cores=110 risc=b+n n=2048 tpo_med=4895.005 tpo_min=1534.322 tpo_max=5788.234 a_med=0.000 b_med=0.000 makespan_ticks=11854303 makespan_ms=8.781 host_ms=8.991 per_core_GBps_med=2.26 agg_GBps=420.3
[MB] probe=p8_barrier_empty scope=1core cores=1 risc=n n=1048576 tpo_med=32.000 tpo_min=32.000 tpo_max=32.000 a_med=16.000 b_med=16.000 makespan_ticks=33554497 makespan_ms=24.855 host_ms=24.945
[MB] probe=p8_barrier_empty scope=all cores=110 risc=n n=1048576 tpo_med=32.000 tpo_min=32.000 tpo_max=32.000 a_med=16.000 b_med=16.000 makespan_ticks=33554497 makespan_ms=24.855 host_ms=25.030
[MB] probe=p9_fsub_i2f scope=1core cores=1 risc=n n=262144 tpo_med=84.002 tpo_min=84.002 tpo_max=84.002 a_med=0.000 b_med=0.000 makespan_ticks=22020659 makespan_ms=16.312 host_ms=16.423
[MB] probe=p9_fsub_i2f scope=all cores=110 risc=n n=262144 tpo_med=88.003 tpo_min=84.002 tpo_max=88.004 a_med=0.000 b_med=0.000 makespan_ticks=23069616 makespan_ms=17.089 host_ms=17.267
[MB] probe=p9_fmul scope=1core cores=1 risc=n n=262144 tpo_med=90.000 tpo_min=90.000 tpo_max=90.000 a_med=0.000 b_med=0.000 makespan_ticks=23593055 makespan_ms=17.477 host_ms=17.577
[MB] probe=p9_fmul scope=all cores=110 risc=n n=262144 tpo_med=90.000 tpo_min=90.000 tpo_max=90.000 a_med=0.000 b_med=0.000 makespan_ticks=23593051 makespan_ms=17.477 host_ms=17.644
[MB] probe=p9_udiv scope=1core cores=1 risc=n n=262144 tpo_med=20.419 tpo_min=20.419 tpo_max=20.419 a_med=0.000 b_med=0.000 makespan_ticks=5352771 makespan_ms=3.965 host_ms=4.065
[MB] probe=p9_udiv scope=all cores=110 risc=n n=262144 tpo_med=20.418 tpo_min=20.417 tpo_max=20.419 a_med=0.000 b_med=0.000 makespan_ticks=5352750 makespan_ms=3.965 host_ms=4.150
[MB] smi_after board_type=p100a board_id=000004323191b005 aiclk= 800
```
