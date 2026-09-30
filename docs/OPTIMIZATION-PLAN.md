# gsplat-tt — optimization campaign plan

**Written 2026-09-30.** Inputs: `docs/CONTEXT.md` (context recovery),
`docs/benchmark-baseline.md` (reproducible 173.30 ms baseline),
`docs/slack-notes.md` (Slack DM D0C1CV1AJJV), `opt/tt-splat-assessment.md`,
`opt/FINAL-REPORT.md`, `opt/sort-l1-resident-plan.md` (refuted-premises ledger),
`opt/cpu-vs-tt-comparison.md`, and the `render/` sources.

This document changes no rendering code. It is the campaign plan: a roofline
model, the bottlenecks with evidence, the missing evidence, a ranked lever list,
and three implementer-ready tasks.

**Provenance rule.** Every number below is tagged `[M]` measured on our bench,
`[D]` derived from measured numbers, or `[P]` vendor/literature published and
**not** measured. No `[P]` number may enter the status HTML as a project result.

---

## 0. TL;DR — the three findings that reopen the frozen project

| # | Finding | Evidence |
|---|---|---|
| **1** | **Half the data movers are idle for 119 ms of every 173 ms frame.** Every kernel in `tile_assign_device.cpp` (7) and `sort_device.cpp` (8) is created with `DataMovementProcessor::RISCV_1` — NCRISC only. Every kernel in `gather_visible_device.cpp` (2) is `RISCV_0` — BRISC only. So the NCRISC-bound stages (TA 21.9 ms + sort 61.2 ms = **83.1 ms**) run with BRISC idle, and the BRISC-bound gather (**35.8 ms**) runs with NCRISC idle. | `[M]` `grep processor = DataMovementProcessor` across `render/host/*_device.cpp`; zone table `docs/CONTEXT.md` §2 |
| **2** | **NCRISC is stall-saturated, not work-saturated.** `sort_bucket_emit` costs **1 904 cycles per (gaussian,tile) pair** for a loop body of ~45 instructions — ~42 cycles/instruction on a single-issue in-order core. Across all NCRISC stages the RISC runs at roughly **7 % of its own instruction-issue roofline**. "100 % occupancy" in the freeze report means *parked on a stall*, not *doing work at the limit*. | `[D]` 30.6 ms × 1.35 GHz ÷ 21.7k pairs/core; `[M]` iter-129 ablation (null the 8 per-pair L1 stores ⇒ −57 %) |
| **3** | **Every capacity and bandwidth ceiling is essentially untouched.** DRAM moves ~0.95 GB/view against 512 GB/s ⇒ **~1 % of the DRAM roofline**. The entire view's pair payload (2.4 M × 32 B = 77 MB) fits in aggregate L1 (165 MB) ⇒ a fully L1-resident tile-owner pipeline is **capacity-feasible**, contradicting the freeze's "L1 knife-edge" framing (which was about a *fused double-buffered* kernel, not about pair capacity). | `[D]` traffic model §2.1; `[P]` 512 GB/s, 1.5 MB L1/Tensix |

The freeze conclusion — *"173 ms is the architectural floor for this pipeline
design"* — rests on reading a saturated **occupancy** counter as a saturated
**resource**. It is not a hardware floor. The project never measured a single
hardware ceiling, which is why it mistook stall plateaus for floors.

**Honest target ladder** (all `[D]`, to be re-anchored on a p150):

| Milestone | ms/view | FPS | What it takes |
|---|---|---|---|
| Baseline `[M]` | 173.30 | 5.77 | — (p100a `yyzo-bh-07`) |
| **T1 — mover balance** | ~115 | 8.7 | levers R2 + R3 (dual-mover + de-stall) |
| **T2 — beat our own CPU** | ≤55 | 18.1 | + R4/R7/R8 (16 B record, L1-resident tile-owner) |
| T3 — published GPU class | ≤11 | 90 | + R9 (FPU/GEMM Q field). Not reachable without the algorithm change. |

T2 is the campaign's real success criterion: **the same source, same scene,
same 30 views renders at 55.2 ms/view on an Apple M4 Pro CPU `[M]`** — the
Tenstorrent device is currently **3.1× slower than a laptop CPU**. Beating that
is a prerequisite to any GPU claim and is reachable with the levers below.
Published GPU reference: INRIA 3DGS on an RTX A6000, bicycle, 93 FPS at 1080p
`[P]` ⇒ 5.44 ms at 1024² under an optimistic linear-in-pixels normalization
`[D]`. A second, softer anchor: the Kovinić/Stojković TT line reached ≈1.6×
slower than a GTX 4060 `[P]`, which says the remaining gap is algorithmic.

---

## 1. The workload, in numbers

| Quantity | Value | Tag |
|---|---|---|
| Scene | `scenes/bicycle.ply`, **6 131 954** gaussians, SH degree 0 | [M] |
| Resolution / tiles | 1024×1024 = 1.049 Mpx; 32×32 px tiles ⇒ **1024 tiles** | [M] |
| Cores used | **110** (= 11×10; consistent with one harvested column of the p100a's 12×10 Tensix grid) | [M] |
| Visible gaussians `G_vis` | **≈1.4 M (23 %)** = `P_kept / K` | [D] |
| Pairs kept `P_kept` | **2.4 M – 3.7 M** per view | [M] |
| Fan-out `K` = pairs/gaussian | **≈1.71** (1.3–2.0 over 30 views) | [M] |
| Pairs per core | ~21.7 k (at `P_kept` = 2.4 M) | [M] |
| Per-pair record | 32 B, PACK2 (two per 64 B page) | [M] |
| Bucket capacity | `kBucketFit = 8192` records/tile; heaviest tile ≈26.5 k | [M] |
| Overflow | 78–113 tiles/view exceed `kBucketFit`, ~330 k records (14 %); overflow region ~33.9 MB/view | [M] |
| Programs per view | ~15 device programs, one in-order CQ | [M] |
| Host gap | 4.86 ms/view (image unpack 3.24, pybind 0.76, D2H 0.55, head 0.18) | [M] |

Per-view zone makespans (busiest core, iter-131 Tracy) `[M]`, annotated with the
RISC that actually runs them (from `CreateKernel` in `render/host/`):

| Stage | zone | ms | runs on | movers idle |
|---|---|---:|---|---|
| gather | `proj_scatter` | 20.8 | BRISC | **NCRISC + 3 TRISC** |
| gather | `proj_count` | 15.0 | BRISC | **NCRISC + 3 TRISC** |
| tile_assign | `ta_bucket_scatter` | 12.2 | NCRISC | **BRISC + 3 TRISC** |
| tile_assign | `ta_gauss_aabb` | 9.7 | NCRISC | **BRISC + 3 TRISC** |
| sort | `sort_bucket_emit` | 30.6 | NCRISC | **BRISC + 3 TRISC** |
| sort | `sort_subchunk_mat` | 19.6 | NCRISC | **BRISC + 3 TRISC** |
| sort | `sort_tile_depth` | 7.2 | NCRISC | **BRISC + 3 TRISC** |
| sort | `sort_subchunk_dir` | 2.4 | NCRISC | **BRISC + 3 TRISC** |
| sort | `sort_bin_hist` | 1.4 | NCRISC | **BRISC + 3 TRISC** |
| cull | `tile_l1_cull_rd` | 22.0 | NCRISC | — (all 5 used) |
| cull | `tile_mb_mask` | 22.4 | TRISC | — |
| blend | `tile_blend_load` | 28.1 | NCRISC | — |
| blend | `rd_l1_bulk` | 27.9 | NCRISC | — |
| blend | `tile_blend_sfpu` | 28.8 | TRISC | — |
| project | `pfwc` 1.9, `mcam` 0.7 | 2.6 | NCRISC+TRISC+BRISC | — |

RISC makespans: **BRISC-FW 159 ms** (nominal critical path, ~76 ms of it parked
in barriers), **NCRISC-KERNEL 125 ms** (the pole), **BRISC-KERNEL 85.4 ms**,
**TRISC-KERNEL 52 ms** (off-path) `[M]`. Zone makespans do not sum to RISC
makespans — each zone's busiest core differs.

### 1.1 Host-side stage attribution (new, 2026-09-30)

`docs/stage-timing-2026-09-30.md` closes gap E2: host wall-clock spans now cover
the whole frame with a 0.001 ms residual `[M]`, at −0.08 % instrumentation cost.

| stage | ms | % | contents |
|---|---:|---:|---|
| head | 0.19 | 0.1 | py buffers + output memset |
| **project** | **36.96** | 21.3 | fused means_cam+pfwc, then `gather_visible` |
| **tile_assign** | **21.78** | 12.6 | bbox + scan + scatter |
| **sort** | **46.74** | 27.0 | `sort_and_bin_tt` minus its fused cull/blend continuation |
| **blend** | **63.60** | 36.7 | blend enqueue **+ the single CQ drain** (absorbs cull, publish, materialize) |
| assemble | 3.22 | 1.9 | bf16 microblock tiles → fp32 HWC |
| d2h + tail + blend_setup + cull-launch | 0.63 | 0.4 | |
| **sum** | **173.10** | | `view_total` 173.10, residual +0.001 |

These are **host spans containing device execution**, not host CPU time; they
line up with the device zone table above (project 36.96 ≈ `proj_scatter` 20.8 +
`proj_count` 15.0 + `pfwc`/`mcam` 2.6; tile_assign 21.78 ≈ 12.2 + 9.7). The
`sort` span at 46.74 vs a 61.2 ms zone sum reflects that `sort_subchunk_mat` is
piped ahead of the blend drain and lands in the `blend` bucket.

**Correction to that document's "top optimization target" read.** It concludes
`sort` is "~36 ms of single-purpose host CPU binning … with the device idle
throughout". That is not what `bin_ms` measures. `T.bin_ms` is
`(t_bin1 − t_bin0) + launch_bin(1,true)` (`render/host/sort_device.cpp:2081,2106`),
and `launch_bin(1,true)` is the **device** Pass-B scatter — the very kernel Tracy
names `sort_bucket_emit` at 30.6 ms `[M]`. So the ~30–36 ms `bin` figure is
dominated by device time, and the device is not idle. The genuinely host-side
part of the span is Pass-A's blocking `EnqueueReadMeshBuffer` of the 110×1024
histogram plus `host_bin_layout_from_hist` — consistent with the separately
measured 4.86 ms inter-frame host gap `[M]` (iter-136). Two real host items do
survive the correction and are worth levers: that **mid-pipeline blocking D2H**,
which serialises host and device inside the sort stage, and **`publish` ≈9–10 ms
of H2D** per view.

---

## 2. Roofline model

Hardware ceilings. Blackhole figures are vendor-published `[P]` and **must be
re-verified on the box** (`tt-smi -s`, `compute_with_storage_grid_size()`,
`AICLK`) before any of them is used in a published claim.

| Resource | p100a (our baseline board) | p150a (charter board) | Tag |
|---|---|---|---|
| Tensix cores | 120 physical / **110 usable (measured)** | 140 physical / 130 expected usable | [P]/[M] |
| Clock (AI) | 1.35 GHz | 1.35 GHz | [P] |
| L1 per Tensix | 1.5 MB ⇒ 165 MB aggregate over 110 cores | 1.5 MB ⇒ 195 MB over 130 | [P] |
| DRAM | GDDR6, 512 GB/s | GDDR6, 512 GB/s | [P] |
| NoC | 2 independent NoCs; BRISC→NOC0, NCRISC→NOC1 by default | same | [P] |
| Data movers per core | 2 (BRISC, NCRISC) | 2 | [P] |
| Compute RISCs per core | 3 (TRISC0/1/2 = unpack/math/pack), 32-lane SFPU + matrix FPU | 3 | [P] |

### 2.1 DRAM bandwidth — utilization ~1 %

Per-view traffic model `[D]` (to be confirmed by a counter pass, §3):

| Buffer | bytes/view |
|---|---:|
| gaussian attributes read by project+pfwc (6.13 M × ~52 B) | ~319 MB |
| `blendrec` AoS writes (1.4 M × 64 B) | ~90 MB |
| TA pair streams (`gids`/`tids`/`keep`, write + read) | ~72 MB |
| `buf_l1_recs` write + radix read + slab write + cull read + blend read (5 × 2.4 M × 32 B) | ~385 MB |
| keys/ids (write + read) | ~38 MB |
| overflow region | ~34 MB |
| output image D2H | ~6 MB |
| **total** | **~0.95 GB** |

At 512 GB/s that is **1.85 ms**. The frame is 173.30 ms ⇒ effective **5.5 GB/s,
about 1.1 % of the DRAM roofline.** DRAM bandwidth is not the bound and will not
become the bound after a 10× speedup. Corollary: **halving record bytes is worth
~0 ms of bandwidth** — its value is entirely in RISC store count and L1
footprint (§2.3, §2.4).

### 2.2 NoC — utilization ~0.05 % of bandwidth, but transaction *issue* is real

0.95 GB over 110 cores over 173 ms = 50 MB/s per core against ~86 GB/s per link
per direction `[P]` ⇒ bandwidth is irrelevant. What is not irrelevant is
transaction count: ~10 M NoC transactions/view ⇒ ~91 k per core, each costing the
issuing RISC tens of cycles of command-register writes. At an assumed 50 cycles
that is ~3.4 ms/core `[D]` — 2 % of the frame. **The per-transaction issue cost
is one of the two candidate explanations for finding #2 and must be measured, not
assumed** (§3, tool T-A).

### 2.3 RISC instruction issue — utilization ~7 %

NCRISC is single-issue, in-order. Its busy window is 125 ms ⇒ **169 M cycles per
core per view** at 1.35 GHz `[D]`.

Useful instruction estimate for everything NCRISC runs, per core per view `[D]`:
emit ~0.98 M (21.7 k pairs × ~45 instr), counting-sort + radix ~3 M, materialize
~2 M, TA K1+K2 ~1.2 M, bulk readers ~1 M ⇒ **~9–12 M instructions ≈ 7–9 ms at
IPC 1**. Measured 125 ms. **NCRISC therefore spends ~92 % of the pole stalled.**

The sharpest single instance, and the one the freeze called a floor:

```
sort_bucket_emit = 30.6 ms/view (busiest core)   [M]
              ÷ 21 700 pairs on that core        [M]
              × 1.35 GHz                         [P]
            = 1 904 cycles per pair              [D]
```

The per-pair loop body (`render/kernels/dataflow/sort_bin.cpp:442-559`) is: a
shift/mask tile decode (proved free, iter-129), two soft-float subtracts, eight
`volatile uint32_t` stores into `l1_scratch`, two `ksp`/`isp` stores, one
`noc_async_write` enqueued, and one `noc_async_write_barrier()` per 16 pairs.
Call it 45 instructions. **42 cycles per instruction.** The iter-129 ablation
that "proved the 32 B L1 store floor" nulled the eight stores and saw −57 %;
that ablation cannot distinguish *store cost* from *everything the compiler
dead-code-eliminates once the stores are gone* (the soft-float subtracts, the
`memcpy` bit-casts, and most of `pack_invariants`). The conclusion "this is a
store floor" is therefore unproven, and the 42-cycles-per-instruction ratio says
whatever it is, it is ~20–40× away from the hardware.

Three named suspects, in order of prior probability:

1. **`volatile` word stores to L1.** `volatile` forbids coalescing, reordering
   and register promotion, so eight separate ordered stores are emitted where
   one 32 B block move would do; on a baby RISC with no store buffer each may
   stall to completion.
2. **Unpipelined NoC read latency.** `blendrec[g]` is read with a *single*
   outstanding request followed immediately by `noc_async_read_barrier()`
   (`sort_bin.cpp:499-500`) — once per gaussian, i.e. once per 1.71 pairs. One
   exposed DRAM round trip per gaussian is ~600 cycles/pair if a round trip is
   ~1 µs. iter-137 already proved multi-buffering fixes this class (8 pages
   before one barrier on `ta_gauss_aabb`).
3. **NoC write issue + per-batch barrier** (§2.2).

All three are removable. None is a floor.

### 2.4 SRAM capacity — the L1-resident pipeline is feasible

| Footprint | bytes | % of budget |
|---|---:|---:|
| Whole view's pair payload @32 B (2.4 M) | 77 MB | 47 % of 165 MB aggregate L1 |
| Same @16 B | 38 MB | 23 % |
| Per core if tiles are owned: 1024/110 = 9.3 tiles × 2.34 k pairs × 32 B | 700 KB | 47 % of 1.5 MB |
| Same @16 B | 350 KB | 23 % |
| Heaviest single tile (26.5 k × 32 B) | 848 KB | 57 % |
| Heaviest single tile @16 B | 424 KB | 28 % |

`[D]` throughout. **A tile-owner architecture — each core permanently owns ~9
tiles and holds those tiles' complete pair lists in its own L1 from emit through
blend — fits, including the heaviest overflow tile, with ≥40 % headroom.** The
freeze's "L1 is exactly 1.5 MB and the fused kernel fits only on a ~4 %
knife-edge" was measured for a *double-buffered fused emit+cull+blend* kernel.
That is a different, much harder shape than simple tile ownership. The capacity
objection to L1 residency does not survive this arithmetic.

### 2.5 Data-mover balance — the hard bound on pure rebalancing

Mover-busy per view: NCRISC 125 ms + BRISC 85.4 ms = **210 ms across 2 movers**
`[M]` ⇒ a perfectly balanced pole is **105 ms ⇒ 9.5 FPS** `[D]`. That is the
ceiling of rebalancing with zero work deletion. The reachable near-term number
is looser: halving the makespan of the single-mover stages only (gather 35.8 →
17.9, TA+sort 83.1 → 41.6) gives **173 → ~115 ms** `[D]`. Going below ~105 ms
requires deleting stalls or work, which is what §2.3 says is available.

### 2.6 Compute engines — 3 of 5 RISCs idle for 119 ms; the FPU is unused

TRISC-KERNEL is 52 ms of a 173 ms frame `[M]`, and during the entire
project/gather/TA/sort window (~119 ms) the three TRISCs have **no kernel at
all**. The matrix FPU is used nowhere in the pipeline. The per-pair scalar work
that is strangling NCRISC (float subtracts, `fp32→UNORM16` conversions) is
exactly the kind of 32-wide elementwise work an idle SFPU does ~32× faster.

---

## 3. Missing evidence, and the profiling that gets it

The project's measurement stack can see per-view device-zone makespans and
nothing else. Five gaps block the levers below. Two require new tooling.

| # | Missing | Why it blocks | How to get it |
|---|---|---|---|
| **E1** | **Any hardware ceiling.** No measured L1 load/store latency, NoC issue cost, NoC round-trip latency, DRAM achieved bandwidth, or dual-mover scaling factor. | Every "floor" in `opt/FINAL-REPORT.md` is a plateau with no ceiling next to it. R2's expected 2× and R3's expected −8..−18 ms are both unquantified without it. | **Build tool T-A** (§5, task 1): a standalone microbenchmark kernel suite. |
| **E2** | ~~Per-stage host timers~~ | — | **DELIVERED 2026-09-30** — `render/host/stage_timers.{h,cpp}`, results in §1.1 and `docs/stage-timing-2026-09-30.md`. Follow-up: push the same span discipline *inside* `project` (36.96 ms is still one bucket) and split the `blend` drain from blend proper. |
| **E3** | **Per-RISC busy vs stall split.** Tracy gives zone makespan; nothing separates "issuing instructions" from "parked on a load/barrier". | Finding #2 is derived, not directly measured. The de-stall work (R3) needs a stall counter to target. | **Build tool T-B**: permanent nested phase sub-zones in the three heaviest dataflow kernels, plus a cycle-counter read (`riscv_read_cycle`) bracketing barrier waits, reported as a `[ZONES]` digest line per view. The project kept adding these and reverting them — make them permanent and compile-time gated. |
| **E4** | **DRAM traffic counters.** §2.1 is a model, not a measurement. | The 1 % DRAM utilization claim is the basis for "record bytes don't matter for bandwidth". | Host-side: sum `EnqueueWrite/Read` bytes + a per-kernel static byte model emitted as `[TRAFFIC]`. Cheap; no device cost. |
| **E5** | **Tile-load distribution beyond the mean.** We know `P_kept`, `K`≈1.71, `max_tile`≈26.5 k, and the overflow tile count. We do not know the per-tile pair histogram or the per-tile *contribution* distribution. | Gates R12 (whole-tile skip) and sizing for R7 (tile ownership). | Host-side histogram over `r.counts[]` — the array already exists in `sort_device.cpp:1148`. Two lines of code. |

**Correction to a prior plan item.** `opt/tt-splat-assessment.md` §5.2 proposes
single-destination scatter + neighbourhood gather, gated on "a pairs-per-gaussian
histogram we have never measured". That measurement already exists: **K ≈ 1.71**
`[M]` (`opt/sort-l1-resident-plan.md`, iter-137 scope). With a mean fan-out of
1.71, collapsing to one destination can remove at most 42 % of pairs while adding
a neighbourhood gather — so the lever's ceiling is ~13 ms on `sort_bucket_emit` +
~5 ms on `ta_bucket_scatter`, against a new gather cost. It drops from rank 2 to
the bottom of the list. (The "median K=4" in `docs/CONTEXT.md` is *microblocks
per tile*, a different K; the two were conflated.)

---

## 4. Ranked levers

Ranked by expected gain per unit effort. Effort: **S** ≤1 day, **M** 2–4 days,
**L** ≥1 week. Every "expected" is `[D]` and must be replaced by `[M]` from the
30-view bicycle bench before it enters the status HTML.

| # | Lever | Expected | Effort | Risk | Quality |
|---|---|---|---|---|---|
| **R1** | Hardware-ceiling microbenchmark suite (tooling) | 0 ms direct; gates R2–R5 | S | none | n/a |
| **R2** | Split single-mover stages across BRISC **and** NCRISC | **−30 to −55 ms** | M | low–med | bit-identical |
| | **Status (task #22 T-C pilot, measured, yyzo-bh-07 p100a):** `sort_bucket_emit` split at mid of each core's page range, BRISC `[lo, mid)` + NCRISC `[mid, hi)`, sharing the core's layout (NCRISC cursors start at the count pass's snapshot at mid; no host layout change). `bin_emit` 16.15 → 9.00 ms, emit makespan 15.92 → 8.43 (1.89x), frame 152.99 → 145.42 ms, bit-identical. Next: `proj_scatter`/`proj_count` (BRISC, 35.8), tile_assign (NCRISC, 19.7), `sort_subchunk_mat` (12.4). See `docs/sort-emit-dual-mover-2026-09-30.md`. | | | | |
| **R3** | De-stall the emit inner loop (register-built record, prefetched `blendrec`, batched `ksp`/`isp`) | **−8 to −18 ms** | S–M | low | bit-identical |
| | **Status (task #20, measured, yyzo-bh-07 p100a):** the dominant stall was soft-float, not stores or reads. NCRISC has no FPU; the emit's UNORM16 pack + tile-local mean were ~24 libgcc float calls per gaussian (~15 ms by ablation). Integer bit-exact replacement + blendrec page prefetch + flush-not-ack write waits: `bin_emit` 30.26 → 16.15 ms, frame 167.1 → 152.6 ms, bit-identical. Register-built record (8 stores → memcpy) was flat. Other NCRISC/BRISC kernels should be audited for the same soft-float cost. | | | | |
| **R4** | Per-pair record 32 B → 16 B (fp16 tile-local conic + mean) | −8 to −15 ms, halves L1 footprint | M | low–med | refreeze golden |
| | **Status (task #23, measured, yyzo-bh-07 p100a): REJECTED.** At equal (integer) packing the 16 B record is +2.37 ms/view slower than 32 B (157.23 vs 154.87) and 9.3 dB worse (hero_vs_ref 59.40, a thin-splat streak from the half-precision conic). Record bytes and L1 stores are not what the emit pays for; the half conversions cost more than the 4 saved stores. Only benefit: half the materialize bucket L1 (512 → 256 KB), useful only with R7, and then with a Cholesky-form conic from pfwc. See `docs/r16-record-2026-09-30.md`. | | | | |
| **R5** | p150 re-baseline + re-test stale refutations | −15 to −25 ms from 110→130 cores | S (resource-gated) | none | bit-identical |
| **R6** | Hierarchical pre-project frustum/size cull (6.13 M → ~1.4 M before project) | −10 to −25 ms | M | low | bit-identical if conservative |
| **R7** | Tile-owner L1-resident pair pipeline (delete the DRAM record round-trip and the materialize stage) | −20 to −40 ms | L | med–high | bit-identical achievable |
| **R8** | Fixed-capacity bucket + NoC atomic fetch-add offsets (delete the global prefix-sum barrier) | 0 alone; enables R7 + cross-stage pipelining | M | med | bit-identical with total-order key |
| **R9** | Move per-pair scalar float/UNORM work to the idle SFPU | −5 to −12 ms | M–L | med | bit-identical-ish (fp32 SFPU) |
| **R10** | Overlap the 3.24 ms image unpack + 0.55 ms D2H with the next view | −3.8 ms (throughput only) | S | low | bit-identical |
| **R16** | Kill the mid-sort host serialisation: move `host_bin_layout_from_hist` on-device (or overlap it with the previous view) and shrink the ~9–10 ms `publish` H2D | −5 to −12 ms | M | low–med | bit-identical |
| | **Status (task #18, measured):** the host part was only ~2.4 ms and `publish` was device time (radix + a single-core directory kernel). Landed −6.5 ms/view on sort (layout rewrite, host-uploaded directory, emit reads the count histogram, batched count reads); host bridge now ~0.65 ms, at its floor. See `docs/sort-stage-split-2026-09-30.md`. | | | | |
| **R11** | Adaptive radix bucket count in `sort_tile_depth` | −2 to −3 ms | S–M | low | bit-identical |
| | **Status (task #26, measured, iter-159):** done. Key-range digits (1–4 passes), one fused histogram scan, ids-only last pass, u16 histograms in RISC local memory (L1 histograms only saved 0.4 ms: bucket read-after-write), 4-wide load-ahead. `publish_wait` 8.02 → 5.02 ms, frame 96.95 → 94.10 ms/view, Tracy `sort_tile_depth` 6.97 → 3.98 ms, byte-identical (yyzo-bh-07 p100a). | | | | |
| **R12** | `Q = Φθᵀ` as `matmul_tiles` on the FPU + tilized θ operand | ~0 now; −10 to −20 ms after the pole flips to SFPU | L | med | refreeze golden |
| **R13** | Fitted polynomial replacing `_sfpu_exp_21f_bf16_` | ~0 alone; prerequisite for R12 | S | low | refreeze golden |
| **R14** | Whole-tile skip for negligible-contribution tiles | unknown; gated on E5 | M | low | lossy |
| **R15** | Single-destination scatter + neighbourhood gather (tt-splat §5.2) | ≤−13 ms minus a new gather cost | M | med | bit-identical possible |
| — | Weighted-Sum Rendering / polynomial splat as a *model* | **not a 2026-H2 lever** — needs retraining, loses occlusion (−9 to −18 dB `[P]`) | — | — | — |

### One-line rationales

- **R1** — The project declared four floors without ever measuring a ceiling.
  One day of microbenchmarks converts every future "floor" claim into a ratio.
- **R2** — 119 ms of every frame runs on one of two data movers while the other
  idles; this is pure arithmetic from `CreateKernel` call sites and the zone
  table, and 141 prior iterations never tried it.
- **R3** — 1 904 cycles/pair for ~45 instructions; the three named stall sources
  (volatile stores, 1-deep NoC read, per-batch write barrier) are each a known,
  already-proven fix pattern in this codebase.
- **R4** — Halves the per-pair store count that R3 is attacking *and* doubles
  the L1 headroom R7 needs; worthless for bandwidth (DRAM is at 1 %).
- **R5** — Charter-mandatory, and 130 vs 110 usable cores is ~18 % more
  parallelism on every per-core-partitioned stage; also re-tests two refutations
  that were explicitly scoped to "this board, this program count".
- **R6** — 6.13 M gaussians are projected every view to keep ~1.4 M (23 %); the
  30-view bench is a smooth orbit, so a chunk-level AABB frustum test (or a
  view-coherent visibility cache) removes ~77 % of a 36 ms BRISC stage and
  ~319 MB of attribute reads.
- **R7** — The whole view's pair payload fits in aggregate L1 at 47 % (23 % at
  16 B); tile ownership deletes the DRAM record round-trip, the `materialize`
  stage (19.6 ms) and the overflow gather entirely.
- **R8** — Removes the barrier that made `sort_bucket_emit` "0 ms pipelinable"
  and thereby killed the iter-141 lever; zero gain alone, so schedule it *with*
  R7, never as a standalone iteration.
- **R9** — The work that is strangling NCRISC is 32-wide elementwise float math,
  and three SFPU-capable RISCs per core have no kernel at all during sort.
- **R10** — Cheapest remaining host lever; note it improves the 30-view average
  and not single-frame latency, so label it as such in the ledger.
- **R16** — The sort stage contains a *blocking* mid-pipeline histogram D2H plus
  a host prefix/LPT build, and then pays ~9–10 ms re-uploading the layout; in a
  pipeline whose stated design goal is host-free, that round trip is pure
  serialisation. Size it first with the E2 follow-up (split `bin_ms` into its
  device-launch and host-compute halves) — the win is only as big as the host
  half plus the publish upload.
- **R11** — Previously deferred at "~2-3 ms, risk > reward"; after R2/R3 shrink
  the pole, 2-3 ms is a larger fraction and the risk is unchanged.
- **R12** — The endgame: the only identified route to a per-view cost of
  `O(pixels × K)` independent of gaussian count, and the only user of the idle
  matrix engine. Deliberately last: it shrinks the SFPU shadow that R7/R8 want.
- **R13** — Small, self-contained, and the gate on whether an all-FPU blend is
  numerically viable; do it as R12's first step, not as a perf iteration.
- **R14** — Plausible (MIMD cores can exploit irregular sparsity where GPUs
  cannot `[P]`) but completely unquantified; costs one host histogram to size.
- **R15** — Demoted: mean fan-out is 1.71, not 4, so the pair-count prize is
  ~42 % of pairs at best against a new gather cost.

### Sequencing

```
R1 ──> R3 ──> R2 ──┬─> R4 ──> R8 ──> R7 ──> R9 ──> R13 ──> R12
                   └─> R6, R10, R11, R16 (independent, any time)
R5 (p150) as soon as a board frees; re-anchor all numbers, then continue.
```

R1 before R3 because the microbenchmark says *which* stall to attack. R3 before
R2 because it is cheaper, bit-identical, and needs no host-side layout change —
and its measured speedup calibrates R2's expectation. R4 before R7 because R7's
L1 budget wants the 16 B record. R8 only ever lands together with R7.

---

## 5. The first three tasks, specified for an implementer

All three land on branch `smarton/tt-project-opt` in `~/dev/gstt2`. No PRs.
All device runs go through `~/dev/tt-workflows/scripts/devrun.sh` with
`--timeout 580`. Baseline to beat: **173.30 ms/view, stdev 0.26 ms**, so the
noise band is ±0.5 %; a change must move `avg_frame_ms` by >0.9 ms to count.
Quality gate: `hero_vs_ref ≥ 50 dB` against
`tests/fixtures/hero/hero_golden_8bit.png` (md5 `e3fefb116d860f99d92bba1ef51d820c`);
bit-identical = 100.00. Secondary anchor `hero_vs_cpu ≈ 63.95 dB`.
Board: prefer a free p150; `yyzo-bh-07` (p100a) is the comparable fallback —
**label every number with the board it came from.**

---

### Task 1 — `T-A`: hardware-ceiling microbenchmark suite  *(tooling, blocks R2/R3)*

**Goal.** Produce, for this board, the measured per-operation costs the roofline
model currently assumes. Answer six questions with numbers.

**Deliverable.** `render/bench/` containing one host driver
(`risc_microbench.cpp`, built as a separate CMake target — do **not** touch
`render_clean`) and one dataflow kernel per probe, plus
`docs/hw-ceilings.md` with the result table and the raw `[MB]` log lines.

**Probes.** Each probe runs N = 2²⁰ iterations on one core, brackets the loop in
a single `DeviceZoneScopedN`, and reports `cycles/op`. Run each on 1 core and on
all 110 cores to expose shared-resource contention.

| Probe | Question |
|---|---|
| `p1_l1_store_volatile` | cost of one `volatile uint32_t` store to local L1 |
| `p2_l1_store_block` | cost of a 32 B block store built in registers (non-volatile struct + one `__builtin_memcpy`) — **the ratio p1×8 : p2 is R3's expected gain** |
| `p3_l1_load` | cost of one `volatile uint32_t` load from local L1 |
| `p4_noc_write_issue` | cost of `noc_async_write` issue, amortized over a batch of 16, excluding the barrier |
| `p5_noc_read_rt` | exposed latency of `noc_async_read` + immediate `noc_async_read_barrier()` (1 outstanding) vs 8 outstanding + 1 barrier — **quantifies R3's prefetch gain and reproduces the iter-137 pattern** |
| `p6_dual_mover` | the same store-and-scatter loop run on NCRISC only vs split across BRISC + NCRISC; report the **scaling factor** — **this is R2's expected speedup, measured** |
| `p7_dram_bw` | achieved GB/s for large sequential `noc_async_read`/`write`, to validate the 512 GB/s `[P]` figure |

**Also record**, from the same run: `AICLK` (so cycles→ns is measured, not
assumed), `compute_with_storage_grid_size()`, `tt-smi -s` board type and
`board_id`. This replaces the `[P]` clock and core-count rows in §2 with `[M]`.

**Acceptance.** `docs/hw-ceilings.md` exists with all seven probe results at
1 core and 110 cores, plus the measured clock and grid. For each of the three
stall suspects in §2.3, state the measured cycles/op and the implied share of
the 1 904 cycles/pair. Explicitly answer: *is `sort_bucket_emit` within 2× of
any measured hardware ceiling?* A "no" is the expected and acceptable answer.

**Risks.** None to the production renderer — separate target, separate binary.
Do not link it into `render_clean`. Keep the whole suite under one 580 s devrun.

---

### Task 2 — `T-B`: de-stall the `sort_bucket_emit` inner loop  *(R3, bit-identical)*

**Scope.** `render/kernels/dataflow/sort_bin.cpp` only. No host changes, no
layout changes, no format changes. Output must be **byte-identical**
(`hero_vs_ref = 100.00`, hero PNG md5 `e3fefb11…`).

**Three independent changes; land and measure them one at a time** so each gets
its own ledger row and its own attribution:

1. **Register-built record, one block store.** Replace the eight `volatile
   uint32_t` stores in `pack_rec` (`sort_bin.cpp:451-463`) with a non-volatile
   local `uint32_t rec[8]` filled in registers, then one
   `__builtin_memcpy((void*)(l1_scratch + b*32), rec, 32)`. Bit-identical by
   construction (same bytes, same order). Expected gain = `8×p1 − p2` from T-A.
2. **Multi-buffered `blendrec` prefetch.** Today `blendrec[g]` is a single
   outstanding `noc_async_read` followed immediately by
   `noc_async_read_barrier()` (`sort_bin.cpp:498-501`) — one exposed round trip
   per gaussian. Pairs are gaussian-major, so the next 8 distinct `g` values are
   known from the already-resident `gid` page. Issue 8 reads into an 8-slot L1
   ring, one barrier per 8, consume in order. This is exactly the iter-137
   `ta_gauss_aabb` pattern (8 pages before one barrier). Bit-identical: same
   bytes, only the issue schedule changes. Expected gain = `p5(1-deep) −
   p5(8-deep)` per gaussian ÷ 1.71.
3. **Batch the counting-sort stores.** `ksp[li] = key; isp[li] = g;`
   (`sort_bin.cpp:557-558`) are two scattered `volatile` L1 stores per pair.
   Consecutive pairs for the same tile land in consecutive `li`, so accumulate
   short same-tile runs in registers and flush them as block stores. Gaussian-
   major ordering means runs are short (K≈1.71) — **measure before keeping**;
   revert if flat.

**Measurement.** For each change: `devrun.sh` → `python3 render/run.py`, 4×
30-view repeats with `--no-ref`, then one run with the CPU reference for the PSNR
gate. Report `avg_frame_ms` mean and min, and require clean cluster separation
(candidate max < baseline min) as the project's existing standard. Then one
30-view Tracy capture (`bash opt/profiler/capture_tracy.sh <iter-dir>`) analysed
with `analyze_zones.py` **per-view busiest-core makespan** — never aggregate
column sums — reporting `sort_bucket_emit`, `NCRISC-KERNEL` and `BRISC-FW`.

**Acceptance.** Each change is a separate ledger row in `opt/ttw/iters.jsonl`
with `hero_vs_ref = 100.00`, its own Tracy deltas, and a keep/revert verdict by
the existing gate (bit-identical but frame-neutral ⇒ revert). Regenerate
`opt/REPORT.html` via `python3 opt/build_report.py` and clear
`opt/current-iter.json`. Target: `sort_bucket_emit` 30.6 → ≤24 ms and
`avg_frame_ms` 173.3 → ≤168.

**Risks.** Low. The `volatile` qualifier may be load-bearing for a reason not in
the comments — if change 1 produces wrong output, the block store is crossing a
CB or NoC visibility boundary; add `invalidate_l1_cache()` before the dependent
read and re-test before abandoning. Watch L1 pressure from the 8-slot blendrec
ring (8 × 64 B = 512 B — negligible).

---

### Task 3 — `T-C`: dual-data-mover split of `sort_bucket_emit`  *(R2 pilot, bit-identical)*

**Why this first, of all the R2 candidates.** `sort_bucket_emit` is the largest
single-mover zone (30.6 ms), its input partitioning is already a contiguous pair-
page range (`pg_lo`..`pg_hi`), and its output slot allocation is already a
per-`(core, tile)` disjoint base table — so widening it to per-`(core, mover,
tile)` is a mechanical change to an existing host structure rather than a new
data structure.

**Design.**

1. **Host** (`render/host/sort_device.cpp`, `host_bin_layout_from_hist` and the
   `[OVERFLOW-DIST]` block around line 1148): the per-`(core, tile)` histogram
   and prefix become per-`(core, mover, tile)` with **mover 0 (BRISC) ordered
   before mover 1 (NCRISC)** within each core. Splitting the pair-page range in
   two and concatenating the two movers' outputs in that order reproduces
   exactly today's gaussian-major sequence within each tile ⇒ the stable per-tile
   radix produces the identical permutation ⇒ **byte-identical output**. This
   ordering invariant is the correctness argument; assert it in a comment and in
   a host-side check.
2. **Kernel**: `sort_bin.cpp` is unchanged except that `pg_lo`/`pg_hi`,
   `l1basep`, `ov_basep` and the `l1_scratch`/`CB_REC`/`CB_PACKOC` pointers come
   from per-mover runtime args. The two movers must use **disjoint L1 staging
   regions** (separate CB pages) and their **own** `curp` cursor array.
3. **Host program**: create the emit kernel twice — once with
   `{.processor = RISCV_0, .noc = NOC::RISCV_0_default}`, once with
   `{.processor = RISCV_1, .noc = NOC::RISCV_1_default}` — on the same
   `CoreRangeSet`. The two movers drive separate NoCs, so there is no NoC
   contention; L1 port contention is the only shared resource, and T-A probe
   `p6_dual_mover` has already measured its effect.
4. **Load split**: start with a 50/50 pair-page split. If `p6` showed asymmetric
   scaling, use the ratio it implies.

**Measurement.** Same protocol as Task 2. Additionally report **BRISC-KERNEL**
makespan (it must rise by roughly what NCRISC-KERNEL loses — if it rises by
*more*, the split is a cost-shuffle and must be reverted) and the
`p6_dual_mover` scaling factor from T-A as the predicted-vs-actual check.

**Acceptance.** `hero_vs_ref = 100.00` (byte-identical). `sort_bucket_emit`
makespan ≤ 60 % of its pre-task value, `NCRISC-KERNEL` down by ≥8 ms,
`avg_frame_ms` down by ≥5 ms with clean cluster separation. Ledger row + Tracy
+ regenerated `opt/REPORT.html`. If it lands, immediately open follow-ups to
apply the same pattern to `ta_bucket_scatter` (12.2), `ta_gauss_aabb` (9.7),
`sort_subchunk_materialize` (19.6) and — in the opposite direction, BRISC→NCRISC
— `proj_scatter` (20.8) and `proj_count` (15.0).

**Risks.**
- *Medium: ordering.* Any deviation from "mover 0's records precede mover 1's,
  per tile" breaks bit-identity. The fix if it does break is to make the sort key
  a total order `(depth_key, gaussian_id)` — which is also R8's prerequisite, so
  the work is not wasted.
- *Medium: L1 pressure.* Two staging regions, two `curp` arrays (1024 × 4 B
  each), two `CB_REC`/`CB_PACKOC`. Budget ~10 KB extra; check against the
  70 656 B per-core kernel-config ceiling documented in `docs/CONTEXT.md` §3.
- *Low: hangs.* Two data movers in one program is the ordinary tt-metal
  reader/writer pattern (already used by blend), not the 2-command-queue pattern
  that hung iters 52–59. Keep the single in-order CQ and the single drain at
  blend readback.
- If a run hangs, do **not** SIGKILL — that wedges the ARC firmware and requires
  `recover.sh --unwedge`.

---

## 6. Prior conclusions this plan reopens, and on what grounds

| Frozen conclusion | Status | Grounds |
|---|---|---|
| "173 ms is the architectural floor for this pipeline design" | **reopened** | Rests on NCRISC "100 % occupancy"; occupancy measures parked-on-stall, not throughput. NCRISC runs at ~7 % of its issue roofline. |
| "`sort_bucket_emit` is at its per-pair 32 B L1 store floor" | **reopened** | 1 904 cycles/pair for ~45 instructions. The supporting ablation (null the stores) cannot separate store cost from dead-code-eliminated dependent work. |
| "L1 (1.5 MB) blocks the L1-resident pipeline" | **narrowed** | True for a *fused double-buffered emit+cull+blend* kernel. False for tile ownership: 700 KB/core at 32 B, 350 KB at 16 B, heaviest tile 848 KB. |
| "Program fusion is not a makespan lever" (iter-133) | **upheld, and superseded** | Reclaimable launch firmware ≈0.11 ms. But the diagnosis ("BRISC parked in barriers waiting on NCRISC") is precisely finding #1 — the fix is to give BRISC work inside those programs, not to fuse them. |
| "Metal Trace removes <0.5 ms" (iter-126) | **upheld; re-test after R2/R7** | Explicitly scoped to this board and this program count; R7 changes the program count. |
| "The host is out of the loop (4.86 ms)" (iter-136) | **upheld** | R10 is the only host lever left and it is throughput-only. |
| "Blend transmittance early-out has no headroom" (iter-139) | **upheld** | Perfect-early-out ceiling ~2.3 ms. |
| tt-splat single-destination scatter is a ~20 ms lever | **demoted** | Mean fan-out is 1.71, not 4 — the "median K=4" figure is microblocks per tile. |
| WSR / polynomial splat as a model | **out of scope** | Needs retraining; −9 to −18 dB `[P]` on solid geometry; bicycle has hard occluders. |

---

## 7. Reporting and cadence

- Every measured change gets one row in `opt/ttw/iters.jsonl` (idea, verdict,
  metrics, Tracy path, build ID) and `opt/REPORT.html` is regenerated with
  `python3 opt/build_report.py`. Never hand-edit the HTML. Clear
  `opt/current-iter.json` when an iteration finishes.
- The build-delta gate stays: a kept code iteration must not share a binary
  fingerprint with its predecessor (kernel `.cpp` files are JIT-compiled and
  leave the `.so` byte-identical).
- Every number carries its board. The ledger currently mixes p100a numbers under
  a charter that mandates p150; add a board column and re-anchor on p150 at R5.
- Stop rule, per the charter: no lever below ~1 % without a compounding
  rationale. R1's whole purpose is to keep that judgement grounded in a measured
  ceiling rather than a plateau.
