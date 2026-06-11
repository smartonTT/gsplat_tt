# Sort: L1-resident, DRAM-free plan

Status: PLANNED (not yet executed). Owner: supervisor.
Companion to `opt/blend-cull-speedup-plan.md`.

---

## ════════════ REBASELINE + ROADMAP (iter 132, 2026-06-11) ════════════

### NEW REF
- **Golden** = iter-132 hero, md5 `e3fefb116d860f99d92bba1ef51d820c`, committed at
  `tests/fixtures/hero/hero_golden_8bit.png`. Gate metric `hero_vs_ref` is now
  **8-bit PSNR vs this golden** (bit-identical = 100.00 dB; gate >= 50). Legacy
  float-vs-CPU PSNR retained as the secondary `hero_vs_cpu` diagnostic (~63.95 dB).
  Ledger has a `NEW REF` divider row. Refreeze + add a new divider whenever an
  accepted iteration legitimately changes output.

### Where the frame stands (measured, iter 132)
Frame **177.1 avg / 142.3 min ms/view**. Critical path = **BRISC-FW ~161.7 ms**
(data mover + per-program launch firmware). NCRISC-KERNEL ~127 ms (dataflow, some
slack). TRISC (all SFPU) ~52 ms — **off the critical path**. Iters 128-132 ground
NCRISC zones down (`sort_bucket_emit`, `sort_subchunk_mat`, op/color pre-pack) but
the wins keep landing on a RISC that has slack, so **BRISC-FW barely moves**. We are
out of cheap NCRISC micro-opts that move the frame.

### Prioritized roadmap — what I want to do next (in order)
1. **PROGRAM FUSION (top lever).** BRISC-FW carries ~90 ms of per-program launch
   firmware across ~dozen device programs. Cheap NCRISC reshuffles can't touch it;
   only **fewer, fused programs** can. Fuse adjacent same-grid stages
   (project→gather→tile_assign, then sort→cull→blend) into persistent kernels with
   on-device phase barriers instead of host-relaunched programs. This is the only
   evidenced path to a step-change, and it's the same direction the host-free plan
   already wants. Measure launch-count vs BRISC-FW first to size the prize.
2. **NCRISC dataflow + DRAM-free handoff.** Keep collapsing the sort→cull→blend
   data path so it stays L1-resident (no DRAM scatter/gather round-trip): subchunk
   directory + materialize folded into the sort emit, payload read once. Banks the
   NCRISC slack into real frame time *once fusion frees the BRISC long pole*.
3. **Blend/cull reader cluster (`~78 ms`, SFPU-backpressure).** Needs the blend
   phasing rework (Track 1 failed once — high risk). Only worth it after (1)/(2)
   shift the bottleneck onto SFPU. Parked until then.

### "1 ms north star — where would you have to leave to?" (user question)
The current design **cannot reach ~1 ms** without leaving these constraints:
- **Per-frame host dispatch must go to zero.** ~90 ms today is on-device firmware
  launching ~dozen programs every frame. 1 ms needs a **single persistent
  super-kernel** (or Metal Trace replay *if* the launch cost were host-side — it
  is NOT here, measured iter 126, so persistent kernels are the route), i.e. the
  full host-free-l1 design end state, not incremental fusion.
- **DRAM must leave the per-frame loop entirely.** All inter-stage handoff
  L1-resident; DRAM touched only for the initial gaussian load + one final D2H.
- **SFPU must become the bottleneck, then be saturated.** At 1 ms the ~52 ms of
  SFPU compute itself is 50× too slow — needs microblock-major blend + transmittance
  early-out (skip saturated tiles) and full SFPU occupancy, i.e. a different blend
  algorithm, not the current gaussian-major no-early-out one.
- Bluntly: ~1 ms is a **different renderer** (persistent L1 super-kernel + early-out
  blend), not this host-driven multi-program pipeline tuned harder. Treat 1 ms as
  the destination of the host-free-l1 rewrite, and **~40-60 ms as the realistic
  near-term floor** for the current architecture once fusion lands.

### Parked levers — user requests kept for LATER evaluation (NOT abandoned)
- `emit-record-layout`: cut `sort_bucket_emit` L1 store volume via record-layout
  change / once-per-gaussian fill. Not bit-identical → needs golden refreeze. Higher
  risk; revisit after fusion.
- `blend-reader-cluster`: blend/cull reader cluster rework (see roadmap #3).
- **Publish-only-for-overflow** op/color words (the iter-132 next-lever): skip the
  per-gaussian publish for in-budget tiles so NCRISC doesn't re-absorb the BRISC win.
- Metal-Trace / host-free-trace family: REFUTED as a perf lever for THIS workload
  (see refuted ledger) — keep only as documentation of why persistent kernels (not
  trace) are the host-dispatch fix.

### What WON'T be done anytime soon (so you can verify I'm not dropping levers)
- **Metal Trace capture/replay** (Stage 5.4-5.6): measured <0.5 ms payoff here
  (firmware is on-device, not host dispatch). Shelved unless re-tested proves
  otherwise.
- **Blend SFPU micro-optimization as a frame lever:** TRISC is off the critical
  path; won't move the frame until fusion/early-out flips the bottleneck. (Track 1b
  ILP already kept but frame-masked.)
- **Any record-layout / output-changing change** until after fusion lands (would
  force a golden refreeze mid-flight and muddy the rebaseline).

### Refuted-premises ledger (RE-TEST periodically — premises go stale)
Per user: do NOT treat these as settled forever. Each is tagged with the conditions
under which it was refuted; re-test when HW/thermal/code state changes materially.
- **"Metal Trace removes the ~92 ms host dispatch"** — REFUTED iter 126 (the 92 ms
  is on-device firmware + NCRISC dataflow, not host-side `Finish`). Conditions: this
  BH board, cpp#~126, this program count. RE-TEST if program count drops a lot.
- **"`sort_bucket_emit` cost is scattered DRAM writes"** — REFUTED iter 128 (real
  cost was redundant per-pair packing of gaussian-invariant fields; only ~1.6 ms was
  scatter). RE-TEST after record-layout change.
- **"`be_main` cost is per-pair div/mod arithmetic"** — REFUTED iter 129 (residual
  is per-pair 32B L1 store traffic, ~17.5 ms structural floor). RE-TEST after layout
  change.
- **"`sort_subchunk_mat` cost is the L1 permute"** — REFUTED iter 130 (real cost was
  overflow gather re-read/re-pack + load imbalance; LPT-balanced it). RE-TEST if
  overflow rate changes.
- **"Stage-2b: `proj_scatter` cost is `blendrec` writes"** — REFUTED iter 119
  (`proj_scatter` is required SoA writes; `blendrec` only ~2 ms). RE-TEST after SoA
  layout change.
- **"cull+blend SFPU ~175 ms is the killer"** — REFUTED iter 116 (aggregate-sum
  trap; TRISC makespan is ~52 ms, off-path). RE-TEST if blend algorithm changes.
- **"K ≈ 1 (one microblock per tile)"** — REFUTED (fan-out histogram showed
  meaningful K>1). RE-TEST per scene/view set.

### Fusion scoping (MEASURED iter-132, `opt/profiler/ttw-132/profile_log_device.csv`)
Per-frame the chain launches **~17 programs (14 full-grid 110-core + 3 single-core
coordinators)**; the busiest BRISC core sees **15.50 program dispatches/view**.
Arithmetic on the long pole:
```
BRISC-FW/view      = 161.71 ms (frame long pole)
BRISC-KERNEL/view  =  85.39 ms (real data-mover work)
BRISC non-kernel   =  76.32 ms (per-program launch/dispatch/barrier firmware)
per-program launch ≈ 76.32 / 15.50 ≈ 4.8-4.9 ms / program
```
So **each pair of adjacent same-grid programs fused ≈ −4.8 ms/view** off the long
pole. Collapsing the whole chain into one persistent super-kernel reclaims most of
the 76 ms — that is the north-star end state, not the first step.

Ranked fusion candidates (same-grid, no cross-core barrier preferred, bit-identical):
1. **project(means_cam)+pfwc** — identical 110 grid, identical `split_chunks(ceil(N/1024))`
   per-core split, pure per-gaussian element-wise, NO cross-core dep / no semaphore.
   `means_cam` is an fp32 DRAM round-trip today → becomes a core-local L1 CB pass.
   **CHOSEN FIRST.** Expected ~4.5-5 ms launch + small dataflow saving (~4-6 ms/view),
   bit-identical (fp32 lossless, same HiFi4). → iter 133.
2. radix+publish (110, per-core LPT tiles, no cross-core) — low risk, publish tiny.
3. cull+radix (110, same per-core tiles) — low-med risk, 1 launch saved.
4. pfwc+gather (110) — med: gather has an internal count→core0-prefix→scatter sync.
5. emit+cull+radix (110) — med-high: emit scatters gaussian→tile ACROSS cores →
   needs a global semaphore phase barrier before radix.
- BLOCKED: bin_hist+bin_emit (host prefix-sum+LPT sits between; on-device layout was
  a regression iter-127); anything+single-core coordinators (grid mismatch 110 vs 1).

---

## Correction up front (read this first)

My earlier "sort ≈ 70 ms, with `bucket_emit` 34 ms + `materialize` 25 ms" was
**wrong** — those came from device-zone CSV columns that are **aggregate sums
across cores × subchunks × views**, not wall-clock makespan. (Same trap the
`iter 111` "tile_blend_sfpu ~291k ms" label falls into.)

The measured numbers (lessons.md, dated 2026-06):
- **Sort STAGE makespan ≈ 26 ms/view** (`99.4 → 26.2 ms` after the duplicate
  `BUCKET_CULL` pass was dropped).
- Of that, the literal radix is **~7.5 ms + publish ~1.3 ms ≈ 9 ms**; the other
  ~17 ms is histogram + the gaussian→tile bucket scatter + subchunk directory +
  materialize.
- Frame total ≈ **192 ms/view** (`iter 111`).

### Stage 0 MEASURED — authoritative makespan breakdown (iter 116, 195.5 ms/view, `analyze_zones.py`, 30 views)
The earlier "cull+blend SFPU ~175 ms is the killer" claim was **WRONG — the same
aggregate-sum trap.** Per-view makespan (busiest-core proxy) from
`opt/profiler/ttw-116/profile_log_device.csv`:

| RISC role | per-view makespan | reading |
|-----------|-------------------|---------|
| **BRISC-FW** (data mover, long pole) | **179.7 ms** | ≈ frame critical path |
| BRISC-KERNEL | 87.1 ms | → **~92 ms/view is BRISC non-kernel = host dispatch/launch/barrier serialization** |
| **NCRISC-KERNEL** (data mover) | **144.5 ms** | genuinely busy doing dataflow |
| **TRISC-KERNEL** (ALL SFPU compute) | **52.4 ms** | **NOT the bottleneck** |

Dominant named zones (per-view makespan): `sort_bucket_emit` 39.7 (NCRISC),
`tile_blend_load` 28.5, `rd_l1_bulk` 28.3, `sort_subchunk_mat` 26.1,
`tile_l1_cull_rd` 22.0 (NCRISC reads); `proj_scatter` 20.8, `proj_count` 15.0
(BRISC); `tile_blend_sfpu` 29.3, `tile_mb_mask` 22.4 (TRISC, off-path).

**Conclusion: the frame is DATAFLOW + HOST-DISPATCH bound, not compute-bound.**
Blend/cull SFPU micro-opt cannot move the frame (TRISC is the short pole). The
levers, in value order: (1) the ~92 ms BRISC host-dispatch overhead → host-free
trace (Stage 5); (2) NCRISC dataflow `sort_bucket_emit`+`sort_subchunk_mat`+readers
and BRISC `proj_scatter`+`proj_count` → Stages 2b/3 (fuse project, L1 handoff,
collapse materialize). This VALIDATES the sort plan's strategic direction and makes
it the primary lever, not a prerequisite-only side quest.

So: the prize for fixing sort is **not the 26 ms** — it is enabling a **DRAM-free
sort → cull → blend handoff** AND collapsing the BRISC `proj_scatter`/host-dispatch
that actually dominate the frame.

## Why it's DRAM shuffling on the data movers (root cause)

The active path (`GSPLAT_TT_RESIDENT_PAIRS`, `sort_bin.cpp` +
`sort_radix_tile.cpp` + `sort_subchunk_{directory,materialize}.cpp`) does:
1. `sort_bin_hist` — per-tile histogram of kept pairs → DRAM `bin2d`.
2. host prefix sums → page-aligned per-tile starts + per-core base.
3. `sort_bucket_emit` — **DRAM scatter**: read pairs, random-gather `blendrec[g]`
   from DRAM, write 32B records into per-tile DRAM buckets (`buf_l1_recs` — the
   "l1" is the *record format*, the buffer is DRAM-interleaved) + write (key,id).
4. `sort_radix_tile` — per-tile LSD radix of (key,id) in DRAM (read block → L1 →
   sort → write back).
5. `sort_subchunk_directory` — split heavy tiles (`max_tile_n ≈ 26.5k >
   BUCKET_FIT 8192`) into ≤4 subchunks.
6. `sort_subchunk_materialize` — **DRAM scatter #2**: apply the radix permutation,
   write depth-sorted PACK2 slabs to DRAM (the slabs the cull/blend readers load).

Two things to be precise about:
- **Data movers are the *correct* engine.** A scatter is NoC traffic; that is
  exactly what NCRISC/BRISC are for. The sin is **how many times** the payload is
  moved through DRAM and the **random `blendrec` re-read** in `bucket_emit`, not
  the RISC choice and **not** the use of DRAM as the rendezvous. (DRAM as a
  contention-free, per-producer mailbox is the right call — see target design.
  NO core-to-core / NoC L1↔L1 is needed or wanted for sort.)
- **There are two full passes over the record set** (scatter to bucket, then
  scatter again to apply the sort permutation). The second (`materialize`) is the
  classic "sort indices, then permute payload" — but the permute is a DRAM scatter
  instead of an in-L1 move.

## Capacity reality (this is why DRAM was chosen, and the real constraint)

From `GSPLAT_TT_LPT_STATS` (hero): 1024 non-empty tiles / 110 cores, mean **39,151
cand/core**, max/core 39,664 (makespan/mean **1.013** — LPT is already near-
optimal, `build_lpt`), heaviest single tile **26,544 = 0.678×** the per-core mean.

- **A core processes its ~9 tiles ONE AT A TIME**, so only one tile's bucket is
  live in L1 at once — not the full 1.25 MB per-core set. The binding constraint
  is therefore the **single heaviest tile**, not the per-core total.
- Heaviest tile ≈ 26.5k × 32B ≈ **849 KB** < 1.5 MB L1. **One bucket fits.** ✓
  - Sorting **(key, index) = 8B** pairs ping-pongs in 2 × 26.5k × 8B ≈ **424 KB**,
    leaving the 849 KB records in place → ~1.27 MB total: **fits even the max
    tile.** This is why "sort indices, not records" is load-bearing, not just a
    perf nicety.
  - Sorting the **full 32B records** with an LSD ping-pong needs 2 × 849 KB ≈
    **1.7 MB > L1** for the heaviest tile → that path (only) forces splitting the
    few heaviest tiles into subchunks, or an in-place permute.
- Aggregate sanity: 3.37M × 32B = **108 MB** candidates < 120 × 1.5 MB = **180 MB**
  total L1 — fine, and irrelevant since each core only needs one bucket resident.

Prior art: `iter 42` already proved an **L1-resident LSD radix over 32B records is
bit-exact**; its scary "232 ms" was a lumped-pipe timer artifact, not a slow sort.
So the algorithm is demonstrated — the open work is the *data path* (where the
gaussian→tile transpose happens), not the sort itself.

## Target architecture: TA-bucketed, DRAM-rendezvous, per-core L1 sort

**No core-to-core. No NoC L1↔L1.** The gaussian→tile transpose is done once, by
**TA writing each pair's 32B payload into per-`(core, tile)` DRAM buckets**
(contention-free: each TA core writes only its own region, no shared pages, no
atomics). Then **each sort core reads ITS tile's bucket from DRAM into its own L1
and sorts locally** — cores never touch each other's L1. Tile-stationary
(`build_lpt`, mirrored in `compute_lpt_assignment`) so the same core blends out of
that L1.

Pipeline:
1. **Fused project+TA, record born in L1 (O1)**: each core computes `blendrec[g]`
   for its gaussians **in L1** (visible/core × 32B ≈ 160–290 KB, fits), counts tile
   footprints → per-`(core, tile)` histogram. The gaussian-major `blendrec` DRAM
   array is **never written**.
2. **Device prefix → offsets (O3)**, then the **scatter**: each core writes its 32B
   records into the DRAM tile-buckets at its dense offsets. Bucket layout is
   **`[tile][core][slot]` (O8)** so a tile's records are contiguous across cores
   (one coalesced 2048B-page read for the consumer) while each `(core,tile)` slice
   stays disjoint (contention-free). **This scatter is the ONLY DRAM write.**
3. **Per-core L1 sort + perm-SoA emit (O2)**: each core reads its tile's bucket in
   one coalesced run into L1, radix-sorts **(key, index)** in L1, then **emits
   directly into the blend SFPU's vector layout** — field-SoA in `cull_perm(g,m)`
   order so the SFPU loads each field as 32 contiguous lanes (no AoS→lane, no
   broadcast-gather). Cull stamps the coverage mask into the same resident layout.
   **This read is the ONLY DRAM read.**
4. **Tile-stationary L1 consume**: the same core culls/blends straight from that
   L1 layout (contiguous vector loads).

This deletes `materialize` entirely (no depth-ordered DRAM slab — blend consumes
from L1), removes the **random per-pair `blendrec` gather** (TA reads gaussian-
major), and lets blend/cull **drop their DRAM slab readers** (the ~29 ms blend
reader is ~98 % backpressure per the blend ablation, but it still pins a DRAM
dependency that blocks fusion).

What it does NOT make free: the transpose is irreducible — it MOVES into the
fused proj+TA scatter (~108 MB written once) + the sort's per-tile coalesced read
(~108 MB read once). So TA gets heavier, sort drops toward radix + one gather (not
~0). Net frame win is modest because sort (~26 ms) isn't the killer (cull+blend
SFPU ~175 ms is); the value is the clean L1 handoff that unblocks the blend plan.

## End-state: host-free single-trace frame (O3–O5)

Target: the host does **one tiny camera write + one `ReplayTrace` per view**,
nothing else. Today the render loop fires ~50+ `CreateProgram`/`EnqueueProgram`/
`Finish` calls per frame (sort 22, tile_assign 20, blend 9, …); Metal trace
(`BeginTraceCapture`/`ReplayTrace`) collapses that to ~zero. Requirements:

- **O3 — all inter-stage glue on-device.** Trace cannot host-compute sizes between
  stages, so device-ify: count→prefix→offsets, the subchunk directory, and
  **LPT**. LPT balance is **per-view** (depends on this view's tile sizes) so it
  MUST run on-device each frame, reading the resident histogram and writing a
  resident `tile→core` assignment buffer the kernels consume.
- **O4 — one full-frame trace.** Statically allocate every buffer to the per-frame
  safe ceiling (`padded_n` trick, lessons §host-free); upload the scene once;
  kernels read per-view extrinsics/K from a small **device** buffer (not runtime
  args). Per view: `EnqueueWriteBuffer(camera)` + `ReplayTrace`.
- **O5 — persistent per-core kernels** that loop over their assigned tiles (work
  list from the resident assignment buffer), with **per-tile double-buffering**
  (prefetch tile T+1's bucket while sorting/culling/blending T) to hide the one
  remaining DRAM read. The variable per-view iteration count is device-side data,
  so the SAME trace replays correctly for any view — this is the clean answer to
  the "microblocks/tiles need different iteration counts" worry.

Per-core frame shape:
`(fused proj+TA → scatter to [tile][core] DRAM bucket) → [device prefix+LPT] →
for t in my_tiles { coalesced read bucket[t] → sort (key,index) in L1 → emit
perm-SoA slab in L1 → cull stamps mask → blend } `, replayed per view.

**Tension to measure:** device-LPT is now mandatory (host LPT + single trace are
incompatible). If device-LPT is too costly, fall back to a **static,
view-independent** tile→core map (accept worse than LPT's 1.013 makespan/mean) to
preserve the single-trace property — a balance-vs-dispatch trade-off.

## Staged plan (lowest-risk first)

### Stage 0 — Measure the real per-sub-kernel split (do this first)
Capture device-zone makespans for `bin_hist / bucket_emit / radix / dir /
materialize` with the sort→blend pipe **OFF** (clean stage split), reported as
**ms/view** (divide aggregate zone sums by cores×subchunks×views, or use a single
representative core's zone). Confirm where the ~17 ms non-radix cost actually
lives before optimizing. (Avoids repeating the aggregate-sum mistake.)

### Stage 1 — Delete the `materialize` DRAM scatter (in-place L1 permute)
`materialize` already does "bulk-copy bucket + L1 radix permute" for **in-budget**
tiles. Make that the *only* path: sort the records where they land and emit the
slab from the in-L1 permuted order, with no second DRAM round-trip. Overflow
tiles (>BUCKET_FIT) keep the gather fallback for now.
- Win: removes one of the two full record-set scatters. Unconditional.
- Risk: low — it's a restructure of an existing code path. Keep PSNR ≥ 60 dB.

### Stage 2 — Move the bucket fill into TA (delete sort's gather + scatter)
Have **TA write each pair's 32B payload into its per-`(core, tile)` DRAM bucket**
instead of writing gaussian-major `(gid, tid)` and re-gathering later. Needs:
- a count/prefix pass for dense per-`(core, tile)` offsets (≈ today's `bin_hist`),
  OR over-reserved `bucket_fit` per `(core, tile)`;
- TA reads `blendrec[g]` **gaussian-major, cached once per gaussian** (consecutive
  pairs share `g`) — *not* the random per-pair gather `bucket_emit` does today.
- Bucket layout **`[tile][core][slot]` (O8)**: a tile's records contiguous across
  cores → one coalesced read for sort; each `(core,tile)` slice disjoint → no
  write race.
- Win: removes sort's `bucket_emit` scatter **and** the random `blendrec` re-read;
  sort becomes "read my tile (one coalesced run) → L1".
- Risk: medium. TA writes go 8B→32B (108 MB) and become tile-scattered within each
  core's region; the transpose cost moves here. Still race-free DRAM (per-core
  regions). NO core-to-core. Keep PSNR ≥ 60 dB.
- Note `"× num_cores"` = per-core **regions** (address partitioning), NOT data
  replication.

### Stage 2b — Fuse project into the bucket fill (O1): no gaussian-major DRAM
Once Stage 2 works, fold project into it: compute `blendrec[g]` in L1 and scatter
straight to the tile-buckets, so the gaussian-major `blendrec` DRAM array is never
written. **Result: the bucket scatter is the only DRAM write, the per-tile read
the only DRAM read.**
- Risk: medium; couples two stages. Keep the count sub-pass light (AABB/footprint
  from positions only; full `blendrec` computed once in the scatter sub-pass).

> **iter-119 DIAGNOSTIC — Stage 2b (kill `blendrec`) attempted & REVERTED (premise refuted).**
> Implemented the cheaper realization of Stage 2b: gather stops writing the
> gaussian-major `proj_m_blendrec` AoS; `sort_bin` (in-budget fill) and
> `sort_subchunk_materialize` (overflow re-gather) assemble each 32B PACK2 record
> from the compacted SoA `proj_m_*` instead. Output **bit-identical** (hero md5
> `e3fefb11…`). **Measured (30-view device makespan, vs iter-116):**
> - `proj_scatter` 20.83 → 18.79 ms/view (**−2.0 only**) — *`blendrec` is NOT the
>   20.8 ms cost; the bulk is the REQUIRED SoA writes (px,py,rx,ry,a,b,c,depth,op,
>   colors) that blend/cull/tile_assign still read.* The original premise is wrong.
> - `sort_bucket_emit` 39.68 → 36.02 (−3.7) — the in-budget fill *improved* (gaussian-
>   major SoA page caching, 9 pages / 16 gaussians).
> - `sort_subchunk_mat` 26.15 → **35.79 (+9.6)** (naive per-record-barrier form was
>   +18.2 → 44.33; batched 4-record form recovers ~8.5). `blendrec`'s real value was
>   making the **depth-sorted overflow re-gather** cheap (1 contiguous 64B AoS read/
>   record); without it the overflow gather needs ~8 SoA page reads/record (gids
>   depth-sorted → uncacheable) on the NCRISC long pole.
> - **NET ms_view 195.51 → 198.07 (+2.56, +1.3%)**, BRISC-FW 179.7 → 183.6, NCRISC-KERNEL
>   144.5 → 150.5. Clear structural regression on a refuted premise → reverted to
>   pristine HEAD (md5 re-verified, ms_view 195.24). Tracy: `opt/profiler/ttw-119/`.
> - **Re-scoped lever:** the only *winning* form of Stage 2b keeps ALL records (incl.
>   overflow) in the per-(core,tile) buckets so `materialize` reads them coalesced
>   from `buf_l1_recs` — NO depth-sorted SoA re-gather. (Grow/segment the bucket for
>   overflow subchunks instead of re-fetching by `gid`.) Until then, leave `blendrec`.
>   The real frame win is the **host-dispatch** long pole (BRISC-FW 179.7 ≈ 92 ms is
>   host overhead) → Stage 5 host-free glue, not `blendrec`.

> **iter-128 — `sort_bucket_emit` DIAGNOSED then OPTIMIZED (KEEP, bit-identical, frame win).**
> Built a 4-mode runtime-arg ablation (gated, since reverted) + nested phase zones
> (`be_cnt`/`be_main`/`be_kv`, since reverted) and measured the ~40 ms scatter on a
> 30-view Tracy capture (`analyze_zones.py`). **The "scattered DRAM writes" premise
> was REFUTED.** Full per-view-makespan attribution of `sort_bucket_emit` = 40.9 ms:
> - `be_cnt` (sub-pass-1 count) 1.41, `be_kv` (final page-aligned key/id writes)
>   0.39, prefix/prefill 0.33, **`be_main` (the per-pair scatter loop) 37.84**.
> - Within `be_main` (ablation, each = skip one component, subtract makespans):
>   - **`pack_rec` scalar packing = 27.0 ms (71 % of bucket_emit!)** — the per-pair
>     fp32→UNORM16 conversions (float multiplies) + volatile L1 re-loads of the
>     cached blendrec, on the NCRISC (a dataflow RISC; scalar float is slow).
>   - blendrec 64B read (once/gaussian, iter-114) = 5.84 ms.
>   - **32B record scatter writes = only 1.64 ms (4 %)** — coalescing them is NOT
>     the lever; NoC write traffic is cheap. (Refutes candidate-cause #1/#3.)
>   - counting-sort core (pair-stream reads + ksp/isp) = 3.37 ms.
> - Structural (30-view): P_kept 2.4M–3.7M, num_tiles 1024, num_cores 110,
>   ~21.7k recs/core; 78–113 overflow tiles drop ~330k recs (~14 %) past BUCKET_FIT.
>
> **Fix (bit-identical):** cov(0,1,2), depth_key(3) and the op/color UNORM16
> words(6,7) are functions of the **GAUSSIAN only** (identical for all K tiles a
> gaussian touches); only the tile-local mean (words 4,5) varies per pair. Hoisted
> the invariant packing — the 4 UNORM float-muls + the blendrec re-loads — to run
> **once per gaussian** (in the existing blendrec-read branch, `pack_invariants`),
> leaving only `mean − tile_origin` per pair. Bytes written to `buf_l1_recs` are
> byte-identical. (`render/kernels/dataflow/sort_bin.cpp` only; host untouched.)
> - **Measured (clean verify, 30 views): hero md5 `e3fefb11…` BIT-IDENTICAL, 63.95 dB;
>   ms_view 195.6→187.4 avg (−8.2), min 163.7→157.6 (−6.1).** Tracy (`ttw-128`):
>   **`sort_bucket_emit` 40.92→30.57 (−10.35, −25 %)**, **NCRISC-KERNEL 144.6→134.2
>   (−10.4)**, **BRISC-FW (frame critical path) ~179→169 (−10)**; all other zones
>   unchanged (`sort_subchunk_mat` 26.1, `tile_blend_*`, TRISC 51.9 off-path). The
>   sort scatter IS on the frame critical path (BRISC-FW tracks it), so the NCRISC
>   win propagated to the frame. **KEPT.**
> - **Residual `be_main` after hoist ≈ 27 ms** still has a per-pair tail: 2 fp32
>   mean subtracts + the `tt % / tt / l1_tiles_x` integer **div/mod** (no HW divide
>   on NCRISC) + 8 L1 stores, × ~21.7k pairs/core. **Next lever:** precompute a
>   per-tile origin table (1024×2) in L1 once and replace the per-pair div/mod with
>   a lookup; or move the whole pack into the (future) TA-side fill so the record is
>   born once per (gaussian) not per pair. Either is bit-identical.

> **iter-129 — per-pair div/mod attacked, MEASURED, REFUTED & REVERTED (decision=blocked).**
> Killed the per-pair tile-coord integer div/mod in `be_main`'s `pack_rec`
> (`tile_x = tt % grid_w`, `tile_y = tt / grid_w`). `grid_w = l1_tiles_x = 32` for
> the hero/bench config (1024x1024 image / 32 px tiles), a **power of two**, so the
> `%`/`/` collapse to `tt & 31` / `tt >> 5` — **provably bit-identical** integer
> results, **zero extra storage** (cheapest of the two task options; the per-tile
> origin LUT was unnecessary). Non-pow2 grids keep the original div/mod via a
> loop-invariant branch.
> - **Measured (clean verify + 30-view Tracy, ttw-129): BIT-IDENTICAL** (hero md5
>   `e3fefb116d860f99d92bba1ef51d820c`, **63.95 dB**) but **FRAME-NEUTRAL** —
>   `sort_bucket_emit` **30.57 -> 30.58** (+0.009, noise), NCRISC-KERNEL 134.23 ->
>   134.28, BRISC-FW (frame critical path) 168.96 -> 169.02, ms_view 187.4 -> 187.0
>   avg / 156.6 -> 157.1 min. **The NCRISC software integer divide is NOT the
>   `be_main` residual hotspot** — replacing it with the cheapest possible integer
>   ops (mask/shift) gained nothing, i.e. the divide is overlapped/hidden by the
>   dominant per-pair L1 store traffic (memory-bound, not ALU-bound). iter-128's
>   guess that the residual was "the integer div/mod" is **refuted**.
> - **Confirming ablation (null the 8 per-pair 32B-record L1 stores `p32[0..7]`;
>   NOT bit-identical, measurement-only):** `sort_bucket_emit` **30.58 -> 13.03
>   (-17.5 ms, -57 %)**, NCRISC-KERNEL -16.7, BRISC-FW -15.7. **The residual
>   `be_main` cost IS the per-pair 32B-record L1 store traffic** (8 x u32 stores x
>   ~21.7k pairs/core staged into `l1_scratch`), not the tile-coord decode and not
>   the float mean subtracts. (Consistent with iter-128: the 32B *DRAM scatter* was
>   only 1.6 ms — it's the *L1 staging stores*, not the NoC writes.)
> - **REVERTED to pristine HEAD** (the shift/mask added a per-pair branch + setup for
>   zero frame gain -> frame-neutral dead complexity; fails the "bit-identical but
>   frame-neutral -> REVERT" gate). Device tree re-synced to HEAD (kernel md5
>   `2e0c4c7...`, bin `2737d886...`). Tracy: `opt/profiler/ttw-129/`.
> - **Re-scoped lever (do NOT re-attempt the div/mod):** the only way to move
>   `be_main` is to **reduce the per-pair 32B record store traffic** — make the
>   record **born once per gaussian, not once per pair** (the Stage-2b TA-side fill:
>   each gaussian's pairs differ only in the tile-local mean, words 4,5; the other 24
>   bytes are gaussian-invariant and re-stored K times today). That is a structural
>   change, not a per-pair micro-op. The div/mod is settled: it is free.

> **iter-130 — `sort_subchunk_mat` DIAGNOSED then OPTIMIZED via subchunk load-balance (KEEP, bit-identical, frame win).**
> Instrumented the materialize kernel with phase sub-zones (`mat_read`/`mat_sort`/
> `mat_perm`/`mat_write`/`mat_gather`), captured a 30-view Tracy, ran
> `analyze_zones.py`. **The assumed target was REFUTED.** The Sort-Stage-1 (iter-113)
> in-place L1 permute that everyone (incl. this task's framing) assumed was the cost
> is NEGLIGIBLE; the cost is the OVERFLOW gather that Stage 1 never touched.
> Per-view-makespan attribution of `sort_subchunk_mat` = 27.1 ms (instrumented):
> - **`mat_gather` 24.6 ms (91 % of the busiest core)** — the overflow path: random
>   `blendrec[gid]` re-read + scalar fp32→UNORM16 pack (the SAME pack iter-128 found
>   = 71 % of `bucket_emit`) + 32B scatter, for the tiles whose `count > BUCKET_FIT`.
> - `mat_sort` 6.3, `mat_perm` (the L1->L1 permute) **1.7**, `mat_read` (bulk DRAM)
>   0.55, `mat_write` (coalesced slab DRAM) 0.02. **The in-place permute is 1.7 ms —
>   not the cost.**
> - **Load imbalance (root cause):** the shared per-tile count-LPT (built from
>   `pad_counts`, also used by sort/cull/blend) balances TOTAL records/core, but an
>   overflow record costs ~10x an in-budget record in materialize, so cores owning the
>   big overflow tiles are overloaded. Per-core `mat_total`: **max 27.1, p50 17.0,
>   min 10.8; balanced floor (sum/110) = 17.0 ms.** The heaviest single tile (~26.5k
>   recs, ~3-4 subchunks) pins ONE core at ~24.6 ms of gather; a tile-reassignment
>   can't help (one tile can't be split) — only subchunk-granular splitting can.
>
> **Fix (bit-identical):** the unit of work is now a **(tile, subchunk) item**, not a
> tile. Each item is independent and writes byte-identical output regardless of core
> (in-budget permute reads `buf_l1_recs` by tile / writes payload by (tile,sc); gather
> reads `sort_sorted_ids`+`blendrec` by global id / writes payload by (tile,sc)). The
> host (`build_mat_worklist`) enumerates every (tile, sc) with `L_sub>0`, weights
> gather subchunks `GATHER_WEIGHT=8 x records` vs in-budget `1 x records`, and
> greedy-LPT-balances ALL items across the 110 cores into a per-core work-list buffer
> (`buf_mat_work`, flat {tile_id, sc} pairs). The kernel iterates its work-item slice
> (packed `(tile<<8)|sc`), re-deriving per-item metadata (ranges/blend_meta/dir) — the
> in-budget vs gather branch is unchanged per item. (`sort_subchunk_materialize.cpp`
> + `sort_device.cpp launch_subchunk_materialize`; the phase sub-zones were stripped
> for the committed kernel.)
> - **Measured (clean verify, 30 views): hero md5 `e3fefb116d860f99d92bba1ef51d820c`
>   BIT-IDENTICAL, 63.95 dB; ms_view 187.0->181.6 avg (-5.4), min 156.6->147.0 (-9.6).**
>   Tracy (`ttw-130`, clean): **`sort_subchunk_mat` 26.22->19.62 (-6.6, -25 %)**,
>   **NCRISC-KERNEL 134.28->131.42 (-2.9)**, **BRISC-FW (frame critical path)
>   169.02->165.87 (-3.2)**; all other zones unchanged (`sort_bucket_emit` 30.58,
>   `tile_blend_*`, TRISC 51.9 off-path). Materialize IS on the frame critical path
>   (the NCRISC/BRISC-FW win propagated to the steady min). **KEPT.**
> - **Residual:** `sort_subchunk_mat` lands at 19.6 ms vs the 17.0 ms balanced floor —
>   the gap is the per-item metadata re-reads + the GATHER_WEIGHT=8 model not being
>   perfectly tuned + chunky subchunk granularity. Tunable, but the bigger lever is
>   STRUCTURAL: the gather SUM itself (1237 ms/view across cores) is redundant
>   re-gather+re-pack of overflow records — the parked **Stage-2b** (keep overflow
>   records pre-packed in the per-(core,tile) bucket so materialize reads them
>   coalesced, NO `blendrec` re-read / UNORM re-pack) would cut the SUM and drop the
>   floor well below 17 ms. Balance only spreads the work; Stage-2b deletes it.

> **iter-131 — FRESH FULL-FRAME RANKING (Phase 1) + Stage-2b pre-pack op/color at birth (Phase 2, KEEP, bit-identical, frame win).**
>
> **Phase 1 — fresh 30-view Tracy at HEAD (iter 130, `ttw-131`), `analyze_zones.py`,
> per-view busiest-core makespan. Frame critical path BRISC-FW 165.86 ms/view;
> NCRISC-KERNEL (data mover) 131.42; TRISC-KERNEL (SFPU, OFF-PATH) 51.89.** Named
> device zones ranked by pv_makespan:
>
> | # | zone | pv_ms | RISC | role / note |
> |---|------|-------|------|-------------|
> | 1 | sort_bucket_emit | 30.58 | NCRISC | sort in-budget bucket scatter (store-bound per iter-129) |
> | 2 | tile_blend_sfpu | 28.80 | TRISC | blend SFPU compute — **OFF-PATH** (TRISC short pole) |
> | 3 | tile_blend_load | 28.06 | NCRISC | blend DRAM slab reader — ~98% backpressure (waits on SFPU) |
> | 4 | rd_l1_bulk | 27.89 | NCRISC | bulk-L1 read helper (35k pairs; shared by blend/cull/mat) — backpressure |
> | 5 | tile_mb_mask | 22.39 | TRISC | cull mask compute — **OFF-PATH** |
> | 6 | tile_l1_cull_rd | 21.99 | NCRISC | cull DRAM slab reader — backpressure |
> | 7 | proj_scatter | 20.83 | BRISC | gather/project compaction + SoA + blendrec write |
> | 8 | sort_subchunk_mat | 19.62 | NCRISC | materialize: overflow gather + in-budget permute |
> | 9 | proj_count | 14.97 | BRISC | project visible count |
> | 10 | ta_bucket_scatter | 12.24 | NCRISC | tile_assign K2 pair scatter |
> | 11 | ta_gauss_aabb | 9.70 | NCRISC | tile_assign K1 bbox |
> | 12 | sort_tile_depth | 7.15 | NCRISC | DRAM radix |
> | — | pfwc 1.92 / sort_bin_hist 1.41 / sort_subchunk_dir 2.41 / mcam 0.67 | | | |
>
> **Top-3 reducible levers (NCRISC dataflow drives BRISC-FW; iter-128/130 proved the
> propagation):** (1) **sort_bucket_emit 30.58** — residual is the per-pair 32B-record
> L1 store traffic (pack hoisted iter-128, div/mod refuted iter-129); structural-only.
> (2) **sort_subchunk_mat 19.62** — dominated by the OVERFLOW gather: per-record random
> `blendrec[gid]` read + fp32→UNORM16 re-pack (the iter-129/130 theme). (3) the
> **blend/cull readers cluster** (tile_blend_load 28 + tile_l1_cull_rd 22 + rd_l1_bulk 28)
> — ~98% backpressure, gated by the SFPU/CB pipeline, NOT raw read traffic (needs the
> parked blend phasing plan, not a reader micro-opt). The off-path TRISC zones
> (tile_blend_sfpu 28.8, tile_mb_mask 22.4) cannot move the frame (TRISC 51.9 short pole).
>
> **Phase 2 — measure-first ablation of the materialize overflow gather (`ttw-131sp`/
> `ttw-131sr`, kernel #if knobs, NOT bit-identical, reverted):**
> - **skip the per-record fp32→UNORM16 pack + tile-mean subtract:** sort_subchunk_mat
>   19.62→14.32, **BRISC-FW 165.86→158.76 (−7.1)**, NCRISC-KERNEL −7.4. The pack is the
>   dominant gather cost.
> - **skip the random `blendrec[gid]` 64B read:** sort_subchunk_mat 19.62→16.78,
>   **BRISC-FW −4.5**, NCRISC −4.8. Real but smaller than the pack.
>
> **Fix (bit-identical):** the op/color UNORM16 words are functions of the GAUSSIAN
> only and were re-derived by BOTH consumers — `sort_bin` `pack_invariants` (once per
> gaussian) AND the depth-sorted `materialize` overflow gather (once per overflow
> record, un-hoistable since the ids are depth-sorted). Move the pack to BIRTH:
> `gather_visible_scatter` now writes `blendrec` words `r[5]=unorm(op)|(unorm(cr)<<16)`,
> `r[6]=unorm(cg)|(unorm(cb)<<16)` (formerly fp32 op,cr,cg,cb in r[5..8]); the two
> consumers just COPY r[5],r[6]. Same UNORM formula, same fp32 inputs ⇒ byte-identical
> record bytes. (`blendrec` op/color words 5–8 are read ONLY by these two — the cull
> reader voids `blendrec`, the blend reader declares but never reads it; verified.)
> Stays 64B/gaussian — no host alloc, no read-width change.
> - **Measured (clean verify, 30 views): hero md5 `e3fefb116d860f99d92bba1ef51d820c`
>   BIT-IDENTICAL, 63.95 dB; ms_view 181.6→178.9 avg (−2.7), min 147.0→143.8 (−3.2).**
>   Tracy (`ttw-131`): **NCRISC-KERNEL 131.42→113.80 (−17.6)**, **BRISC-FW (frame
>   critical path) 165.86→162.12 (−3.74)**; `sort_bucket_emit` 30.58→18.83 (−11.7),
>   `sort_subchunk_mat` 19.62→13.74 (−5.9). The duplicated pack is DELETED (now computed
>   once at birth, not in bucket_emit AND materialize). **KEPT.**
> - **TRADEOFF (documented):** the pack RELOCATED to the producer grows `proj_scatter`
>   20.83→35.40 (+14.6, BRISC) and BRISC-KERNEL 86.6→100.5 — and BRISC-FW is the long
>   pole, so only the NET de-duplication (~materialize's −5.9) reaches the frame (−3.7).
>   The pack is fundamentally ~12–15 ms of scalar UNORM work wherever it runs; we
>   removed the *duplicate*, not the work. **Better follow-up (iter 132+):** keep the
>   pack on NCRISC (it has 48 ms headroom now: 113.8 vs BRISC 162) by having
>   `bucket_emit`'s once-per-gaussian `pack_invariants` publish the 2 packed words to a
>   small resident gaussian-major array that `materialize` reads — deletes materialize's
>   re-pack WITHOUT loading the BRISC long pole. (Needs a tiny buffer + arg plumbing; the
>   gather-side pre-pack shipped here is the lower-risk first increment.)

> **iter-132 — relocate the op/color UNORM16 pack OFF the BRISC `proj_scatter` long pole
> onto NCRISC `sort_bin` `pack_invariants` (the iter-131 follow-up above). KEEP,
> bit-identical, small reproducible frame win.**
>
> **What shipped:** revert iter-131's gather birth-pack — `gather_visible_scatter` again
> writes raw fp32 `op,cr,cg,cb` to `blendrec` words `r[5..8]` (so `proj_scatter` returns
> to ~20.8). `sort_bin` `pack_invariants` (once per gaussian) computes the two UNORM16
> words and (a) feeds them into this core's in-budget bucket record AND (b) PUBLISHES them
> into `blendrec[10],[11]` so the depth-sorted `materialize` overflow gather just COPIES
> `aos[10],aos[11]` (no re-pack, no extra read). Same fp32 inputs + same UNORM formula ⇒
> byte-identical. New CB(13) ring + `cb(13, …)` host alloc; static, no over-provisioning.
>
> **Publish write granularity (measured, this is the crux):** sub-page 8B/4B splats to
> `blendrec[g]`+40 (words 10,11) did **NOT land** on this BH — correctness collapsed to
> 25.5 dB (materialize read zeros). 8B at a non-16B-aligned offset is below the DRAM write
> granule. A full-64B page write-back lands (mirrors gather's only reliable pattern) but
> costs +15.4 NCRISC. The shipped fix writes the **minimal 16B-aligned chunk** = a 16B
> write at byte-offset 32 covering words `[8,9,10,11]` (16-aligned offset + 16B size = the
> natural DRAM granule). Words 8,9 are re-written with their EXACT original gather bytes
> (cb, depth/0), so a boundary gaussian processed by two cores writes byte-identical 16B —
> fully idempotent, no clobber. Diagnostic that localized the bug: temporarily re-packing
> materialize from fp32 `aos[5..8]` gave 63.95 dB, proving gather+sort_bin were correct and
> the publish path was the only fault.
>
> **Measured (same-thermal apples-to-apples, iter-131 HEAD rebuilt + rerun NOW, 4+4 verify
> passes each):** hero md5 `e3fefb116d860f99d92bba1ef51d820c` **BIT-IDENTICAL**, 63.95 dB.
> - **ms_view avg 178.8→177.1 (−1.7); min 143.9→142.3 (−1.6).** Min clusters cleanly
>   separated with no overlap (iter-131 143.7–144.1 vs iter-132 142.1–142.7 across 8 runs)
>   ⇒ real, reproducible, not thermal noise.
> - Tracy (`ttw-132`): **`proj_scatter` 35.40→20.83 (−14.6)** ✓ (revert worked), but the
>   publish cost landed in **`sort_bucket_emit` 18.83→32.45 (+13.6)** / NCRISC-KERNEL
>   113.80→127.34 (+13.5). **BRISC-FW makespan 162.12→161.71 (≈flat, −0.4).**
>
> **Honest read of the tradeoff:** the +14.6 `proj_scatter` cost did NOT vanish — it moved
> to NCRISC `sort_bucket_emit` as ~+13.6 of per-gaussian write-ISSUE overhead (64B vs 16B
> write barely differed: 129.2 vs 127.3 NCRISC ⇒ the cost is the scatter *write count*, not
> bytes). The pack is fundamentally ~13–15 ms of work + a scatter publish wherever it runs.
> BRISC-FW makespan is flat, yet the wall-clock frame improves a small, reproducible −1.6 ms
> (likely reduced BRISC producer-side stall/backpressure that the summed-makespan proxy
> doesn't capture). Gate is `bit-identical AND ms_view improves` — both hold — so **KEPT**,
> but the win is marginal. **Next lever to actually bank the −14.6:** publish ONLY for
> overflow/heavy-tile gaussians (materialize reads `blendrec` only for the dense fallback;
> in-budget records already carry the packed words in their bucket slot) to cut the publish
> write count far below one-per-gaussian — the per-gaussian scatter issue is what eats the
> proj_scatter saving.

### Stage 3 — Per-core L1 sort, emit blend's PACK2 layout into L1
Each sort core reads ITS tile's bucket into L1 and radix-sorts **(key, index)**
(8B ping-pong — small scratch). Then the **emit pass writes the depth-sorted 32B
PACK2 slab directly into L1** — blend's *exact* consumed layout (record `g` at
byte `g*32`, **cull mask stamped into word3**, see
`reader_alpha_blend_mb_devcull.cpp`). The same core
(`build_lpt` / `compute_lpt_assignment`) culls/blends from that L1 slab.
- Fold the permute INTO the emit: `for i: slab[i] = bucket[index[i]]`. **No
  blend-side index/permute, no DRAM `materialize`** — the sort output *is* the
  blend input.
- Capacity: the emit needs a 2nd buffer (source bucket + sorted slab). Both fit
  for ≤~700 KB tiles; the heaviest (849 KB ⇒ 1.7 MB > L1) emit in `MB_BUCKET_FIT`
  subchunks (the reader already loops `sc`) or spill that tile's slab to DRAM.
- **Emit in the blend SFPU's vector layout directly (O2)** — field-SoA in
  `cull_perm(g,m)` order so the SFPU loads each field as 32 contiguous lanes (no
  AoS→lane, no broadcast-gather). Gate the exact lane order on the blend compute's
  DEST load; co-design with the blend plan's phasing.
- Win: deletes `materialize`; cull/blend drop their DRAM slab readers.
- Risk: medium-high; the direct-L1-read-on-MATH race needs the documented
  `invalidate_l1_cache` + MATH→UNPACK handshake (`quirks.md`). Sequence **with**
  the blend plan, not before it.

### Stage 4 — Heavy-tile handling
- Default to **(key, index)** sort + index/permute so the 849 KB max tile stays
  resident (full-32B ping-pong = 1.7 MB does NOT fit — that path alone needs the
  subchunk split).
- **Hybrid spill**: the few tiles whose contiguous depth-ordered copy won't fit
  keep the existing DRAM subchunk path; the in-budget majority goes L1-resident.

### Stage 5 — Host-free single trace (O3–O5)
Make the whole frame one Metal trace replayed per view (see "End-state" above).
- 5a: device-ify the inter-stage glue (count→prefix→offsets, directory) and move
  per-view params to a device buffer.
- 5b: **device-side LPT** writing a resident `tile→core` assignment (mandatory for
  trace; fallback = static map).
- 5c: persistent per-core kernels looping the assignment, per-tile double-buffered.
- 5d: `BeginTraceCapture` the frame; per view = `EnqueueWriteBuffer(camera)` +
  `ReplayTrace`. Static alloc to `padded_n` ceilings.
- Win: ~50+ programs/`Finish` per frame → ~0 host dispatch.
- Risk: high (biggest restructure); depends on Stages 1–4 being device-resident.

> **iter-120 DIAGNOSTIC — host-dispatch CHARACTERIZED; redundant-`Finish` removal
> proven FRAME-NEUTRAL; concrete Stage-5 blocker list below.**
>
> **Phase-1 host-wall breakdown** (`GSPLAT_TT_HOST_PROFILE=1` chrono in
> `render_view`, avg of 30 timed views; the Python `ms_view`/`avg_frame_ms` IS this
> host wall because every stage drains the one shared in-order CQ via `Finish` /
> blocking read, so per-stage wall = dispatch + device makespan + readback):
>
> | stage | ms/view | share | host syncs (steady state) |
> |-------|---------|-------|---------------------------|
> | **project** | **37.5** | 19% | means_cam `Finish`, pfwc `Finish`, gather **M-read** |
> | — means_cam | 0.72 | <1% | 1 enq + 1 Finish (resident bubble) |
> | — pfwc | 1.91 | 1% | 1 enq + 1 Finish (resident bubble) |
> | — gather | 34.9 | 18% | 3 enq (count/scan/scatter) + 1 blocking **M-read** |
> | **tile_assign** | **23.8** | 12% | scan `Finish`, **P-read**, K2 `Finish` |
> | **sort+cull+blend** | **133.8** | **69%** | **P-read(re)**, bin-cnt `Finish`, **hist-read**+host-LPT+6×H2D, bin-scat `Finish`, routeC-cull `Finish`, radix `Finish`, subchunk-dir `Finish`, blend drain + image D2H |
> | post (tile_ranges loop) | 0.002 | 0% | — |
> | **TOTAL** | **195.1** | | ~**18** `EnqueueMeshWorkload`, ~**13** CQ drains/view |
>
> Maps to the iter-116 device zones (BRISC-FW 179.7 ≈ frame, BRISC-KERNEL 87.1 ⇒
> 92 ms BRISC non-kernel): the 92 ms is **on-device BRISC-FW per-program
> launch/barrier firmware overhead**, spread across the ~18 programs — and it lives
> mostly in **sort+cull+blend (69 % of the host wall, the most programs)**, NOT in
> project's two tiny resident `Finish` bubbles (0.72 + 1.91 ms).
>
> **Phase-2 experiment (REVERTED, frame-neutral).** Removed the 5 clearly-redundant
> `Finish` barriers — means_cam (resident), pfwc (resident), tile_assign scan
> (redundant with the very next blocking P-read), tile_assign K2 (bubble; sort's
> P-read drains), sort bin-count (redundant with the next blocking hist-read). All
> are provably safe on the single in-order CQ (resident NoC handoff; a following
> blocking read re-imposes ordering). **Measured 195.1 → 196.1 ms/view (frame-
> neutral, +0.9 = noise):** the per-stage wall just SHIFTS — `ta` 23.8→13.3,
> `gather` 34.9→37.3, `sort` 133.8→145.3. **Root cause:** with negligible host
> compute, a removed `Finish` is immediately re-imposed by the next blocking read
> (M/P/hist), and the on-device per-program launch overhead is unchanged. **So
> host-side `Finish` removal CANNOT move this frame** — reverted to pristine
> (hero md5 `e3fefb11…`, 63.95 dB, ms_view 195.0). The only levers that touch the
> 92 ms are: (a) **fewer programs** (true fusion), or (b) **Metal Trace** that
> pre-records the ~18-program dispatch so BRISC-FW replays it without re-launching.

### Stage 5 — MEASURED: Metal Trace payoff (iter-126, DIAGNOSTIC — REFUTES the trace endgame as a perf lever)
**Verdict: NO-GO.** A gated measurement prototype (`GSPLAT_TT_TRACE_PROTO=1`,
since reverted) opened the `MeshDevice` with a 256 MB trace region + program cache,
captured the real hero-view program stream, and timed `ReplayTrace` vs host dispatch
on the device (`yyzo-bh-07`, N=50). **Metal Trace replays the recorded commands
bit-identically but removes essentially ZERO frame time** — because this frame is
**on-device-execution bound, not host-dispatch bound**, and trace only removes
host-side dispatch (which is already fully overlapped on the in-order async CQ).

Measured (per-iter, ms; bit-identical confirmed — `ta_pairs_gid` FNV-1a
`4587123369709330712` matches across host-reenqueue == trace-replay == after-frame,
≠ zeroed buffer):

| chain (distinct programs, single-enqueue) | host pipelined | host drained (Finish/prog) | **trace replay** | per-drain bubble |
| --- | --- | --- | --- | --- |
| tile_assign sub-chain (5 heavy programs, ~57 ms) | 57.106 | 57.169 | **57.102** | **15.6 µs** |
| project head means_cam+pfwc (2 light programs, ~2.5 ms) | 2.5132 | 2.5333 | **2.5125** | **20.0 µs** |

- **Trace ≈ host, to within noise** (heavy Δ = +0.004 ms pipelined / +0.067 ms
  drained; light Δ = +0.0007 / +0.021 ms). The light-program probe is decisive: the
  per-drain bubble is **program-weight-independent (~16–20 µs)** — heavy TA compute
  was NOT masking a large per-launch cost; the host-dispatch cost is genuinely ~µs.
- **Implied full-frame trace-removable host dispatch** = ~13 CQ drains × ~18 µs +
  ~20 programs × ~1–4 µs ≈ **< 0.5 ms** — three orders of magnitude below the
  hypothesized ~92 ms / ~45–60 ms.
- **The ~145–160 ms post-trace floor is REFUTED.** Metal Trace does NOT reduce the
  on-device BRISC-FW per-program launch/barrier firmware: the bit-identical 5-program
  replay takes the SAME wall time as host dispatch (57.102 vs 57.106). The ~92 ms
  "BRISC-FW − BRISC-KERNEL" is **on-device firmware execution** (launch_msg / NoC /
  CB-setup / barrier / inter-program worker idle), which a trace **re-executes
  unchanged** — it only pre-stages the host→prefetcher command issue, and the
  measurement proves the dispatcher already keeps the device fed.
- This is corroborated by iter-120 (host `Finish`-removal frame-neutral) and by the
  S5.1–S5.4 device-residency landings coming out **frame-neutral / slightly
  regressed** — i.e. the inter-stage host glue that device-residency removes was
  already small/overlapped, so neither the prerequisites nor the trace capstone
  unlock a hidden 45–60 ms.

**tt-metal trace gotchas discovered (for whoever revisits this):**
1. `DEFAULT_TRACE_REGION_SIZE = 0` — must pass an explicit `trace_region_size` to
   `MeshDevice::create_unit_mesh` (used 256 MB) or capture is a no-op / asserts.
2. `enable_program_cache()` is **required** before capture (trace replays cached
   program binaries).
3. **Per-program double-enqueue is uncapturable as one replay unit.** A `MeshWorkload`
   re-enqueued with different `SetRuntimeArgs` within a frame (gather workload mode
   0→1; sort `wl_bin` reuse) bakes whatever args were live at capture; replay re-runs
   them. The measurement had to restrict capture to an **all-distinct sub-chain**
   (verified `chain_distinct=true`). Real Stage 5 would need a distinct program
   instance per logical dispatch, or fully device-resident/stable args.
4. **No host ops inside `BeginTraceCapture`…`EndTraceCapture`** — confirms the listed
   blockers (M/P/hist reads, `build_subchunk_layout`) are hard constraints.
5. **Buffer addresses are baked into the trace** — replay reuses captured addresses;
   works only because `device_state` buffers are persistent (no realloc). A real impl
   must statically allocate to ceilings and never reallocate across views.

**Recommendation:** do NOT spend the 4–6 prerequisite cycles on the host-free single
trace as a *performance* lever — the measured trace payoff is < 0.5 ms. The frame is
bound by **on-device firmware + NCRISC dataflow** (`sort_bucket_emit` ~40 ms,
`sort_subchunk_mat` ~26 ms, the single-core bin-layout ~+13 ms, blend SFPU). The only
levers that move the 92 ms are **fewer programs (true fusion)** and **faster
kernels** — not trace. (Trace may still be worth it later as an ergonomics/
determinism tool once program count is already low, but it is not the perf endgame.)

### Stage 5 — DISABLED (iter-127): trace refuted ⇒ the prerequisites are pure regression

> **DECISION (iter 127, commit on `smarton/stage2-hostfree-l1`): S5.1–S5.5 are
> DISABLED (gated off, code KEPT).** Every Stage-5 sub-landing (S5.1 on-device bin
> layout, S5.2 cheap layout, S5.3 host-free M/P over-provisioning, S5.4 parallel
> Pass-2 emit, S5.5 — none of the persistent-kernel work landed) was justified
> **SOLELY** as a Metal Trace prerequisite. iter-126 **MEASURED** the trace endgame
> to be a no-go (replay removes < 0.5 ms; the ~92 ms `BRISC-FW − BRISC-KERNEL` is
> on-device firmware + NCRISC dataflow that the trace replays **unchanged**). With
> the trace refuted, the prerequisites are no longer prerequisites — they are pure
> single-core / over-provision **regression**:
> - `sort_device_layout_enabled()` (S5.1/S5.2/S5.4): the on-device layout runs
>   **single-core** (the parallel Pass-2 emit only recovered ~1.5 ms of the +23 ms);
>   in isolation it is slower than the fast host `host_bin_layout_from_hist` +
>   `build_lpt`. **+~23 ms** vs the host path (NCRISC-KERNEL `144.5 → 189.9`).
> - `host_free_mp_enabled()` (S5.3): over-provisioning the M-domain
>   (`tile_assign` K1) and P-domain (K2 + pair buffers) to the **static ceilings**
>   (`pair_ceiling = 4,718,592`, `n_ceil`) does worst-case no-op work **every
>   frame** regardless of the real M/P — **+~31 ms** (`ta_gauss_aabb 9.7 → 35.0`,
>   `ta_bucket_scatter 12.2 → 18.4`). It is only "frame-neutral" once a trace would
>   amortize the launches it deletes — which iter-126 proved never happens.
>
> **iter-127 flips BOTH flags `false`** → reverts to the host bin-layout +
> real-M/P-read `tile_assign` (the pre-iter-121 working path). Bit-identical
> (hero md5 `e3fefb116d860f99d92bba1ef51d820c`, 63.95 dB). ms_view recovered
> `236.7 (iter-125 ledger) / ~205–211 (steady)` → **195.6 avg, min 163.7**;
> per-view makespan **BRISC-FW `224.7 → 179.1`, NCRISC-KERNEL `189.9 → 144.4`** —
> exactly the iter-116 best-known baseline. Full regression recovery.
>
> **The S5.x code is KEPT, gated off** (`sort_bin_layout.cpp`, `sort_bin_emit.cpp`,
> the `host_free_mp` over-provision branches). It is NOT dead: it may matter for a
> future **fusion** lever (the real way to attack the 92 ms BRISC-FW is fewer
> programs — e.g. fusing the bin layout INTO the bin-count or sort kernel so it
> adds no separate launch, at which point on-device residency is free). Re-enable
> only behind a measured fusion that makes residency a net win. **Do NOT re-flip
> these on as a standalone "host-free" step — that is the regression iter-127 just
> recovered.** The ordered prerequisite plan below is retained for that future
> fusion attempt; treat S5.1–S5.6 as "available but parked", not "next".

### Stage 5 — concrete ordered host-free plan (what blocks the single trace TODAY)
A trace records a FIXED `EnqueueProgram` sequence with fixed runtime args + buffer
addresses; it cannot host-compute a size/branch mid-frame. So every **blocking host
read that sizes or gates the next dispatch** is a trace blocker. From the iter-120
audit, in dependency order:

1. **gather `M`-read** (`gather_visible_device.cpp:833`). Host reads the visible
   count `M` to size tile_assign + return `depths(M)` + early-out `M==0`.
   → **5b-i:** over-provision tile_assign to the `padded_n` ceiling (`N` rounded);
   its kernels already guard `g0>=M` and read `proj_M` resident — drop the host `M`
   dependency. `M==0` becomes a device no-op (guards already exist).
2. **tile_assign `P`-read** (`tile_assign_device.cpp:702`). Host reads pair count
   `P` to size `buf_gids/tids/keep` (`P_pad`) + the K2/cull work-split.
   → **5b-ii:** alloc pair buffers to a static `P_max` ceiling (worst-case
   Σ tiles_per_gaussian; measure a safe bound), have K2/K4 read `P` from resident
   `buf_pairs_P` and guard their work-splits. Drop the host `P` dependency.
3. **sort `P`-read** (`sort_device.cpp:1217`) — a **redundant re-read** of the same
   `ta_pairs_P`. → free once #2 lands (size from the static ceiling; read `P`
   on-device for the kernel guards).
4. **sort `hist`-read + host LPT + 6×H2D** (`sort_device.cpp:1595` →
   `host_bin_layout_from_hist` → re-upload `bin2d/tmeta/tile_ids/bucket_meta/`
   `l1_rec_base`). This is the **single biggest mid-frame host blocker** (host
   compute in the critical path). → **5a (DO THIS FIRST):** the on-device layout
   path **already exists** — `env_config::sort_device_layout_enabled()` (currently
   `return false`) computes the page layout on-device and reads only a tiny ctrl
   page (`read_bin_layout_ctrl`). Flip it on and validate bit-identical. Then
   **5a-ii** port LPT balance into that kernel (per-view; reuse the `build_lpt`
   algorithm — do not re-tune its 1.013) writing resident `lpt_meta/tile_ids/`
   `tile_ranges`, **or** accept the static tile→core map fallback. Removes the
   hist-read + host LPT + the 6 H2D writes.
5. **host `build_subchunk_layout`** (from `counts`) + `prepare_subchunk_buffers`.
   → device-side or **static padded** subchunk directory/alloc (`O4`).
6. **final `tile_ranges` read** (post): stats-only — drop it; the per-view image
   D2H is the one allowed host op.

**Ordered execution (lowest-risk first, each its own gate):**
- **S5.1 — DONE (iter 121, commit `0a8cd86`).** Flipped
  `sort_device_layout_enabled()=true`. The path's kernel
  `render/kernels/dataflow/sort_bin_layout.cpp` was **MISSING** (host referenced
  it; JIT failed) — that's why the flag was off. Recovered it (git `61f61ad`),
  rewrote as an efficient single-core bulk-row pass (`read_pages`/`write_pages`,
  2 streaming passes over core rows) **with on-device LPT** (`build_lpt`
  algorithm, balance unchanged), and added the missing `l1_rec_base` write
  (arg13 `l1_base_addr`, arg12 `bucket_fit`) so the live L1_RECORD scatter base
  is populated on-device. **VERIFY (`layout_verify=true`, 30 views): device ==
  `host_bin_layout_from_hist` BIT-EXACT, `dev_status==0`, zero mismatches across
  hist/counts/tids/lpt_meta/tmeta incl. heavy/overflow + l1_record tiles.**
  **Removed host work:** hist D2H read, **host-LPT (now fully on-device — NOT
  deferred to S5.2)**, and all **6 H2D re-uploads**
  (`bin2d/tmeta/tile_ids/bucket_meta/l1_rec_base/...`); host now reads only a
  tiny ctrl page + already-resident buffers. **Pixels bit-identical** (hero md5
  `e3fefb11…`, 63.95 dB). **Perf: ms_view 195.5 → 215.0 (+19.5, +10%)** — the
  +23 ms is the new single-core layout kernel (BRISC-FW 179.7→202.5,
  NCRISC-KERNEL 144.5→167.6; all named sort zones unchanged). Single-core serial
  layout is slower than the fast host CPU **in isolation**; the host-dispatch /
  6×H2D win only materializes once Metal Trace hides the per-program launch
  overhead. **KEPT** anyway: bit-identical + it's the load-bearing prerequisite
  that lets the sort stage be captured into a trace (host can no longer compute
  size/branch mid-frame here).
- **S5.2 — PARTIAL (iter 122, commit on `smarton/stage2-hostfree-l1`).** Made the
  single-core layout kernel (`sort_bin_layout.cpp`) cheap, bit-exact:
  1. **L1-cache the per-core histogram** (`num_cores×row_span` u32 ≈ 450 KB on
     hero, CB_HCACHE = 512 KB) so Pass 1 streams DRAM once and Pass 2 reads the
     cache — **one DRAM read pass instead of two** (falls back to a 2nd DRAM read
     if `num_cores*row_span > HCACHE_CAP`).
  2. **LSD radix sort** (4× 8-bit passes on the composite key `(cost<<16)|tid`,
     `cost=tile_pad ≤ 32768 < 2^16`, `tid < 2^16`) replaces the **O(n²)≈524k-iter
     selection sort** for the LPT order. The key is a strict total order on
     `(cost, tile_id)`, so the radix order is **bit-identical** to the old
     comparator. Greedy first-min-load assignment kept sequential/unchanged.
  - **VERIFY (`layout_verify=true`, 30 views + warmup): device == host BIT-EXACT,
    `status=0`, zero mismatches** across hist/counts/tids/lpt_meta/tmeta (incl.
    heavy/overflow tiles). **Pixels bit-identical** (hero md5 `e3fefb11…`, 63.95 dB).
  - **Perf: ms_view 215.0 → 205.0 (−10.0, recovers ~half the +23 ms regression).**
    Per-view makespan (`analyze_zones.py`, ttw-122): **BRISC-FW 202.5 → 192.3
    (−10.2)**, **NCRISC-KERNEL 167.6 → 157.4 (−10.2)**; all named sort zones
    unchanged (`sort_bucket_emit` 39.7, `sort_subchunk_mat` 26.1, …) — the win is
    entirely the (unnamed, single-core) layout kernel, now **≈+13 ms vs the
    iter-116 195.5 baseline** (was +23). **KEPT** (bit-identical + clear win).
  - **Residual ≈+13 ms** = Pass 2's per-`(core,tile)` `bin2d`+`l1_rec_base`
    bulk **DRAM writes** (num_cores×row_pages 64B-page writes) + Pass 1's single
    read pass, all on **one core**. The L1 cache removed the 2nd read pass and the
    radix removed the scalar sort; the irreducible remaining cost is the serial
    write traffic. **To close it: lever #4 — partition the count/emit across cores
    with a two-phase parallel prefix-sum** (per-core-block local sums → exclusive
    scan of block totals → local offsets; needs semaphore sync, keep bit-exact via
    the verify harness), **or** fold it under Metal Trace (the +13 ms single-core
    cost is hidden once BRISC-FW replays the recorded dispatch). Also the static
    `tile→core` map fallback (5b) remains open.
- **S5.4 — PARTIAL (iter-125, commit `78228e0`).** Lever #4, first increment:
  parallelize **Pass 2** (the dominant ~14k per-`(core,tile)` `bin2d`+`l1_rec_base`
  base writes) across **16 emit cores** (new kernel
  `render/kernels/dataflow/sort_bin_emit.cpp` + workload `wl_bin_layout_emit`).
  Design (NO semaphores — the codebase has none; used the proven CQ program-order
  pattern instead):
  1. The coordinator (`sort_bin_layout.cpp`) keeps Pass 1 (count) + tile-major
     prefix + all resident metadata + LPT, but **replaces the Pass-2 write loop with
     a checkpoint publish**: for each of W=16 workers (covering a contiguous
     source-core range via the same `base/rem` split the host uses) it snapshots the
     running `page_acc` (pages) + `rec_acc` (real counts) at the worker's FIRST
     source-core into a small DRAM ckpt buffer (`buf_layout_ckpt`, W×2×row_span u32),
     then advances the prefix over that range. The advance loop is the OLD Pass-2
     accumulation (`ceil_pages(0)==0` so the unconditional add matches the `if h>0`).
  2. The emit kernel (CQ-ordered right after the coordinator → sees its ckpt writes
     and the still-intact histogram in `bin2d`) has each worker seed `page_acc`/
     `rec_acc` from its checkpoint and **replay the exact Pass-2 inner loop** over
     its disjoint source-core rows (read hist row → emit `bin2d` base + `l1_rec_base`
     → write back). Disjoint per-`(core)` DRAM regions ⇒ contention-free, no locks.
  - **VERIFY (`layout_verify=true`, 30 views + warmup): 33/33 `IDENTICAL`,
    `status=0`, zero mismatches** across hist/counts/tids/lpt_meta/tmeta (incl.
    heavy/overflow). **Pixels bit-identical** (hero md5 unchanged, **63.95 dB**).
  - **Perf (matched-thermal A/B, back-to-back, steady-state MIN):** `min_ms`
    **213.0 → 211.5 (−1.5)**; thermally-matched Tracy makespan **BRISC-FW
    226.2 → 224.7 (−1.6)**, **NCRISC-KERNEL 191.4 → 189.9 (−1.6)**, **TRISC-KERNEL
    51.9 unchanged** (off-path untouched). 30-view Tracy `opt/profiler/ttw-125/`
    (725k device-zone rows). **KEPT** (bit-identical + a real, if small, win).
  - **Why only ~1.5 ms of the ~6 ms Pass-2 share lands:** the second program's
    per-launch BRISC-FW overhead (~2.2 ms measured) + the serial checkpoint publish
    (W×2×row_pages ≈ 2k page writes, ~1.3 ms) eat most of it; and **Pass 1's single
    serial count read (~4–5 ms) is still on the coordinator** (untouched). **Bigger
    win remaining:** (a) parallelize Pass 1 too via the partials/combine 3-phase (or
    a single-program semaphore version — needs the cross-core NoC-coord handshake,
    no codebase precedent), and (b) **Metal Trace makes the extra launch ~free** so
    the full Pass-2 parallelism (~6 ms) materializes — i.e. this change is worth
    MORE under the trace endgame than the current +1.5 ms shows.
- **S5.3:** over-provision tile_assign + gather to static ceilings; kernels read
  `M`/`P` resident + guard work-splits → delete the M/P-read drains (5b-i/ii).
  - **GROUNDWORK LANDED (iter-123, KEEP, behind `host_free_mp_enabled()` default
    true):** tile_assign now reads the visible-count `M` from the **resident
    `proj_M` control page** via a runtime `InterleavedAddrGen<true>` ctrl-page read
    (bbox K1 + scan_reduce + scan_add), and over-provisions the M-domain
    offsets/scan buffers + K1 page-split to a **static page-aligned ceiling
    `n_ceil`** (= `proj_m_px` capacity) instead of the host-read `M`. Reversible
    via the flag. **Bit-identical** (hero md5 `e3fefb11…`, 63.95 dB, 30 views),
    **frame-neutral** at the steady floor (`min_ms 204.0`; the verify avg 232.9 is
    thermal from back-to-back runs). This lands the proven **resident-read +
    static-ceiling pattern** the three read-deletions depend on, but **does NOT yet
    delete any host blocking read.**
  - **P-DOMAIN DELETION LANDED (iter-124, KEEP, behind `host_free_mp_enabled()`):**
    deletes the **mid-frame host `tile_assign` `P`-read** by over-provisioning the
    pair buffers + K2 scatter work-split to the static `P_max = pair_ceiling()`
    (4,718,592); K2 reads the clamped `P` from the **resident `ta_pairs_P` ctrl page**
    via `InterleavedAddrGen<true>`; `scan_bases` clamps the published `P` and publishes
    an overflow flag + true `P`; `sort_device` hard-fails on overflow (single-path TT,
    no fallback — a too-small ceiling would corrupt output). All behind the flag
    (reversible). **Bit-identical** (hero md5 `e3fefb116d860f99d92bba1ef51d820c`,
    63.95 dB, 30 views; per-view P varies 2.4M–3.7M, all < ceiling, no overflow fired),
    **frame-neutral** (`min_ms 213.0`; the verify avg 238.4 is thermal). 30-view Tracy
    at `opt/profiler/ttw-124/render.tracy` (705k device-zone rows). Code preserved on
    `wip/s5.3-p-read-iter124` (`dc7f4b5`) and now merged here.
  - **STILL PENDING read-deletions (mid-frame host blocking-read drains) — both
    ASSESSED + DEFERRED (NOT a clean tonight landing; not the "small follow-ons"
    first expected):**
    1. `sort` `P`-read (`sort_device.cpp:~1240`) — **DEFERRED.** Two real blockers,
       not a drop-in of the iter-124 ceiling pattern:
       (a) **Load-balance / frame regression.** The bin pass per-core work-split
       (`split_pages(P_pad/16, num_cores)`, `sort_device.cpp:1269`) is **contiguous**
       and sized to the REAL `P` so the dominant sort bin compute (~59 ms,
       `sort_bin.cpp` count+scatter) spreads evenly across all cores. Re-sizing it to
       the static `pair_ceiling()` (the iter-124 trick) concentrates the real pairs on
       only the first `~(P/ceiling)·N` cores (≈76 % active at P≈3.6 M, ≈51 % at
       P≈2.4 M), inflating the bin pass ~+30 % to ~+2× → ~+9 % to much-larger frame
       regression — **fails the ≤3 % gate.** iter-124's K2 scatter tolerated the
       ceiling split because it is a thin pass; the sort bin pass is the frame's
       largest compute. A clean deletion needs an **on-device balanced page-split**
       (small kernel computing per-core `[start,count)` from resident `P` at the REAL
       `P`, not the ceiling) — bit-identity needs the split to stay contiguous +
       gaussian-major.
       (b) **Overflow safety-net coupling.** iter-124 deliberately placed the
       pair-ceiling overflow hard-fail (`ta_pairs_P[2]/[3]`) on THIS sort P-read (the
       last mid-frame consumer of `ta_pairs_P`). Deleting it drops the safety net
       unless it is relocated on-device — cleanest is to have `sort_bin_layout.cpp`
       surface overflow/`P_true` into its layout ctrl page so the host picks it up via
       the EXISTING post-count `read_bin_layout_ctrl` read (no new drain).
       NB: the sort stage also still does ~5 other mid-frame metadata reads in
       device-layout mode (`read_bin_layout_ctrl` P_kept/P_aligned, tile_counts,
       tile_ranges, tmeta, lpt_meta, tile_ids), so deleting just the P-read does **not**
       make the sort stage trace-able on its own.
    2. `gather` `M`-read (`gather_visible_device.cpp:~833`) — **DEFERRED** (higher
       risk, as flagged). `M` (post-cull visible count) is read after count+scan+
       scatter and used to size `depths`/outputs + gate downstream; the contract change
       (size to capacity, `M==0`→device no-op, `M` post-frame for stats) is a larger
       refactor best done after the sort metadata reads are also addressed.
- **S5.4:** static/device subchunk alloc (5c) → no host `build_subchunk_layout`.
- **S5.5:** persistent per-core kernels looping the resident assignment (variable
  per-view iteration is device data → one trace replays for any view).
- **S5.6:** `BeginTraceCapture` the frame; per view = `EnqueueWriteBuffer(camera)`
  + `ReplayTrace` + one image D2H. ~~This is the step that finally removes the 92 ms
  BRISC-FW per-program launch overhead (the only thing that can).~~ **SUPERSEDED by
  the iter-126 measurement above: Metal Trace removes < 0.5 ms here (it replays the
  on-device launch firmware unchanged). S5.6 is NOT a perf lever — the 92 ms is
  on-device firmware + dataflow, addressable only by fusion / faster kernels.**

## Sequencing & honest expectation
- Stage 1 is a safe, standalone win (delete the second DRAM scatter, `materialize`).
- Stages 2–3 are where the strategic value is, but they only **pay** once the
  blend/cull SFPU (the real 175 ms) is being restructured — a DRAM-free handoff
  matters because it lets those stages fuse / drop readers. **Do the blend plan's
  Track 1 (phasing) in parallel; land sort Stage 2–3 to feed it.**
- Stage 5 (trace) is the host-bottleneck lever — only worth it once Stages 1–4
  have removed the host glue between stages; it then collapses dispatch to ~0.
- Don't expect the literal 26 ms sort to vanish to 0 — expect it to shrink *and*,
  more importantly, to stop forcing DRAM on the stages that actually dominate.

## Guardrails
- PSNR gate ≥ 60 dB; `hero_vs_ref` must hold.
- Measure ms/view from device zones correctly (per-instance, not aggregate sum).
- Device kernels are an uncommitted rsync of Mac HEAD; restore from file backup.
- `build_lpt` / LPT balancing is already near-optimal (1.013) — reuse the
  *algorithm* and its published `sort_lpt_tile_ids` / `sort_lpt_meta`; do not
  re-tune the balance. For trace (Stage 5) port the same algorithm **on-device**
  (it's per-view), or accept the static-map fallback — but don't change what it
  optimizes.
