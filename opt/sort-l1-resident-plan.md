# Sort: L1-resident, DRAM-free plan

Status: PLANNED (not yet executed). Owner: supervisor.
Companion to `opt/blend-cull-speedup-plan.md`.

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
  - **P-domain deletion IMPLEMENTED + verified, KEEP DEFERRED (iter-124 WIP, branch
    `wip/s5.3-p-read-iter124` @ `dc7f4b5`, NOT on this branch):** deletes the
    tile_assign `P`-read by over-provisioning the pair buffers + K2 work-split to the
    static `P_max = pair_ceiling()` (4,718,592); K2 reads the clamped `P` from the
    resident `ta_pairs_P` ctrl page via `InterleavedAddrGen<true>`; `scan_bases` clamps
    the published `P` and publishes an overflow flag + true `P`; `sort_device`
    hard-fails on overflow (single-path TT, no fallback). All behind
    `host_free_mp_enabled()`. **Verified bit-identical on bh-07 pre-wedge** (devrun
    `--no-ref`, 30 views: hero md5 `e3fefb11…`; max P 3,700,450 < ceiling 4,718,592, no
    overflow; frame-neutral, min ms_view 213.0 thermally inflated). **The keep ceremony
    (30-view Tracy + on-device bit-identity gate) did NOT run:** bh-07 (the warm-tree
    device) wedged and IRD now sticky-assigns the cold `yyzo-bh-26` (empty `/localdev`,
    `--machine yyzo-bh-07` ignored across 4 reserve attempts) — bh-07 shows `idle/free`
    but is not allocatable (post-wedge quarantine). Re-verify + run the full keep flow
    once bh-07 is allocatable, then land on this branch as iter-124.
  - **STILL PENDING read-deletions (mid-frame host blocking-read drains):**
    1. `sort` `P`-read (`sort_device.cpp:~1240`) — redundant; frees once iter-124 lands.
    2. `gather` `M`-read (`gather_visible_device.cpp:~833`) — contract change
       (size `depths` to capacity, `M==0` → device no-op, read `M` post-frame for
       stats).
- **S5.4:** static/device subchunk alloc (5c) → no host `build_subchunk_layout`.
- **S5.5:** persistent per-core kernels looping the resident assignment (variable
  per-view iteration is device data → one trace replays for any view).
- **S5.6:** `BeginTraceCapture` the frame; per view = `EnqueueWriteBuffer(camera)`
  + `ReplayTrace` + one image D2H. This is the step that finally removes the 92 ms
  BRISC-FW per-program launch overhead (the only thing that can).

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
