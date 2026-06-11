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
- Frame total ≈ **192 ms/view** (`iter 111`). The dominant cost is **cull + blend
  SFPU (~80 + ~95 = ~175 ms) run serially on the shared SFPU cores** — fusion is
  blocked by the iter-26 DEST hazard.

So: **sort is not the frame killer.** The user's architectural critique is still
correct (sort *is* all DRAM shuffling on the data movers, and it does *not* honor
the L1-resident / tile-per-core ideal), but the prize for fixing it is **not the
26 ms** — it is enabling a **DRAM-free sort → cull → blend handoff** that lets the
SFPU stages drop their DRAM readers and (eventually) fuse.

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
