# High-Utilization Resident Pipeline — Critique + Proposal

Response to the proposed "fastest algo." Written to be blunt, not agreeable. Your
structure is mostly right; the parts that are naive are called out explicitly,
with the real reasons we are at ~1000 ms while `cpu_cpp` is ~100 ms.

---

## 0. TL;DR verdict

- **Your pipeline shape is basically correct**: upload scene once, per-frame
  project→assign→sort→blend, everything resident, SFPU-centric. Keep it.
- **You are wrong that "everything is embarrassingly parallel except scatter."**
  The real enemies are (a) **FP math running as scalar soft-float on the
  data-mover RISCs** instead of the SFPU, (b) **host bridges + a full device
  barrier between every stage**, and (c) **per-tile load imbalance**. Scatter
  contention is the *least* of our problems and is already solved in-tree.
- **1 ms is below the compositing roofline for this scene** (~1.88 M visible
  gaussians, see §7). The honest near-term target is to *decisively beat 100 ms*
  (10–30 ms is reachable), then cut the pair/op count algorithmically.

---

## 1. Why we are at 1000 ms (the honest diagnosis)

Current per-stage (1 view, bicycle hero, all-resident gates on):
`proj=317  tile_assign=214  sort=65  blend=422` ms.

Root causes, in priority order:

1. **Soft-float FP on the data-mover RISCs.** The two NoC movers (NCRISC/BRISC)
   are scalar integer cores with **no FP hardware** — every `float` op compiles
   to a soft-float library call (tens of cycles each). Our microblock cull
   (`reader_alpha_blend_mb_devcull.cpp::compute_microblock_mask`) and the
   tile_assign AABB/Mahalanobis math run *there*. Profiling proved it: with the
   blend reader doing the cull, `NOBLEND` exec == full exec (411 ms) — i.e. the
   **SFPU compute kernel is 100% hidden and the bottleneck is entirely the
   scalar cull on one mover core**. This is the single biggest mistake in the
   current code.
2. **Host bridges + per-stage barriers.** Between stages we run *single-threaded
   host* prefix-sums, binning (sort Pass1/Pass2), and compaction, with a
   `distributed::Finish()` after each device launch. That serializes the whole
   pipeline and adds D2H/H2D. No stage overlaps any other → giant bubbles.
3. **Random-access gathers by id.** Blend gathers each gaussian's attrs by id
   from DRAM. Even after pipelining, it's scattered traffic competing with the
   cull for the same mover core.
4. **We pay for N then M.** Project runs over N=6.13 M, then we compact to
   M=1.88 M. Compaction is a scan; doing it on the host (or single-core) is slow.

None of these are fundamental to your design. They're "wrong tool for the unit"
and "host in the loop" bugs.

---

## 2. The hardware you must design against (1 screen)

Per **Tensix core** (~130 usable on Blackhole/P100):
- **2 data-mover RISCs** (NCRISC/BRISC): move bytes over NoC. **Integer/scalar
  only. Any float here = soft-float = death.** Use them for `noc_async_read/write`
  and address math, nothing else.
- **3 compute RISCs** (UNPACK/MATH/PACK). The MATH RISC drives:
  - **SFPU**: 32-lane FP32 vector unit. **One 32-lane vector == 4 rows × 8 cols
    of a 32×32 tile** (verified in the blend kernel). This is your FP workhorse:
    ~32 fp32 ops/instruction × ~130 cores ≈ **~4000-wide fp32 machine**.
  - **FPU / matrix engine**: 32×32 tile MAC. Great for matmul; **useless for
    per-splat 3×3 transforms** — don't try to force projection onto it.
- **L1**: ~1.5 MB per core. Big enough to hold a tile's working set; not big
  enough to be careless.
- **DRAM**: GDDR6, many banks, ~TB/s aggregate — but only if accesses are
  coalesced and spread across banks.

Design rule of thumb: **movers move, SFPU computes, host does nothing per
frame.** Every float on a mover or the host is a bug.

---

## 3. Point-by-point critique of your proposal

> **(1) host uploads scene to DRAM once per scene**

Correct and already done. No notes.

> **(2) project: split scene into L1/2 chunks; reader loads chunk, compute
> projects + frustum-culls, writes to L1**

Right shape. Two corrections:
- **Projection belongs on the SFPU, not the FPU.** It's per-splat mat-vec +
  2×2 covariance → vectorize *across splats* (SoA: 32 splats per SFPU op), not
  as 32×32 matmuls. The matrix engine will sit idle this whole pipeline; that's
  expected — GS is not a matmul workload.
- "writes to L1 (enough space since only L1/2 loaded)" — fine, but the
  **frustum-cull output is a stream compaction** (keep only survivors). Within a
  core's chunk that's a local scan — cheap. The mistake to avoid is doing the
  *global* compaction on the host (what we do now). Keep it per-core + a single
  exclusive-scan of per-core counts (the pattern we just used to get
  `gather_visible` 3730→317 ms, 11.8×).

> **(3) writer does tile assignment: scatter survivors (full data, not indices)
> into per-tile DRAM buckets. Worried about scatter contention → per-core tile
> buffers, then coalesce. Variable bucket sizes → chunked allocation. Careful not
> to serialize on size writes.**

This is the most-debated point. Several things:

- **You don't need per-core buffers or atomics.** The accepted, contention-free
  pattern is **count → exclusive-prefix-sum → scatter-to-exact-offsets**, and
  it's *already in-tree* (`tile_assign_bbox.cpp` K1 counts `tiles_per_gaussian`,
  host H1 prefix-sums, `tile_assign_scatter.cpp` K2 writes whole 64B-aligned
  pages by binary-searching the owning gaussian). Every core writes **disjoint,
  page-aligned ranges → zero write contention, zero atomics.** Your
  per-core-buffer+coalesce idea works but is strictly worse: it needs a second
  full-data pass to coalesce. The only thing wrong with our version is the
  prefix-sum is on the **host** — move it on-device and the worry evaporates.
- **Atomics-on-popular-tiles is the contention trap you correctly smelled.**
  Count-first sidesteps it entirely; never use NoC atomic-inc on per-tile
  counters — sky/background tiles would hotspot one bank.
- **"Full data, not indices" is bandwidth-naive — but your underlying goal
  (contiguous blend reads) is right.** Each gaussian overlaps *many* tiles
  (overlap factor ~4–20× for big splats). Scattering full attrs duplicates them
  per overlapped tile: with ~11 M (gaussian,tile) pairs × ~44–64 B ≈ **0.5–0.7
  GB of DRAM writes/frame**, then re-read in blend. That's the dominant traffic.
  vs. **(key=depth, gid)** pairs at 8 B ≈ ~90 MB.
  - **Recommended middle path:** scatter only `(depth_key:u32, gid:u32)` (8 B)
    for the sort. After sort, do **one** gather pass that writes the
    **depth-ordered, per-tile-contiguous blend payload** (conic a,b,c + mean
    x,y + opacity + rgb ≈ 36 B). Blend then streams that payload **contiguously,
    no gather**. You pay one gather (M_pairs, sequential-out) instead of
    duplicating full data through the scatter *and* the sort. This gives you the
    contiguous-blend win you want without the 0.7 GB scatter.
  - If profiling later shows the post-sort gather is the bottleneck, *then*
    promote to scattering the 36 B payload directly and sorting payloads. Decide
    with the bandwidth number, not by default.

> **(4) sort each tile bucket in L1; overflow → chunk + selection/merge sort**

- In-L1 per-tile sort: yes. **Never selection sort** (O(n²)); use the **per-tile
  LSD radix** already in `sort_radix_tile.cpp` (depth keys, byte passes). For
  buckets that overflow L1, segment + merge — not selection.
- **The real risk you under-weight is load imbalance, not overflow.** Tile
  populations are heavily skewed; one-tile-per-core leaves most cores idle
  waiting on a few monster tiles. **This is the dominant utilization killer in
  GS rasterization.** Mitigation: a **work queue / dynamic tile assignment**
  (cores pull the next tile when free) and **split monster tiles across cores**
  (sort segments, merge). Static "tile i → core i%N" will cap you well under
  50% util regardless of how good the kernels are.

> **(5) microblock assignment after sort: per-tile per-microblock index arrays
> (variable sized, chunked buffers)**

- This is the **cull**, and it **must be SFPU-vectorized** — this is exactly the
  357 ms soft-float disaster today. Compute the Mahalanobis/conic
  microblock-overlap test for a splat against the 8×8 microblock grid using SFPU
  lanes, not scalar divides on a mover.
- **Don't build per-microblock index arrays.** That's another variable-size
  compaction. Carry a **single 32-bit microblock mask per splat** (1 word; we
  already do) and let blend skip untouched microblocks by testing mask bits. No
  extra buffers, no extra scan. Cheaper and simpler than what you propose.

> **(6) blend: per tile bucket, mb-major over index arrays, read tile splat
> array from L1, accumulate in DST or L1, writer writes framebuffer**

- Right model, and it's what the compute kernel already does (R/G/B/T persistent
  in DST slots 0–3, X/Y ramps in 4–5, SFPU `exp` + composite per microblock).
- **Accumulate in DST, not L1.** SFPU writes DST directly; L1 accumulation adds
  round-trips. DST has 8 fp32 32×32 tiles — R/G/B/T + 2 ramps fits with margin.
- **Know the hard serial dependency:** alpha compositing is **sequential over
  depth per pixel** (front-to-back `T *= 1-α`), **parallel over pixels**. So per
  tile you iterate splats serially and do all 1024 pixels (32 microblocks) in
  parallel on SFPU. You cannot parallelize different splats into the same pixel.
  The cull mask is what keeps the per-splat work to the few microblocks it
  actually touches.

> **(7) "all embarrassingly parallel except scatter"; "most doable on SFPU, even
> FPU for projection"**

- **No.** Compositing is serial per pixel; sort is per-tile but *imbalanced*;
  prefix-sums/compactions are *scans* (not embarrassing); and stage handoffs are
  sync points. "Embarrassingly parallel" hides exactly the costs that bite.
- **SFPU yes, FPU no** (see §2). The matrix engine stays idle; that's fine.

> **(8) "1000 ms vs 100 ms is atrocious — you must be doing very wrong stuff"**

- Agreed it's bad, and yes we are — but the design isn't the problem. It's
  soft-float-on-movers + host-in-the-loop (§1). Used correctly the device is a
  ~4000-wide fp32 machine with ~TB/s; it *should* beat a 16-thread AVX2 CPU
  comfortably. The CPU's 100 ms is mature 8-wide×16-thread SIMD with zero
  marshaling; we're losing to it only because most of our FP currently runs on
  the one scalar unit per core that has no FP.

---

## 4. Proposed pipeline (stage by stage)

North star: **one resident program set, launched once; camera/uniforms in,
framebuffer out; no host compute and no full-device barrier inside the frame.**
All FP on SFPU. Movers only move. Scans/compactions on-device.

Shared resident state (DRAM, allocated once per scene; registered in
`device_state`): scene SoA (means, cov/quats, opacity, SH/rgb), and per-frame
intermediates reused across frames (grown on demand, never per-frame realloc).

### Stage A — Project + frustum cull → compact visible SoA
- **Input:** resident scene SoA (N), camera uniforms.
- **Output:** resident M-compact `proj_m_*` SoA (conic-ready: a,b,c, mean x,y,
  radius, opacity, rgb, depth).
- **Processing:** per-core chunk of N → reader streams chunk to L1 → SFPU
  projects + computes 2D cov/conic + depth + 3-sigma radius + visibility
  predicate across 32 splats/op → per-core local compaction → exclusive scan of
  per-core counts (device) → scatter to global M-compact offsets.
- **Kernels:** `reader` (chunk load), `compute` (SFPU project, **new SFPU
  kernel**), `writer` (compact scatter). Reuse the `gather_visible` count→scan→
  scatter we just proved.
- **Status:** project math exists; compaction is parallelized (317 ms). Next:
  move projection compute itself fully onto SFPU and fuse the predicate.

### Stage B — Tile assignment (binning) → resident pairs
- **Input:** `proj_m_*` (mean, radius, conic, opacity).
- **Output:** resident `(depth_key:u32, gid:u32)` pairs, plus per-tile
  `[start,end)` ranges.
- **Processing:** **K1 (SFPU):** per-gaussian AABB → `tiles_per_gaussian` (+
  optional cheap Mahalanobis per-tile reject). **Scan:** exclusive prefix-sum of
  `tiles_per_gaussian` *on-device* (Blelloch scan across cores). **K2 (mover):**
  pair-centric scatter to exact page-aligned offsets (already contention-free).
- **Kernels:** `tile_assign_bbox` (move AABB math to **SFPU**), on-device
  `prefix_scan` (**new, replaces host H1**), `tile_assign_scatter` (keep).
- **Fix vs today:** AABB/cull math is on the mover (soft-float) and the
  prefix-sum + compaction are on the host. Move both on-device. This is R4.

### Stage C — Per-tile depth sort → ordered pairs
- **Input:** resident pairs + tile ranges.
- **Output:** resident depth-sorted `gid` per tile + tile ranges
  (`sort_sorted_ids`, `sort_tile_ranges`).
- **Processing:** per-tile LSD radix in L1 (exists). **Load-balance:**
  dynamic/work-queue tile→core assignment; split monster tiles into segments +
  merge.
- **Kernels:** `sort_radix_tile` (keep), **+ on-device binning** (replace host
  Pass1/Pass2), + work-queue scheduler.
- **Fix vs today:** host binning + host compaction → on-device. This is R5.

### Stage C2 — (recommended) Gather depth-ordered payload
- **Input:** sorted `gid` per tile + `proj_m_*`.
- **Output:** per-tile **contiguous** blend payload in depth order (conic a,b,c,
  mean x,y, opacity, rgb ≈ 36 B/pair).
- **Processing:** one scatter/gather pass: read `proj_m_*` by sorted gid, write
  contiguous per-tile payload. **This removes the blend gather** and makes blend
  reads pure sequential streams.
- **Kernels:** `payload_gather` (**new**; mover-only, sequential out).
- **Tradeoff:** one M_pairs gather now vs. random gather inside blend forever.
  Pick with the bandwidth number; default to building the payload.

### Stage D — Microblock cull (mask) — **SFPU**
- **Input:** per-tile payload (or `proj_m_*` + sorted ids).
- **Output:** a **32-bit microblock mask per pair** appended to the payload row.
- **Processing:** SFPU computes the conic/Mahalanobis overlap of each splat
  against the 8×8 microblock grid; emits the 32-bit mask. **No index arrays.**
- **Kernels:** `microblock_cull` (**new SFPU kernel** — this is the 357 ms fix).
- This can be fused into Stage C2 (write payload + mask in one pass) or into the
  blend reader, but the math runs on **SFPU**, never a mover.

### Stage E — Alpha blend → framebuffer
- **Input:** per-tile depth-ordered payload + masks.
- **Output:** framebuffer tiles (RGB) in DRAM.
- **Processing:** mb-major; R/G/B/T persistent in DST; reader streams payload
  rows **contiguously** (no gather); SFPU composites masked microblocks; PACK
  writes tile RGB.
- **Kernels:** `reader_blend` (contiguous stream, **mask test only — no cull**),
  `alpha_blend_compute_mb` (keep — already SFPU and fully hidden), `writer`.
- **Fix vs today:** cull leaves the reader (→ Stage D, SFPU); reader becomes a
  pure contiguous streamer. With cull gone, the hidden SFPU compute becomes the
  bottleneck — which is the *correct* place to be bottlenecked.

### Framebuffer out
- Single D2H of the framebuffer at end of frame. The only per-frame host↔device
  traffic.

---

## 5. Contention & utilization risks you haven't named

| Risk | Why it bites | Mitigation |
|---|---|---|
| **Tile load imbalance** (dominant) | skewed tile populations → few monster tiles stall all cores | dynamic work-queue tile→core; split monster tiles + merge |
| **Soft-float on movers** | no FP hardware on NCRISC/BRISC | all FP on SFPU; movers do only NoC + int addr math |
| **Per-stage `Finish()` barriers** | global sync between stages → bubbles | persistent kernels / fused program; semaphore handoff, no host barrier |
| **Scan/compaction on host** | single-threaded, + D2H/H2D | on-device exclusive scan (count→scan→scatter) |
| **Blend random gather** | scattered DRAM, mover-bound | build per-tile contiguous payload once (Stage C2) |
| **DST register pressure** | only 8 fp32 tiles | R/G/B/T + 2 ramps fits; don't add more live tiles |
| **DRAM bank hotspotting** | popular tiles / atomics | count-first offsets (no atomics); interleave buckets across banks |
| **Mover↔SFPU imbalance within blend** | if reader can't feed SFPU | double-buffer payload reads (depth-2) so reads overlap composite |

---

## 6. What "near 100% ALU/BW utilization" actually requires

1. **Every FP op on SFPU**, vectorized across 32 elements (splats or pixels). No
   float touches a mover or the host.
2. **No host in the frame, no full-device barrier between stages.** Either a
   single fused program or persistent per-stage kernels handing off via
   semaphores/resident buffers.
3. **Load balancing** for the sort/blend (work queue + monster-tile splitting),
   or you cap at <50% util no matter how good the kernels are.
4. **Overlap movement with compute** (double-buffer L1) so SFPU never stalls on
   DRAM.
5. **Minimize DRAM round-trips:** keep each tile's working set in L1 through
   sort→cull→blend; only the framebuffer and the resident SoA hit DRAM.

---

## 7. Reality check on 1 ms (roofline)

Order-of-magnitude for this scene: M≈1.88 M visible, overlap ≈6 tiles/gaussian →
~11 M (gaussian,tile) pairs; say ~4 touched microblocks/pair; ~20 SFPU
vector-ops/microblock (conic, `exp`, composite). That's

`11e6 × 4 × 20 ≈ 9e8` SFPU vector-instructions for blend alone.

At ~130 cores × ~1 GHz × ~1 SFPU vec-op/cycle ≈ `1.3e11` vec-ops/s →
**~7 ms for blend's SFPU work at 100% utilization**, before any overhead.

**Conclusion:** for this scene, **1 ms is below the compositing roofline** with a
splat-by-splat approach. Realistic targets:
- **Beat `cpu_cpp` 100 ms decisively** — very achievable once cull is on SFPU and
  the host leaves the loop (kill the ~357 ms soft-float + the host scans).
- **~10–30 ms** total with the §4 pipeline at good utilization.
- **Sub-10 / toward 1 ms** requires *algorithmic* reduction of the pair/op count:
  tighter culling (fewer pairs), early-`T`-termination (stop a pixel once
  opaque), hierarchical/coarse-tile rejection, and lower-precision composite —
  not just better kernels.

So: 1 ms is the north star, but the next concrete milestone is **"SFPU cull + no
host in loop → beat 100 ms,"** then attack pair count.

---

## 8. Immediate next moves (ordered)

1. **Move microblock cull off the mover onto SFPU** (Stage D). Kills the dominant
   ~357 ms; turns the blend reader into a pure streamer. *Biggest single win.*
2. **On-device exclusive scan** to delete host H1 prefix-sum + host compaction
   (Stages B/C). Removes the host bridges (R4/R5, in flight).
3. **Build per-tile contiguous payload** (Stage C2) → blend reads sequential, no
   gather.
4. **Persistent kernels / drop per-stage `Finish()`** → remove cross-stage
   bubbles.
5. **Load-balancing** (work queue + monster-tile split) for sort/blend.
6. Only then chase 1 ms via algorithmic pair/op reduction (§7).
