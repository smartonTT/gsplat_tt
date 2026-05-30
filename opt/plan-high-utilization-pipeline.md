# High-Utilization Resident Pipeline — Critique + Proposal

# User's proposal:
This is how I see the Gaussian Splat rendering being fastest:

host uploads scene to device dram only once per scene
then, per frame
  project processes the full scene, split into L1/2 sized chunks (or smaller for load balancing)
    chunk loaded from dram into L1 by reader
    compute projects and frustum culls each splat, writes to L1 (enough space since only L1/2 was loaded)
    writer (or another compute kernel?) does tile assignment: scatters the splats that passed into per-tile buckets in dram (full data, not indices). This scatter is the only thing I'm worried about that could stall due to contention (maybe there's an accepted solution to that? maybe have a per-core tile buffer in dram, so none of the cores conflict, then coalesce the buffers when done). Because tiles have variable sized numbers of splats, maybe we need to do some sort of chunk buffer allocation to get stuff decently localized in dram. We also have to be careful not to serialize on array size writes. I hope there are accepted writer patterns in the codebase to glean from.
  sort stage 
    sorts each bucket in L1 (if some buckets overflow L1, we'll need to chunk them up and do a selection sort or merge sort after)
    does microblock assignment (reading from L1 after last stage of sort) - sets up the per-bucket(tile) per-microblock index arrays (these are again variable sized so might use chunked buffers)
  alpha blend stage
    processes each tile bucket (or sub-tile if sort couldn't fit in L1) by looping mb major over mb idex arrays, reading from the tile splat array from L1, accumulating in dst or L1, whichever turns out to be faster
    writer writes to framebuffer
  frame done, framebuffer is copied from device to host

What am I missing? Where am I going wrong? All of these stages look embarrassingly parallel except maybe the tile assingment scatter. Most of this should be doable on sfpu, and maybe even FPU (ex: projection) for everying but the microblock blend. Am I missing other areas where contention could increase and utilization drop? Your current frame time of 1000ms vs cpu_cpp's 100ms is atrociousl. You must be doing very wrong stuff. Tell me which part of my proposal is naive and couldn't achieve near 100% alu/bandwidth utilization. 

Write me a clear md file plan responding to my proposal point by point and writing your own proposal for the ultimate high utilization pipeline with no stalls. Make each pipeline stage's input, output, processing and kernel breakdown clear but concise. Don't just agree with me because I'm the user, critique each point and come up with a better proposal.


---

#AI response

Response to the proposed "fastest algo." Written to be blunt, not agreeable. Your
structure is mostly right; the parts that are naive are called out explicitly,
with the real reasons we are at ~1000 ms while `cpu_cpp` is ~100 ms.

## 0. TL;DR verdict

- **Your pipeline shape is basically correct**: upload scene once, per-frame
  project→assign→sort→blend, everything resident, SFPU-centric. Keep it.
- **Your stages ARE essentially embarrassingly parallel** at their natural
  granularity (per-splat for project/cull, per-tile for sort/blend) — I was
  wrong to imply otherwise. The only genuinely non-embarrassing bits are (a) the
  **scans** (stream-compaction / prefix-sum) and (b) the **within-tile
  front-to-back composite** (serial in depth) — and (b) doesn't actually hurt
  because tiles ≫ cores, so you parallelize *across* tiles. See §0.5-Q1.
- **The real enemies are not your design.** They are (a) **FP math running as
  scalar soft-float on the data-mover RISCs** instead of the SFPU/FPU, (b)
  **host bridges + a full-device `Finish()` barrier between every stage**, and
  (c) **load balancing** (your LPT queue, §0.5-Q3, solves this). Scatter
  contention is the *least* of our problems and is already solved in-tree.
- **Honest correction: the pipeline is data-resident, NOT host-free.** I called
  it "fully on device," which was overstated — see §0.5-Q0. Intermediates stay
  in DRAM across stages (no per-frame re-upload), but the host still drives
  per-stage launches, the project compaction prefix-sum, and sort's residual
  D2H/compact/publish, with a `Finish()` between stages. Removing the host from
  the loop (persistent kernels, §0.5-Q3/Q9) is still open work.
- **1 ms is below the compositing roofline for this scene** (~1.88 M visible
  gaussians, see §7). The honest near-term target is to *decisively beat 100 ms*
  (10–30 ms is reachable), then cut the pair/op count algorithmically.

---

## 0.5 Round 2 — your follow-ups (quoted + answered)

These corrections **supersede** any conflicting claim in §1–§4 below.

### Q0 — "you didn't port everything to device with no host involvement, but pretended you did"

**You're right, and that's on me, not you.** "Fully resident on device" referred
to *data residency* (intermediates live in DRAM across stages; no per-frame
re-upload of attrs/ids — the ~127 MB/frame upload is gone). It did **not** mean
"no host in the loop," and I shouldn't have used milestone language that implied
it. What is *still* on the host today:
- per-stage program launches with a `distributed::Finish()` barrier between
  stages (no cross-stage overlap),
- the project compaction's exclusive prefix-sum of per-core counts,
- sort's residual `d2h + compact + publish` (~23 ms),
- all orchestration / control flow.

"No host in the render loop" requires **persistent kernels with on-device
scan + semaphore handoff** (Q3, Q9). That work is *not* done. I'll stop calling
the pipeline host-free until it actually is.

### Q1 — "I meant everything in MY proposal is embarrassingly parallel. Is it?"

**Mostly yes — I conflated "your proposal" with "the algorithm" and overreached.**
At the natural granularity each of your stages is embarrassingly parallel:

| stage | unit of independent work | embarrassingly parallel? |
|---|---|---|
| project + frustum cull | per **splat** | **yes** |
| tile assignment (geometry/AABB/Mahalanobis) | per **splat** | **yes** |
| …the scatter *offsets* | global scan | **no — it's a scan** (but cheap, count→prefix→scatter) |
| sort | per **tile** | **yes across tiles** (within a tile it's a local sort) |
| microblock cull | per **(splat,tile)** | **yes** |
| blend | per **tile** | **yes across tiles**; within a tile the depth composite is **serial** |

So the only things that aren't "do N independent items": the **scans**
(compaction / prefix-sum, inherent to turning variable per-item counts into
packed offsets) and the **within-tile front-to-back composite**. The composite's
seriality is a non-issue for utilization because there are **far more tiles than
cores** — you fill the machine across tiles. Your instinct was correct.

### Q2 — "Can't the FPU do more than matmul? Where can it do uniform full-tile work, one splat per element?"

**Yes. I was wrong to write "FPU = matmul, useless here."** On Tenstorrent the
matrix/FPU datapath also runs **full-tile (1024-element) elementwise binary ops**
— `mul_tiles` / `add_tiles` / `sub_tiles` with `acc_to_dest`, sourcing **CB×CB →
DST**. That is *precisely* your "uniform work on a 32×32 tile where each element
is a different piece of data." (Confirmed by the earlier `gs-fpu.md` sketch:
assemble the quadratic form with repeated `mul_tiles(..., acc_to_dest=true)`,
leaving only `exp` and DST-resident accumulation on the SFPU.)

**First-principles dividing line:**
- **FPU is for DENSE, UNIFORM elementwise (and true matmul/contraction).** Pack
  1024 independent items (e.g. **1024 splats, SoA, one per tile element**) and
  apply the same op to all → one FPU tile-op does 1024 lanes. Throughput per
  *instruction* is 32× the SFPU's 32 lanes.
- **SFPU is for transcendentals** (`exp`, reciprocal, `sqrt`, `log`),
  **data-dependent control**, and **DST×DST accumulation** (the running RGB/T
  update, where both operands are already in DST).
- **Movers** do only NoC + integer address math. Never floats.

**Where the FPU genuinely applies (per stage), one-splat-per-element packing:**
- **Project — means transform:** `W(3×3) @ means(3×N)` is a *real matmul* →
  `matmul_tiles`. High operand reuse (camera matrix shared across all splats).
- **Project — per-splat arithmetic:** `det = a·c − b·b`, conic `= cov/det`,
  depth, 3-σ radius, the frustum/visibility predicate — all elementwise across
  splats → **FPU `mul/sub/add_tiles`**, 1024 splats/op. Only the **`1/det`
  reciprocal** drops to SFPU.
- **Tile assignment — AABB + per-tile Mahalanobis reject:** the box math and the
  quadratic-form corner test are elementwise across splats → FPU, with SFPU only
  for any reciprocal. (This is exactly the soft-float-on-mover code today.)
- **Microblock cull:** the quadratic-form terms are elementwise; can use FPU for
  the products, SFPU for the lane→mask compare/reduce.

**Where the FPU does NOT help:** **blend** (we deliberately cull microblocks, so
the work is *sparse* — a dense full-tile FPU `mul_tiles` over all 1024 pixels
wastes ~the untouched fraction; you already concluded this), **sort** (integer,
data-dependent), **scatter** (pure data movement). So: **FPU for the dense
front half (project / assign / cull), SFPU for blend + transcendentals.** Net:
the matrix engine is *not* idle — I was wrong; it should carry the dense
per-splat arithmetic. (Caveat to verify on-device: the FPU eltwise path wants
operands from CBs as bf16/fp16 tiles for full rate; fp32 accuracy for `det`/conic
may force SFPU for a few terms. Measure precision vs the 39.4 dB gate.)

### Q3 — "Load balance = dispatch tiles in decreasing splat-count order + work queue; tiles ≫ cores; render many frames in parallel; cores advance to later stages before a stage finishes. Bad ideas?"

**Both are good — better than my "split monster tiles." I over-prescribed.**
- **Decreasing-count + work-queue is Longest-Processing-Time-first (LPT)**, the
  classic near-optimal greedy: with tiles ≫ cores it provably lands within 4/3
  of optimal and in practice ~stalls only on the *single* heaviest item. So make
  **LPT the primary** load-balancer.
- **Monster-tile splitting (chunking) is a *fallback*, only for a tile so heavy
  it alone exceeds ~total/cores** (or overflows L1). You correctly name its
  costs: extra traffic, a merge step, and — critically — **it breaks per-tile
  transmittance early-out** (a chunk can't know the global front-to-back `T`).
  So avoid it unless a tile is genuinely dominant. I've demoted it to last
  resort in §4-C.
- **Many frames in parallel:** great for **throughput / training** (fills DRAM +
  all cores). Note it does **not** reduce single-frame **latency** — so it helps
  the training north-star but not a 1 ms/frame *latency* target. Both valid; just
  don't conflate them.
- **Cores advancing to later stages before a stage globally finishes** = software
  pipelining with **no global barrier** = exactly the persistent-kernel /
  semaphore-handoff model (my §6.2, your cite of 321–323). This is the *right*
  way to kill the per-stage `Finish()` bubbles. No reason it's bad; it's the goal.

### Q4 — "Mask vs per-microblock index arrays: with a mask the inner loop iterates all splats and branches on dead microblocks. Isn't 32× the loop/branch overhead approaching the real blend cost?"

**Legitimate concern — but the 32-way unroll is forced by the hardware, and
index arrays wouldn't remove it.** The SFPU encodes the DST vector address as a
**compile-time immediate** (`SFPLOAD`/`SFPSTORE`), so the per-gaussian dispatch
over the 32 microblocks **must be a 32-way *unrolled* sequence with a templated
index** (see the kernel's own comment). The mask bit just gates each unrolled
slot: set → one SFPU blend; unset → a single scalar bit-test, **no SFPU work**.
- So the per-gaussian overhead is **32 scalar bit-tests**, not 32 wasted SFPU
  blends. Real work = `k` SFPU microblock-blends (k = microblocks this splat
  touches), each ~15–20 SFPU vector-ops. Overhead ratio ≈ `32 bit-tests /
  (k·~18 SFPU ops)`. For typical `k`≈2–6 that's small; for `k`=1 it's
  non-trivial — **so this is worth *measuring* (instrument avg k), not assuming.**
- **Index arrays don't help** because you'd *still* need compile-time microblock
  indices for the SFPU, i.e. still unroll; and they *add* a per-tile
  variable-size compaction (the very buffers you wanted to avoid).
- **The real lever** if `k` is small: bound the unroll to the splat's microblock
  **AABB sub-range** (the cull already computes `mx_lo..mx_hi, my_lo..my_hi`) via
  *templated range dispatch*, so you only emit bit-tests for the few microblocks
  in the bounding box, not all 32. That removes most dead bit-tests **without**
  index buffers. Good follow-up optimization; measure `k` first.

### Q5 — "We don't actually do 32×32 pixels iterating each splat in parallel. For each 4×8 we iterate its splats, then the next 4×8."

**Correcting *both* of us.** I mis-described it as "all 1024 pixels in parallel."
But the *current* kernel is **gaussian-major, not microblock-major**: it loops
gaussians once per tile and, per gaussian, updates the microblocks it touches;
**all 32 microblocks' R/G/B/T live simultaneously in DST** (slots 0–3, vector
index = microblock). One SFPU op = one microblock = 4×8 = 32 pixels. So:
- **Today (gaussian-major):** read each splat **once**, splat it into its `k`
  microblocks. Best splat-data reuse. **Cannot early-out on transmittance**,
  because a given depth's gaussian is applied across all its microblocks before
  the next depth — there's no per-microblock "am I saturated yet" gate in the
  loop structure.
- **Your model (microblock-major):** for each 4×8, walk its depth-sorted splats
  and composite until **`T` saturates, then stop**. Enables **early-out** (big
  win in dense/occluded tiles) at the cost of **re-reading splats per
  microblock** (up to 32× the payload reads if a splat spans many microblocks).

This is a real, open design axis, not a settled point. Early-out can dwarf the
re-read cost in occluded scenes; gaussian-major wins when overdraw is low. The
honest answer: **measure overdraw / saturation depth**; likely a hybrid
(gaussian-major within a microblock-group, with a saturation check) is best. I've
added this tradeoff to §4-E rather than asserting one structure.

### Q6 — "Why is projection not embarrassingly parallel?"

**It is. That was a flat error on my part.** Projection is per-splat independent
— maximally parallel, and (Q2) a prime FPU/matmul target. The non-parallel things
near projection are only the **stream-compaction scan** of survivors (turning a
per-splat keep/drop predicate into packed output offsets) — not the projection
math itself.

### Q7 — "317 ms for compaction is insanely bad. Is it already parallel or later?"

Clarification: **317 ms is the whole *project stage*, not the compaction alone.**
The `gather_visible` **compaction *is* already parallelized** (single-core →
full grid, 3730 → 317 ms, 11.8×). What remains inside the 317 ms is the
**projection compute itself** (transform + 2D-cov/`pfwc`) plus residual launch /
transfer overhead — and it has **not** been profiled or moved fully to FPU/SFPU
yet. So: compaction = done; the 317 ms is now dominated by un-optimized
projection compute and needs its own profiling pass (Q2 is the fix — FPU eltwise
+ matmul transform). Flagged in §4-A.

### Q8 — "Mahalanobis is mandatory, not optional."

**Agreed — corrected.** The per-tile (per-pair) Mahalanobis/conic reject is
**required**, not an optional add-on: an AABB-only assignment emits every tile the
bounding box touches, including the ellipse's empty corners → false-positive
pairs that inflate sort + blend. Dropping it is a correctness/perf regression.
§4-B now lists it as mandatory.

### Q9 — "Fuse C2 + D + E so the tile payload stays in L1 and never round-trips to DRAM?"

**Yes — this is the single most important structural change, and you're right it
was missing from the stage plan (I only mentioned it in §6).** Once a tile's
depth-ordered payload is built (C2), **microblock cull (D) and blend (E) only
read it** → keep it **resident in L1** for the whole tile and write **only the
final framebuffer tile** out to DRAM. That collapses three DRAM round-trips
(payload-out, mask-out/in, payload-in) into one L1-resident pass:

```
per tile (one persistent fused kernel, work-queue scheduled, LPT order):
  build depth-ordered payload in L1   (C2: gather proj_m_* by sorted id)
  cull → 32-bit mask per splat in L1  (D: SFPU/FPU, no DRAM)
  blend masked microblocks            (E: SFPU, R/G/B/T in DST)
  write framebuffer tile              (only DRAM write)
```

**Constraint:** the tile's payload (`S·~40 B` + masks) must fit in L1 (~1.5 MB →
~30 k splats/tile). Tiles that overflow fall back to the chunked path (Q3) —
which is also the only case that loses early-out. So **fusion is the default;
chunking is the L1-overflow/monster-tile exception.** §4 now presents C2/D/E as
one fused L1-resident stage.

---

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
  - **FPU / matrix engine**: 32×32 tile datapath. Does **matmul** (`matmul_tiles`)
    *and* **full-tile elementwise binary** (`mul/add/sub_tiles`, CB×CB→DST,
    `acc_to_dest`) — i.e. 1024 independent lanes/op. **Use it for dense uniform
    per-splat arithmetic** (pack 1024 splats/tile) and the shared-matrix means
    transform; see §0.5-Q2. *Not* for sparse blend or transcendentals. (Earlier I
    wrongly called it matmul-only/idle — corrected in §0.5-Q2.)
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
- **Processing:** per-core chunk of N → reader streams chunk to L1 → **FPU**
  means transform (`matmul_tiles`, shared camera matrix) + **FPU** elementwise
  per-splat arithmetic (det, conic, depth, 3-σ radius, predicate; 1024 splats/op)
  with **SFPU** only for `1/det` → per-core local compaction → exclusive scan of
  per-core counts (device) → scatter to global M-compact offsets (§0.5-Q2).
- **Kernels:** `reader` (chunk load), `compute` (**FPU eltwise + matmul, SFPU
  reciprocal** — new), `writer` (compact scatter). Reuse the `gather_visible`
  count→scan→scatter we just proved.
- **Status:** the **317 ms is the whole stage**, not compaction — compaction is
  already parallelized (3730→317 ms, 11.8×, §0.5-Q7). The residual is
  **un-profiled projection compute** still partly on the wrong units. Next:
  profile it, move the math to FPU/SFPU per Q2, and put the scan on-device.

### Stage B — Tile assignment (binning) → resident pairs
- **Input:** `proj_m_*` (mean, radius, conic, opacity).
- **Output:** resident `(depth_key:u32, gid:u32)` pairs, plus per-tile
  `[start,end)` ranges.
- **Processing:** **K1 (FPU/SFPU):** per-gaussian AABB → `tiles_per_gaussian`
  **with the per-tile Mahalanobis/conic reject — MANDATORY** (AABB-only emits the
  ellipse's empty corners → false-positive pairs that inflate sort+blend;
  §0.5-Q8). **Scan:** exclusive prefix-sum of `tiles_per_gaussian` *on-device*
  (Blelloch scan across cores). **K2 (mover):** pair-centric scatter to exact
  page-aligned offsets (already contention-free).
- **Kernels:** `tile_assign_bbox` (move AABB + Mahalanobis math to **FPU eltwise
  / SFPU**, off the soft-float mover), on-device `prefix_scan` (**new, replaces
  host H1**), `tile_assign_scatter` (keep).
- **Fix vs today:** AABB/cull math is on the mover (soft-float) and the
  prefix-sum + compaction are on the host. Move both on-device. This is R4.

### Stage C — Per-tile depth sort → ordered pairs
- **Input:** resident pairs + tile ranges.
- **Output:** resident depth-sorted `gid` per tile + tile ranges
  (`sort_sorted_ids`, `sort_tile_ranges`).
- **Processing:** per-tile LSD radix in L1 (exists). **Load-balance (primary):**
  **LPT** — feed tiles to a work queue in **decreasing splat-count order**, idle
  cores pull next (§0.5-Q3); near-optimal with tiles ≫ cores. **Fallback only:**
  split a tile that alone exceeds ~total/cores or overflows L1 (loses
  transmittance early-out — last resort).
- **Kernels:** `sort_radix_tile` (keep), **+ on-device binning** (replace host
  Pass1/Pass2), + **LPT work-queue scheduler** (shared across sort + the fused
  blend stage).
- **Fix vs today:** host binning + host compaction → on-device. This is R5.

### Stage C2+D+E — FUSED per-tile L1-resident: payload → cull → blend → framebuffer
**One persistent kernel per tile (LPT work-queue scheduled). Build the payload
once in L1; cull and blend read it in place; only the framebuffer tile leaves to
DRAM (§0.5-Q9).** This collapses three DRAM round-trips into one L1 pass.

```
per tile:
  C2  build depth-ordered payload in L1   gather proj_m_* by sorted id (mover)
  D   cull → 32-bit microblock mask/splat  FPU/SFPU, in L1, no DRAM
  E   blend masked microblocks             SFPU; R/G/B/T persistent in DST
  out write framebuffer tile               the ONLY DRAM write
```

- **Input:** `sort_sorted_ids` + ranges, `proj_m_*`. **Output:** framebuffer tile.
- **C2 payload** (≈36 B/pair: conic a,b,c, mean x,y, opacity, rgb): makes blend
  reads **sequential, no per-splat gather**. Built once per tile into L1.
- **D cull (the 357 ms fix):** mask computed on **SFPU/FPU**, never the
  soft-float mover. Carry a **32-bit mask/splat**, not index arrays (§0.5-Q4).
- **E blend:** SFPU composite; the 32-way microblock dispatch is unrolled (SFPU
  immediate addressing); mask bit gates each slot. **Open axis (§0.5-Q5):**
  gaussian-major (read splat once, no early-out) vs microblock-major (re-read,
  but **transmittance early-out**) — pick by measured overdraw; likely hybrid.
- **L1 budget:** payload must fit (~1.5 MB ⇒ ~30 k splats/tile). Overflow tiles
  take the chunked fallback (§0.5-Q3) — the only case that loses early-out.
- **Kernels:** one fused `tile_render` kernel (mover gather + FPU/SFPU cull +
  SFPU blend + pack), replacing today's separate reader-cull + compute + writer.
- **Fix vs today:** cull off the mover (→ SFPU/FPU), payload stays in L1 across
  cull+blend (no DRAM round-trips), and the hidden SFPU blend becomes the
  bottleneck — the *correct* place to be bottlenecked.

> **Incremental path to the fused kernel** (don't rewrite blind): the in-flight
> step is just "move the cull from the mover to the SFPU" inside the *existing*
> blend program. Land that (blend 422 → ~110 ms already proven, correctness WIP),
> then fuse C2 into L1, then drop the per-stage `Finish()` for persistent
> handoff.

### Framebuffer out
- Single D2H of the framebuffer at end of frame. The only per-frame host↔device
  traffic.

---

## 5. Contention & utilization risks you haven't named

| Risk | Why it bites | Mitigation |
|---|---|---|
| **Tile load imbalance** (dominant) | skewed tile populations → a few heavy tiles stall all cores | **LPT work-queue** (decreasing splat-count, idle cores pull); chunk only a single dominant/L1-overflow tile (§0.5-Q3) |
| **Soft-float on movers** | no FP hardware on NCRISC/BRISC | FP on **SFPU + FPU eltwise/matmul** (§0.5-Q2); movers do only NoC + int addr math |
| **Per-stage `Finish()` barriers** | global sync between stages → bubbles | persistent kernels / fused program; semaphore handoff, no host barrier |
| **Scan/compaction on host** | single-threaded, + D2H/H2D | on-device exclusive scan (count→scan→scatter) |
| **Blend random gather** | scattered DRAM, mover-bound | build per-tile contiguous payload once (Stage C2) |
| **DST register pressure** | only 8 fp32 tiles | R/G/B/T + 2 ramps fits; don't add more live tiles |
| **DRAM bank hotspotting** | popular tiles / atomics | count-first offsets (no atomics); interleave buckets across banks |
| **Mover↔SFPU imbalance within blend** | if reader can't feed SFPU | double-buffer payload reads (depth-2) so reads overlap composite |

---

## 6. What "near 100% ALU/BW utilization" actually requires

1. **Every FP op on a vector unit** — **FPU eltwise/matmul** for dense uniform
   per-splat work (1024/op), **SFPU** for transcendentals + DST accumulation +
   blend (§0.5-Q2). No float touches a mover or the host.
2. **No host in the frame, no full-device barrier between stages.** Either a
   single fused program or persistent per-stage kernels handing off via
   semaphores/resident buffers. (Not done yet — §0.5-Q0.)
3. **LPT load balancing** for sort/blend (decreasing-count work queue; chunk only
   a dominant/overflow tile — §0.5-Q3), or you cap at <50% util regardless of
   kernel quality.
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
4. **Fuse C2+D+E into one L1-resident per-tile kernel** (§0.5-Q9) → payload
   built once in L1, cull+blend read it in place, only the framebuffer leaves.
5. **Persistent kernels / drop per-stage `Finish()`** → remove cross-stage
   bubbles and the host from the loop (§0.5-Q0).
6. **LPT load-balancing** (decreasing-count work queue; chunk only dominant/
   overflow tiles — §0.5-Q3) for sort + fused blend.
7. **Move the dense front half to the FPU** (project means transform via
   `matmul_tiles`; det/conic/AABB/Mahalanobis via FPU eltwise; §0.5-Q2) and
   profile the 317 ms project stage (§0.5-Q7).
8. Only then chase 1 ms via algorithmic pair/op reduction — tighter cull,
   **transmittance early-out** (needs microblock-major, §0.5-Q5), coarse-tile
   rejection, lower-precision composite (§7).
