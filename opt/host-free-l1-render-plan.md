# Host-Free, Work-Queue, L1-Resident Render Loop — Clean-Slate Plan

> Fresh design. Ignores prior plans/learnings on purpose. Goal: the best
> *theoretical* loop, then the shortest path to build it. Your outline is the
> skeleton; this keeps all of it and fixes the gaps that would otherwise make it
> incorrect or leave performance on the table.

---

## 0. North star & hard invariants

Render the *same* scene from a new view every frame, with the device doing
essentially all the work and DRAM touched only where it is physically
unavoidable.

**Per-frame DRAM budget (the contract):**

| # | Movement | Direction | Why it's unavoidable |
|---|----------|-----------|----------------------|
| 1 | view data (~128 B) | H2D | host chooses the camera |
| 2 | contiguous splat read, chunked | DRAM→L1 | scene lives in DRAM (uploaded once) |
| 3 | per-tile bucket scatter | L1→DRAM | producer core ≠ consumer core |
| 4 | per-tile bucket load | DRAM→L1 | same crossing, other side |
| 5 | framebuffer | D2H | host displays it |

Everything else — projection math, SH eval, cull, sort, microblock cull,
blend accumulation — happens **in L1** and never round-trips DRAM.

**One-time (frame 0 / scene load):** single H2D of all splats to DRAM.

Movements **3 and 4 move the same bytes** (write then read of the buckets), so
the dominant per-frame DRAM cost is `Σ_gaussians (tiles_touched × record_bytes)`.
That product — *write amplification* — is the number to attack, not the count of
"reads/writes." (§7)

---

## 1. Critique of the outline

Your skeleton is right and I keep 100% of it. What it is **missing or gets
subtly wrong**, in priority order:

1. **It isn't actually host-free yet.** "Per frame: project… blend… D2H" reads
   like the host re-dispatches programs each frame. Per-frame dispatch is the
   real host cost (program launch + a fistful of `Finish` drains). True
   host-free ⇒ **persistent kernels** that loop over frames and take the view
   from a mailbox the host pokes once. This is the single biggest lever and it's
   absent. (§3, §10)

2. **No global phase barrier named.** A tile cannot start blending until *every*
   chunk has finished scattering (any chunk may drop a splat into any tile).
   Phase 1 must fully complete before phase 2 begins → one device-wide barrier
   per frame. It's implied but must be designed, because it's also the seam
   where frame-pipelining lives. (§3, §11)

3. **Bucket offsets are undefined.** To scatter "full splat data into per-tile
   buckets" each producer needs *where* to write in each tile's bucket. That
   needs a per-tile base + a per-tile cursor. Doing this host-free ⇒ a count
   pass + on-device prefix-sum + per-tile **atomic** write cursors. Not in the
   outline. (§5)

4. **"Work queue" is in the title but not in the design.** Tiles are wildly
   uneven (mean ≈ 1 K splats, heaviest ≈ 26 K in our scenes). Static tile→core
   assignment makes one core decide the whole frame. The fix *is* the work
   queue: a global atomic "next tile" counter, tiles handed out heaviest-first
   (LPT). This is what makes it fast "quickly." (§8)

5. **Oversized tiles don't fit in L1.** "Load tile into L1, stay in L1" breaks
   for the hot tail: 26 K × 32 B ≈ 0.8 MB *just for records*, before sort
   scratch + masks + DST + code. Needs a capacity plan: tight records, a
   `BUCKET_FIT` cap with a streaming fallback, and ideally hot-tile splitting.
   (§9)

6. **No transmittance early-out.** Blend should stop visiting splats for a
   microblock once its 32 pixels are all saturated (T < ε). On dense tiles this
   is a large constant-factor win and it's free. (§6)

7. **SH→RGB must be pinned to projection.** Color is view-dependent (SH). Eval
   it once per gaussian in phase 1 and bake RGB into the scattered record;
   never in blend (blend would redo it per pixel). The outline says "writes
   non-culled ones to L1" — make "evaluate SH and pack RGB" explicit. (§5)

None of these make your plan *less* optimal — they're the parts that make it
*correct and complete*. The accepted DRAM costs (scatter + bucket load) are kept
as the baseline; §11 names the one move that beats them (L1→L1 NoC scatter) and
why we don't start there.

---

## 2. Hardware model (the constraints we design against)

- ~130 usable **Tensix** cores. Each: 2 data-movement RISCs (**reader**/NCRISC,
  **writer**/BRISC over NoC) + 3 compute RISCs (unpack/math/pack) driving the
  matrix **FPU** and vector **SFPU**.
- **~1.5 MB L1 per core.** This is the whole budget for code + CBs + tile
  records + sort scratch + masks + DST.
- **SFPU** processes ~32-element vectors. ⇒ **microblock = 4×8 = 32 px = one
  SFPU op.** Render **tile = 32×32 px** (matches the DST tile) ⇒ 32 microblocks
  ⇒ **32-bit coverage mask**. Everything lines up; this is why your numbers are
  the right numbers.
- **NoC** supports atomic increment and multicast semaphores ⇒ device-side work
  queues and barriers without the host.
- DRAM is large GDDR; bandwidth, not capacity, is the constraint.

---

## 3. Architecture: persistent kernels, two phases, one barrier

All cores run **one persistent kernel** that contains *both* phase bodies and
loops forever:

```
on each core (persistent):
  loop:
    wait(frame_ready)                 # host bumped the view mailbox
    load view from L1 mailbox
    # ---------- PHASE 1: project + tile-assign ----------
    while (c = atomic_next(chunk_queue)) < num_chunks:
        project_and_scatter(chunk c)  # read chunk -> L1 -> cull/SH -> scatter
    arrive_and_wait(phase_barrier)    # device-wide: all scatter done
    # ---------- PHASE 2: sort + cull + blend ----------
    while (t = atomic_next(tile_queue)) < num_tiles:   # heaviest-first
        render_tile(t)                # bucket -> L1 -> sort/cull/blend -> FB
    arrive_and_wait(frame_barrier)
    if core0: signal(frame_done)      # host may now read FB
```

Why this shape:

- **Persistent** ⇒ zero per-frame program dispatch, zero per-stage `Finish`
  drains. The host's per-frame job shrinks to "write view, bump semaphore, wait
  done, read FB."
- **All cores do both phases** (not partitioned) ⇒ no core idles through a phase
  it didn't get assigned. Cost: the kernel must hold both code paths in L1 —
  fine given the L1 budget and that we control the kernel-config ceiling.
- **One barrier** between phases is mandatory (the produce/consume hazard).
  Within a phase, work is fully parallel and self-balancing via the queue.

---

## 4. Per-frame dataflow

```
            host writes view (128B H2D) ──► [view mailbox in L1/DRAM] ──► bump frame_ready
                                                                               │
   DRAM: [ all splats, AoS, written once ]                                     ▼
                │  (2) contiguous chunk read                          PHASE 1 (chunk queue)
                ▼                                                      per core:
        reader → L1 chunk → compute: project + frustum cull + SH→RGB + AABB→tiles
                                       │
                                       ▼  (3) scattered write, atomic per-tile cursor
   DRAM: [ per-tile buckets: tile_base[t] .. tile_base[t]+count[t] ]
                                       │
            ===================  phase_barrier  ===================
                                       │
                ▲  (4) contiguous per-tile bucket read                PHASE 2 (tile queue, LPT)
                │                                                      per core:
        reader → L1 bucket → compute: L1 radix sort (depth) →
                              microblock cull (32-bit mask/splat) →
                              microblock-major blend into DST (early-out) →
        writer → (5) write 32×32 tile → DRAM framebuffer
                                       │
            ===================  frame_barrier  ===================
                                       │
                       core0 signals frame_done ──► host D2H framebuffer
```

---

## 5. Phase 1 — fused project + tile-assign (per chunk)

One program, gaussian-parallel, work-queued by chunk index.

Per chunk on a core:
1. **Reader** streams the chunk DRAM→L1, double-buffered (prefetch chunk *k+1*
   while compute runs on *k*) so DRAM latency is hidden.
2. **Compute**, per splat:
   - Transform mean to view/clip space; **frustum-cull** (behind camera /
     off-screen) → drop.
   - Build the 2D **conic** (projected inverse covariance) and the 3σ screen
     **AABB**; clip AABB to the tile grid → `[t_x0,t_x1]×[t_y0,t_y1]`.
   - **Evaluate SH → RGB** for this view direction (once, here).
   - Pack the compact **record** (§7): conic, 2D mean, depth-key, RGB, opacity.
3. **Scatter** the record into every covered tile's bucket.

**Bucket offsets, host-free (the missing piece):** scatter needs a slot in each
tile's bucket. Two viable mechanisms — pick (b) for simplicity/throughput:

- **(a) count → scan → place.** A first sub-pass histograms per-tile counts;
  on-device **exclusive prefix-sum** yields `tile_base[t]`; second sub-pass
  scatters using a per-tile atomic cursor seeded at `tile_base[t]`. Exact sizes,
  no overflow, two passes over the gaussians.
- **(b) pre-sized buckets + atomic cursor (recommended start).** Reserve
  `BUCKET_FIT` slots per tile (`tile_base[t] = t × BUCKET_FIT`). Scatter does
  `slot = atomic_inc(cursor[t])`; if `slot ≥ BUCKET_FIT`, route the overflow to
  a small spill list (§9). One pass, no global scan; trades DRAM footprint and a
  rare spill path for half the gaussian traffic.

Either way the **only** DRAM write here is the record scatter (movement 3).
Projected data never lands in DRAM in a separate buffer — it's produced in L1
and scattered straight out. (This is the project+TA fusion; it removes a whole
projected-buffer round trip.)

---

## 6. Phase 2 — fused sort + microblock-cull + blend (per tile)

One program, tile-parallel, work-queued by tile (heaviest-first, §8). Once the
bucket is in L1 the tile is **self-contained** — no more DRAM until the final
write.

Per tile on a core:
1. **Reader** bulk-loads the tile bucket DRAM→L1 (movement 4), one contiguous
   read of `count[t]` records.
2. **Sort (in L1):** stable **LSD radix sort** on a 32-bit depth key,
   front-to-back. Radix (not compare-sort) ⇒ deterministic order and no
   branch/compare bound. 32-bit key preserves ordering exactly enough to hold
   PSNR (the proven recipe).
3. **Microblock cull (in L1):** for each splat, test its conic/AABB against the
   32 microblocks; set a **32-bit mask** of which 4×8 blocks it actually covers.
   Store the mask alongside (or packed into) the L1 record. This is the work
   that lets blend skip.
4. **Blend (in L1), microblock-major:**
   ```
   for mb in 0..31:                     # 32 px = one SFPU vector, in DST
     T = 1; C = 0                        # transmittance, color accum
     for splat in sorted order:
       if (mask[splat] >> mb) & 1 == 0: continue   # not in this microblock
       a = opacity * exp(-0.5 * d^T Σ⁻¹ d)         # SFPU exp, per pixel
       C += T * a * rgb;  T *= (1 - a)
       if all_lanes(T) < ε: break        # ← transmittance early-out
     write C to DST[mb]
   ```
   - Microblock-major keeps the 32-px accumulator resident in DST across the
     whole splat list for that block — no streaming of partial results.
   - **Early-out** when the block saturates is a large constant-factor win on
     dense tiles and costs nothing.
5. **Writer** drains the finished 32×32 DST tile to the DRAM framebuffer
   (movement 5).

---

## 7. Record format & write amplification (the real cost)

Per-frame dominant DRAM = `Σ_gaussians tiles_touched × record_bytes`. Two
levers, both here:

**Shrink the record.** Target ≈ **32 B**:

| field | bytes | note |
|-------|-------|------|
| conic a,b,c | 6 | fp16 ×3 |
| mean x,y | 4 | fp16 ×2 (tile-local) |
| depth key | 4 | uint32 monotone key |
| rgb | 6 | fp16 ×3 (or 3 B fp8 if PSNR holds) |
| opacity | 2 | fp16 |
| pad/flags | ~10 | mask scratch, alignment |

Quantize aggressively but **gate every narrowing on the 63.85 dB bar** — color
to fp8 and depth-key width are the first things to A/B.

**Shrink tiles_touched.** Don't over-assign: clip the AABB tightly, and
optionally a cheap corner/edge test so a gaussian grazing a tile's margin (but
covering no microblock) isn't written there. Every avoided tile saves a full
record write *and* a later read.

This is also the knob that fixes §9: smaller records ⇒ more tiles fit in L1.

---

## 8. Load balancing — the work queue

Static assignment loses to the heaviest item. Use **dynamic** dispatch:

- **Chunk queue (phase 1):** `atomic_inc(chunk_cursor)`; chunks are uniform so
  this mainly cleans up the tail.
- **Tile queue (phase 2):** `atomic_inc(tile_cursor)` over tiles **sorted by
  population descending** (LPT — longest processing time first). Tile counts are
  already known from §5's cursors; a 1024-element sort of `(count,tile)` is
  trivial. LPT-greedy is within a small constant of optimal makespan and, being
  dynamic, is robust to estimate error.

Makespan model (why this is fast): phase 2 ≈ `Σ_tiles work(t) / cores` once no
single tile dominates — **except** the heaviest tile is a hard floor (one tile,
one core). That floor is exactly what §9's hot-tile splitting attacks.

---

## 9. L1 capacity & the hot tail

L1 ≈ 1.5 MB must hold: code + double-buffered read CB + the tile records + radix
scratch (≈ records) + masks + DST + ramps. Budget so the **common** tile is
fully resident:

- With 32 B records, `BUCKET_FIT = 8192` ⇒ 256 KB records + 256 KB radix scratch
  + 32 KB masks ≈ 0.55 MB — comfortable. Most tiles fit.
- **Oversized tiles (count > BUCKET_FIT):** three options, escalate as needed:
  1. **Spill + stream:** overflow records go to a spill list; the tile is
     processed as resident-prefix + a streamed merge pass. Correct, slower for
     the rare tile.
  2. **Hot-tile split:** partition a giant tile into vertical 32×k strips, blend
     each strip on a different core (each still microblock-major), then the
     writer composites — *no* re-sort needed if each strip reads the same sorted
     order. This directly lowers the makespan floor of §8.
  3. **Cap + accept:** clamp `BUCKET_FIT` and drop the deepest tail; only if
     PSNR proves it's invisible (unlikely at 63.85 dB target — measure).

Start with (1) for correctness; add (2) when the heaviest tile is the measured
bottleneck.

---

## 10. Host-free control plane

- **View mailbox:** a fixed L1 (or DRAM) address holding the 4×4 view matrix +
  camera intrinsics + frame index. Host writes it (movement 1) and bumps
  `frame_ready` (a semaphore / monotonic counter).
- **Persistent kernels** spin on `frame_ready` (or sleep on a semaphore), render,
  then `frame_done`. Host waits on `frame_done` and issues the framebuffer D2H.
- **Barriers** (`phase_barrier`, `frame_barrier`): device-wide via a multicast
  semaphore or an arrival-counter in L1 that every core increments then spins
  until it equals `num_cores`. No host involvement.
- The host loop is literally: `write_view → bump → wait_done → D2H`. That's the
  whole "host-free" claim, made real.

---

## 11. Optimization frontier (after the baseline runs green)

Ordered by leverage:

1. **Frame software-pipelining.** Double-buffer the tile buckets and overlap
   **phase 1 of frame N+1 with phase 2 of frame N** across the barrier. Hides a
   whole phase ⇒ steady-state throughput ≈ `max(T_phase1, T_phase2)` instead of
   their sum. Biggest throughput win once correctness holds.

2. **L1→L1 NoC scatter (beats movements 3+4).** Instead of scattering records to
   DRAM and reading them back, scatter **directly into the consumer core's L1
   bucket** over the NoC. Then the *only* per-frame DRAM is the framebuffer.
   Constraints that keep this from being the default: the consumer tile must be
   **statically** owned (breaks the dynamic tile queue) and must **fit in L1**
   (breaks on the hot tail). ⇒ **hybrid:** small/medium tiles go L1→L1 with a
   statically-balanced core subset; giant tiles use the DRAM bucket + work
   queue. This is the asymptotic optimum and why DRAM buckets are the *baseline*,
   not the ceiling.

3. **Per-microblock splat lists.** During microblock cull, build 32 short index
   lists so blend iterates only covering splats per block instead of testing the
   mask each splat. Trades a little L1 for fewer skipped iterations on sparse
   blocks.

4. **fp8 color / narrower keys.** Each narrowing shrinks the record ⇒ less
   write-amp ⇒ more tiles resident ⇒ faster phase 2. Gate on PSNR.

---

## 12. Build order (shortest path to green, then fast)

Each milestone ends at **63.85 dB** on the ideal path and a recorded ms/view.

- **M0 — record + buckets.** Define the 32 B record; phase 1 scatters into
  pre-sized buckets via atomic cursors (§5b); a trivial phase-2 that just reads a
  bucket back and blends *unsorted-but-correct* to prove the crossing. Gate PSNR.
- **M1 — L1 sort.** Add the L1 LSD radix; prove bit-order matches the reference.
- **M2 — microblock cull + microblock-major blend + early-out.** The §6 loop.
  This is where ms/view should drop hard (compute-bound, DRAM-quiet phase 2).
- **M3 — work queues.** Chunk queue + LPT tile queue. Kills the static-imbalance
  tail.
- **M4 — persistent kernels + mailbox + barriers.** Remove per-frame dispatch
  and `Finish` drains. This is the "host-free" milestone.
- **M5 — capacity hardening.** Spill/stream then hot-tile split for the tail.
- **M6 — frontier.** Frame pipelining (§11.1), then hybrid L1→L1 scatter
  (§11.2).

**Validation gates (every milestone):** PSNR ≥ 63.6 dB on the ideal
TILE_BUCKET path; per-stage timings from device-zone sums (no host-side
guesses); hero + 10× diff screenshot; ms/view recorded; report + commit.

---

## 13. Risks / open questions

- **Phase-2 floor = heaviest tile.** If hot-tile splitting (§9.2) is hard, the
  26 K-splat tile caps ms/view regardless of core count. Measure the tile
  histogram early; decide split vs. spill on data.
- **Record width vs PSNR.** 32 B assumes fp16 conic/rgb survives. If not, record
  grows and L1 budget tightens — re-derive `BUCKET_FIT`.
- **Persistent-kernel code size.** Both phase bodies + sort + SFPU blend in one
  kernel may press the kernel-config L1 ceiling; we control that ceiling, but
  watch worker-L1 left for buckets.
- **Atomic-cursor contention** on hot tiles during scatter. If it shows up,
  shard the cursor (per-core sub-cursors merged by base) or fall back to §5a.
- **Barrier cost** at ~130 cores should be negligible vs a phase, but verify it
  isn't a serialization point under frame pipelining.
