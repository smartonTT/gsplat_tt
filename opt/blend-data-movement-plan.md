# Removing the data-movement bottleneck from fused blend — concrete plan

**Status:** make-or-break. Rewritten 2026-06-01 after the user (correctly) pointed out
that the whole "gather" is an artifact of a CPU-style port, not a necessity. The right
design — stated in `opt/plan-high-utilization-pipeline.md` lines 4–18 — is **scatter
full splat data into per-tile DRAM buckets at tile-assign, then load each tile once,
contiguously, into L1 and do ALL of sort/cull/blend in L1. There is no gather, ever.**
Grounded in `opt/host-free-fusion.md` §1 (the read breakdown that quantifies how bad
the current gather is) and the iter 16–21 results.

---

## 0. TL;DR — the correct design and why we're not on it

- **The blend "gather" should not exist.** Today `tile_assign_scatter` (K2) writes
  `(gid, tid)` **index pairs**, and the blend reader **gathers** each candidate's
  attributes by gaussian-id from the SoA `proj_m_*` arrays → **~35 M random NoC
  reads/frame, 2.3 GB moved to use ~115 MB** on one serial mover. This is a direct copy
  of the *CPU reference's* access pattern (fine on CPU — caches absorb it; fatal on TT —
  no cache, no coalescer; see §2).
- **The fix (the user's pipeline):** tile-assign scatters the **full AoS splat record**
  into per-tile **contiguous** DRAM buckets. Each tile is then read in **ONE contiguous
  DRAM→L1 transfer** and **sort + microblock-cull + blend all run in L1**. Sort only
  permutes **indices** (random L1 access is free); the records never move again. Blend
  is mb-major with a simple transmittance early-out; double-buffer the next tile's load
  behind the current tile's compute.
- **Cost arithmetic.** Duplication is only **~1.79× per gaussian** (P=3.37 M pairs /
  M=1.88 M gaussians, `opt/ta-stage-analysis.md`). At ~64 B/record that's **~216 MB
  written + ~216 MB read ≈ ~430 MB, all contiguous/bulk**, vs **2.3 GB random** today.
  Contiguous bulk runs near peak GDDR6 bandwidth; random runs at single-digit %. That is
  the order-of-magnitude, and it is the access pattern TT is **best** at.
- **L1 budget.** Avg tile ≈ 3.37 M/1024 ≈ **3.3 k candidates ≈ ~210 KB**, vs **L1
  ~1.5 MB** → fits ~7× over. Only a few "monster" tiles need the chunked fallback.
- **What's already done that helps:** iter 21 (S1) made the per-splat **AoS record**
  (`proj_m_blendrec`) — that is exactly the record tile-assign should *scatter*. It was
  applied at the wrong stage (it made each *gather* read contiguous instead of removing
  the gather), which is why it only saved ~6 ms and proved blend is latency/spin-bound,
  not transaction-bound. The record layout is reusable; the gather is what must go.
- **Obsoleted by this design (do NOT pursue):** sort-emits-payload (old S2), per-stage
  pack passes (old B1), **gaussian-major (old B3)** — pointless when every splat is
  already in L1 — and **multi-mover for the gather (old B4)** — there is no per-candidate
  gather to saturate. The `MB_CULL_SPIN` is a DRAM read-completion artifact that
  vanishes when blend reads from L1.

---

## 1. The bottleneck, precisely (the numbers — why the current gather is fatal)

Per kept candidate (`reader_alpha_blend_mb_devcull.cpp` `issue_chunk_reads`, lines
113–141; see `host-free-fusion.md §1.1`):

| reads | layout | NoC txns | bytes fetched | bytes used | efficiency |
|---|---|---:|---:|---:|---:|
| cov_a,cov_b,cov_c,mean_x,mean_y,opacity,colors(rgb) | **7 separate SoA arrays** | 7–8 | ~480–576 | 36 | ~6–7% |

- **3.2 M candidates/frame** → **~29 M gather NoC reads** (+~6.4 M mask reads) =
  **~35 M tiny transactions**, **~270 k per core**, all serialized on **one mover
  RISC** (NCRISC). The MATH RISC (SFPU) sits idle — `NOBLEND` exec == full exec.
- **2.3 GB DRAM read / frame to deliver ~115 MB useful** (3.2 M × 36 B).
- Root causes, in order of damage:
  1. **The gather exists at all** — blend reads attrs from DRAM per candidate instead of
     from a tile-local L1 buffer.
  2. **SoA layout** → each field a separate random page, 16× over-fetch (iter 21's AoS
     record fixed this part, but only matters once there's still a DRAM read at all).
  3. **One serial mover** → latency cannot be hidden; barrier + `MB_CULL_SPIN` dominate.

---

## 2. Why TT looks "so much worse" than GPU/CPU here (the honest hardware answer)

It is the **same** access pattern on three very different memory systems:

| mechanism | CPU | GPU | Tenstorrent |
|---|---|---|---|
| transparent cache | L1/L2/L3 = **MBs**, absorbs random reuse | L2 = **MBs** shared, + L1/tex | **none** — L1 is an *explicit scratchpad*, DRAM is never auto-cached |
| gather/coalescing HW | OoO load/store + line fill | **warp coalescer** merges 32 scattered lanes into cache lines | **none** — the mover issues each NoC packet explicitly |
| latency hiding | deep OoO + HW prefetch | **massive thread oversubscription** | a few RISC movers; you must *manually* issue many async reads |
| native efficient unit | cache line (64 B) | coalesced 32–128 B | **bulk contiguous tile DMA** (kilobytes) |

A random scalar gather: the CPU hides it in cache + OoO; the GPU coalesces + hides it
under thousands of threads + a big L2. **TT has neither a cache nor a coalescer**, and
its sweet spot is the *opposite* (huge contiguous DMA + an explicit L1 scratchpad). The
exact code that costs ~190 ms on TT would also hammer GPU/CPU DRAM — they just hide it.
The 2.3 GB / 115 MB = 20× over-fetch is invisible on GPU/CPU and fatal on TT.

**Corollary:** the moment we (a) move bytes as **bulk contiguous transfers** and (b)
keep the per-tile working set **resident in L1**, TT's GDDR6 bandwidth is
competitive-to-better and the scratchpad is an *advantage* (no cache pollution, fully
deterministic). We are not fighting the hardware; we are fighting a CPU-port access
pattern.

---

## 3. The correct pipeline (THE plan) — scatter full data, process in L1, never gather

Per the original design (`plan-high-utilization-pipeline.md` 4–18). Per frame:

```
PROJECT + TILE-ASSIGN  (gaussian-parallel, FUSED L1-resident per chunk — NO DRAM between them)
  reader loads a scene chunk DRAM→L1 (the only input read; scene was uploaded once)
  compute projects + frustum-culls + compacts → projected AoS records stay IN L1:
    blendrec[g] = {a,b,c,px,py,opacity,r,g,b,depth}   (iter 21 already produces this record)
  tile-assign reads those projected records FROM L1 (not DRAM) and scatters them:
    count pass: tiles_per_gaussian + prefix-sum  → per-tile bucket offsets   (already exists)
    scatter (K2, REWRITTEN): for each (gaussian,tile) AABB pair, write the FULL blendrec
      into tile_bucket[tid] in ARBITRARY (gaussian/scan) order — full data, NOT (gid,tid),
      and NOT depth-sorted. SEQUENTIAL read of the projected records + scatter-WRITES; the
      ONLY DRAM write of this stage. Per-core DRAM staging buffers then coalesce, OR
      atomic-bump per-tile cursors. Duplication ≈1.79×; writes are fire-and-forget.

  >>> THE CRUX: scatter happens BEFORE the depth sort, in arbitrary order, so it needs NO
      random read. The depth sort then runs IN L1 in the per-tile pass below (index permute,
      records stay put — free). The current code does the OPPOSITE: it sorts FIRST (in DRAM),
      so any per-tile bucket must be built by reading record[sorted_ids[k]] = a RANDOM read.
      That sorted-id gather is the trap (and what the abandoned TILE_REC scaffold did). Sort
      in L1 ⇒ zero random DRAM access. Sort in DRAM ⇒ unavoidable random gather.

  >>> the single, unavoidable DRAM round-trip of the bulk records is HERE: the scatter
      WRITE above (end of the gaussian-parallel side) + the per-tile READ below (start of
      the tile-parallel side). It exists ONLY because the producer is gaussian-parallel and
      the consumer is tile-parallel, so the records must cross DRAM to be regrouped by tile.
      Nothing else round-trips: project→tile-assign is all in L1.

PER-TILE PASS (tile-parallel, fully L1-resident — NO gather)
  for each tile (double-buffered: load tile N+1 while computing tile N):
    1 contiguous DRAM→L1 read of the whole tile bucket   (~210 KB avg; bulk, near-peak BW)
    SORT in L1: produce a depth-sorted index permutation (records stay put; indices only)
    MICROBLOCK CULL in L1: per-candidate mask, no DRAM
    BLEND in L1: mb-major over sorted indices, read records from L1 (free random access),
      accumulate; simple transmittance early-out
    write ONLY the framebuffer tile back to DRAM
```

**DRAM traffic per frame:** the scene-chunk input read (once) + ~216 MB contiguous
scatter-WRITE + ~216 MB contiguous per-tile READ + the framebuffer (~a few MB). The only
bulk-record DRAM round-trip is the scatter-write/per-tile-read pair (the axis change);
project→tile-assign stays in L1. Everything is bulk and contiguous. **Zero random gather.
Zero per-candidate DRAM read. No `cull_masks` round-trip, no `MB_CULL_SPIN`** (the mask is
computed and consumed entirely in L1).

> **NOTE — the current code violates this twice.** (1) Today `project_device.cpp` writes
> the `proj_m_*` buffers to DRAM, `gather_visible` compacts via DRAM, then `tile_assign`
> reads `proj_m_*` back FROM DRAM — an extra project→DRAM→tile-assign round-trip the ideal
> design removes by **fusing project+compact+tile-assign L1-resident per chunk** (a separate
> lever, "T0", on the ~58 ms proj/ta side — lower priority than killing the 193 ms blend
> gather). (2) tile-assign scatters `(gid,tid)` indices, forcing the blend gather. The
> in-flight worker fixes (2) first.

### Why each worry is a non-issue
- **"Duplicating splats wastes DRAM/BW."** Only ~1.79× here, contiguous, and writes are
  not latency-critical. ~430 MB total bulk beats 2.3 GB random by ~5× in bytes and far
  more in *effective* bandwidth.
- **"Scatter contention."** Per-core DRAM staging buffers (no cross-core conflict) then a
  coalesce pass, or per-tile atomic cursors. The count/prefix-sum that sizes the buckets
  already exists (the ta scan kernels).
- **"Compute stalls waiting for the whole-tile load (sort needs all splats)."**
  Double-buffer: prefetch tile N+1's contiguous load while sorting/blending tile N. One
  extra L1 buffer; trivial.
- **"Fuse it all into one kernel?"** Keep the **data** L1-resident across sort/cull/blend,
  but they may be **separate small programs** that hand off via the resident L1 buffer.
  Do **not** make one mega-kernel — iter 14 proved a single fused cull+blend program
  blows the **70656 B kernel-config ceiling** (it was 92688 B). L1 data residency is
  independent of code size.
- **"Monster tiles overflow L1."** A few high-overlap tiles may exceed ~1.5 MB; chunk
  just those (load + sort-merge in passes). Common tiles take the one-shot path.

---

## 3b. Spin diagnostic + the DECISIVE finding: the spin settles the GATHER, not the masks (2026-06-01, this worker)

**Diagnostic 1 (device, hero, 1 view, `GSPLAT_TT_CULL_SPIN=0`):** blend **198.6→109.7 ms**
(the per-candidate settle costs **~89 ms**) but PSNR **63.85→37.73 dB** (gate FAIL).
Initially read as a `cull_masks` read-completion settle. **This interpretation was wrong.**

**Step A attempt (gated `GSPLAT_TT_TILE_L1=1`, reader+host, default OFF) — bulk-load each
tile's whole `cull_masks` region DRAM→L1 once/tile, read masks from L1, drop the spin:**

| variant | blend ms | hero_vs_ref |
|---|---:|---:|
| baseline (per-chunk masks + spin=512) | ~193–198 | **63.85** ✅ |
| TILE_L1, no settle | 110.2 | 37.13 ❌ |
| TILE_L1, post-read settle 4096 | 110.2 | 39.19 ❌ |
| TILE_L1, pre-read settle 4096 | 109.8 | 38.99 ❌ |
| TILE_L1, pre-read settle 65536 | 113.4 | 38.97 ❌ |
| TILE_L1, pre-read settle 262144 | 123.9 | 38.84 ❌ |
| TILE_L1 + **`Finish` between cull & blend** (DRAM fully settled), no spin | 109.4 | 38.76 ❌ |

PSNR is **flat at ~38 dB across a 64× settle sweep AND with a hard `Finish`** that
guarantees every cull DRAM write is committed before blend reads. So it is categorically
**not** a settle/visibility problem.

**Diagnostic 2 (DEVICE-PROVEN, the smoking gun):** with `TILE_L1` on, for the first 3000
candidates/core, read each mask BOTH from the L1 bulk buffer AND with a fresh direct DRAM
read and compare. **Every one of ~110 cores reported `dn=3000 mm=0 ok=3000` — ZERO
mismatches.** The L1-resident masks are bit-identical to DRAM. **The masks are 100%
correct, yet PSNR is still 37.5 dB.**

**Conclusion — the `MB_CULL_SPIN` was never about masks. It settles the per-candidate
ATTRIBUTE GATHER** (`issue_chunk_reads`/`issue_chunk_reads_aos` → the scattered
`proj_m_*` / `proj_m_blendrec[g]` reads). Removing the spin lets the candidate loop
consume **stale attribute bytes** from the gather scratch (the `noc_async_read_barrier`
returns before the scattered reads physically land) → wrong cov/mean/opacity/color →
~38 dB. The masks were a red herring; **Step A (bulk masks) can never recover the 89 ms
because the cost lives in the attribute gather, not the masks.**

**This VALIDATES Step B / the §3 architecture and tells us exactly why it works:** the
same bulk contiguous L1 load that made the masks correct did so with **a single barrier
and NO spin** (mm=0). Bulk contiguous reads settle reliably; only **scattered
per-candidate gathers** need the spin. Therefore loading a **per-tile contiguous record
bucket** once into L1 (one bulk read + one barrier) removes the gather AND the spin in
one move — no settle needed, as proven by the mask bulk-load.

Step A code is **left in-tree but gated OFF** (`GSPLAT_TT_TILE_L1`, default off; does not
hold the gate, banks no win). Proceeding to Step B (the real fix).

**Step B (true bucket, kills the gather): per-tile contiguous full-record bucket written
inline during the sort bin/scatter (sequential `proj_m_blendrec[g]` reads) + a `slot`
permutation carried through the radix; blend bulk-loads the tile bucket into L1 (one
barrier, no spin — proven safe by Diagnostic 2) and reads records by sorted slot from
L1.** Monster tiles (L×64B>L1, hero max ≈1.7 MB) chunked or fall back to the batched
gather. Removes the ~216 MB random attr gather and the ~89 ms spin together.

## 3c. CORRECTION to §3b + the chosen implementation route (2026-06-01, this worker, post-TILE_REC)

**TILE_REC (the abandoned scaffold) was a relocation, not a fix — confirmed and reverted.**
The cull reader built `tile_recs` by reading `proj_m_blendrec[g]` with `g` taken from the
DEPTH-SORTED order (`sort_sorted_ids`) → a random read. It moved the 3.2 M random reads
from blend into cull; the random DRAM gather still existed. Measured (gated, hero, 1 view):
gate held bit-identical (**63.85 dB**, records in L1 are correct) but blend stayed **199 ms**
— no win, because the relocation doesn't remove the random traffic and the spin persists.

**CORRECTION to §3b:** the spin is **NOT** settling the attribute gather — it settles the
per-chunk **`cull_masks` DRAM read**. Proof: with TILE_REC the attribute gather is gone
(records come from L1), yet `CULL_SPIN` still controls PSNR monotonically — `512→63.85`,
`128→36.0`, `64→34.8`, `0→39.99 dB`; and bulk-L1 masks (TILE_L1) plateau at ~40 dB even
with a `Finish`. So a single bulk read of freshly-written `cull_masks` will not settle; only
the repeated per-chunk re-read + per-candidate spin settles them. **Consequence:** the spin
cannot be killed by relocating records or by bulk-loading masks across the cull→blend
program boundary. It only dies when the **mask is produced and consumed inside the SAME
L1-resident per-tile pass and never crosses the NoC** (T3). TILE_L1 is therefore NOT an
independent win (does not hold the gate); not banked.

**Chosen route — piggyback the scatter on the existing sort BIN (a big simplification).**
`sort_bin.cpp` ALREADY implements the user's "per-tile histogram + prefix-sum + atomic-free
coalesce": mode 0 builds per-core per-tile histograms; the host computes page-aligned
per-tile starts + each core's per-(core,tile) page-aligned global base (`hist[c*stride+t]`);
mode 1 scatters each core's kept pairs into per-tile contiguous blocks via whole-page,
exclusively-owned DRAM writes (no atomics, no shared-page race). The depth **radix runs
AFTER** the bin. The bin reads pairs in **gaussian-major order (g monotonic per core)**, so
reading `proj_m_blendrec[g]` there is a **sequential/page-cached read**, and the per-tile
block placement is the scatter-WRITE. This is exactly scatter-before-(depth-)sort.

Mapping: keys/ids element `e` ↔ record bucket **page `e`**. For a kept pair placed at local
`curp[t]` in core `c`'s tile-`t` block, its global element is `rowp[t] + curp[t]`
(`rowp[t] == hist[c*stride+t]`), so its record goes to `tile_recs[rowp[t] + curp[t]]` — a
single 64 B page write, unique per (core,tile,local), disjoint across cores → **no L1 record
staging (avoids `BIN_LOCAL_MAX×64B`), no atomics, no race**. The bucket is in bin/gaussian
order (NOT depth-sorted); the depth sort moves into the per-tile L1 pass, which is the whole
point. The DRAM **radix is then removed** on the bucket path (T2/T3).

## 4. Concrete implementation steps (ordered, each independently measurable / gated ≥63.6 dB)

1. **T1 — Scatter full records in the BIN (gated `GSPLAT_TT_TILE_BUCKET`).** Extend
   `sort_bin.cpp` mode 1: alongside the (key,id) block write, read `proj_m_blendrec[g]`
   (page-cached, gaussian-major) + assemble a 10-field record `{a,b,c,px,py,op,r,g,b,depth}`
   (depth = the `key` already read) and write it (64 B) to `tile_recs[rowp[t]+curp[t]]`.
   Host `sort_device.cpp`: allocate `sort_tile_recs` (P_aligned × 64 B), register it, pass
   `blendrec_addr` + `tile_recs_addr` as 2 new bin args/accessors under the gate. Default
   pipeline untouched (bucket produced, unused). The per-tile region is `[pstart_elem[t],
   pstart_elem[t]+tile_pad[t])` in PAGE units == `sort_tile_ranges` padded region. Verify
   gate stays 63.85 (bucket is write-only here); spot-check a few records vs `blendrec`.
2. **T2 — Per-tile L1 load + L1 depth sort (gated).** New per-tile reader: one bulk
   `noc_async_read` of `tile_recs[pstart..pstart+pad_n]` into L1; produce a depth-ordered
   index permutation in L1 from the records' depth field (records stay put). Replaces the
   DRAM radix + `sort_sorted_ids` gather for the bucket consumer; the DRAM radix is skipped
   on this path. Verify max padded tile ≤ L1 budget (hero max_n≈26.5k×64B≈1.7 MB → chunk
   fallback for monster tiles).
3. **T3 — L1-resident cull + blend.** Microblock cull writes its mask to **L1** (not
   `cull_masks` DRAM); blend reads records + mask from L1 by sorted index, mb-major,
   transmittance early-out, writes only the framebuffer tile. Delete the per-candidate
   gather, the `cull_masks` round-trip, and `MB_CULL_SPIN` from this path (mask never
   crosses the NoC → no settle, per the §3c correction). Keep stages as ≤N small programs
   sharing resident L1 (mind the 70656 B ceiling).
4. **T4 — Double-buffer the tile load** (prefetch tile N+1 during compute on tile N) to
   hide the one remaining DRAM read behind compute. Optionally use both data-movers for
   the bulk load.
5. **T5 — (later) quantize the record** (bf16 cov/mean/opacity, 8-bit color) to shrink
   the L1 footprint and the scatter/read bytes; re-gate PSNR. This is the only
   "layout" lever still worth doing, and it stacks on the L1 design.

Expected trajectory: with T1–T3 the blend stage stops being mover-bound (compositing
roofline ≈ 7 ms) and the per-tile bulk read is a few ms; combined with the ta win
(iter 20) the frame moves decisively toward the sub-10 ms blend / 1 ms north-star path.

---

## 5. Root-cause note (so we don't regress into it again)

The gather was never a deliberate choice — the device stages were ported one-by-one from
`gsplat_cpu`, which uses `(gid,tid)` index pairs + gather-by-id (correct for a cached
CPU). The TT port mirrored those data structures instead of the L1-resident design, so
blend inherited a million random DRAM reads. iter 21's AoS record was a step toward the
right *layout* but, applied at the blend-read instead of the tile-assign-scatter, it
only made the doomed gather contiguous (+~6 ms) rather than eliminating it. The lesson:
**on TT, intermediates between a producer-parallel stage and a consumer-parallel stage
must be laid out for the consumer's bulk-contiguous, L1-resident access — never ported
as index-gather from a CPU reference.**

---

## Appendix — iter 21 (S1 AoS record), kept for history

> **SHIPPED default-ON (`GSPLAT_TT_BLEND_AOS`, `=0` reverts).** The gather stage
> (`gather_visible_device.cpp` + `gather_visible_scatter.cpp`) emits a contiguous
> 64-B/record AoS buffer `proj_m_blendrec[g]={a,b,c,px,py,opacity,r,g,b}`, registered in
> `device_state`. The blend reader (gated `MB_BLEND_AOS`) reads ONE contiguous record
> per candidate instead of 7–9 random SoA pages; mask still from `cull_masks`. Wired
> through `fused::process_frame_resident` (reader arg 19).
>
> | | baseline (AoS off) | AoS on | delta |
> |---|---:|---:|---:|
> | blend | 198.5 ms | 192.7 ms | −5.9 ms (−3%) |
> | proj (gather) | 54.5 ms | 58.6 ms | +4.1 ms (blendrec writes) |
> | ms/view | 306.3 | ~304 | −2 to −4 ms |
> | hero_vs_ref | 63.85 dB | 63.85 dB | bit-identical |
>
> The small gain **proved blend is latency/spin-bound, not attr-transaction-bound** —
> which is the evidence that the real fix is removing the gather (§3), not optimizing it.
> The record layout is the right scatter payload for T1.

---

## 6. T2/T3 results + the spin blocker (iter 23) — IMPLEMENTED, GATED OFF

**Status: T2 functionally lands (gather eliminated, gate-clean 63.85 dB); T3's blend-time
win is blocked by a measured, platform-level mask-read settle that cannot be hidden on a
single reader core. The full bucket path is a net regression today, so it stays GATED OFF
(`GSPLAT_TT_TILE_BUCKET`, default off). Production path unchanged: 63.85 dB, blend 192.8 ms.**

### What shipped (gated)
- **Bucket-fed blend reader** (`reader_alpha_blend_mb_devcull.cpp`, `MB_TILE_BUCKET`): for
  tiles with `Lb ≤ MB_BUCKET_FIT`, bulk-load the dense per-tile record bucket DRAM→L1 in one
  batched-barrier pass, **stable LSD-radix depth-sort in L1** (insertion for L≤16; 4×8-bit
  radix otherwise → reproduces the DRAM-radix order so `cull_masks[cull_base+k]` stays
  aligned), then emit coeff rows reading records from L1. **No per-candidate attribute
  gather.** Monster tiles (`Lb > FIT`) fall back to the gather path.
- Host wiring in `blend_device.cpp` (`tile_recs`/`bucket_meta` accessors, `CB_BUCKET`/
  `CB_BSORT`/`CB_BMASK`, `GSPLAT_TT_BUCKET_FIT`). Diagnostic gates added (all default off):
  `GSPLAT_TT_BUCKET_DBG_INLINE`, `…_NOSORT`, `…_NO_FALLBACK_SPIN`, `…_PREFETCH_MASK`,
  `GSPLAT_TT_BUCKET_FINISH`.

### Correctness was fully recovered (the 3.6 dB collapse → 63.85 dB)
Root cause was an iteration-count mismatch: the bucket is sized by `Lb` (dense count from
`sort_bucket_meta`) but the loop used `L` from `sort_tile_ranges`. Switching the bucket path
to iterate on `Lb` restored 63.85 dB (bit-identical to baseline). Records, L1 sort, and emit
are all verified correct (host dump of `tile_recs` showed sane floats; inline-cull probe and
mask-from-L1 both reach 63.85 dB at the working spin).

### The blocker: per-tile `cull_masks`→L1 read needs a fixed busy-wait spin (~84 ms)
The bucket path bulk-loads each tile's whole `cull_masks` region into L1 (`CB_BMASK`) once
per tile, then a per-tile `MB_CULL_SPIN` busy-wait before consuming it. Measured (1 view,
`BUCKET_FIT=16384`, all-bucket, no monster fallback):

| config | hero_vs_ref | blend ms |
|---|---:|---:|
| bucket, spin=0   | 37.6 dB | 109.5 |
| bucket, spin=32  | 34.5 dB | 109.5 |
| bucket, spin=128 | 36.3 dB | 118.7 |
| bucket, spin=256 | 39.5 dB | 142.9 |
| **bucket, spin=512** | **63.85 dB** | **192.4** |

The spin is the entire 109→192 ms (~84 ms) delta, and **512 iters/tile is the minimum that
holds the gate** (PSNR climbs monotonically with spin; nothing below 512 is correct). The
spin is NOT the monster-tile fallback: zeroing only the fallback per-candidate spin
(`…_NO_FALLBACK_SPIN=1`) changed neither time (192.7) nor PSNR (63.85) — at FIT=16384
essentially every tile uses the bucket path, so the cost is the **per-tile** bulk-mask
settle.

**Why it can't be removed (each falsified with a device run):**
1. **`noc_async_read_barrier` is insufficient.** With masks bulk-read + barrier but spin=0 →
   37 dB. Only a time-delay spin makes the L1 mask data coherent for immediate consumption.
2. **An inter-program Finish does NOT help.** Forcing a `Finish` between the (validated
   fused) cull triple and the blend (`GSPLAT_TT_BUCKET_FINISH=1`) — masks 100 % settled in
   DRAM — still gives 37 dB at spin=0 and needs spin=512 for 63.85 dB. So the spin is not
   settling a cross-program write→read hazard; it's settling the **local** NoC-read landing
   into `CB_BMASK`.
3. **Issuing the mask read before the L1 sort provides ZERO settle.** `…_PREFETCH_MASK=1`
   (issue read → do the depth sort as the settle window → barrier → emit) reproduces the
   non-prefetch numbers exactly (spin128→118.7/36 dB, spin256→142.9/39 dB). Elapsed compute
   on the reader does not advance the read; only an explicit busy-wait does.
4. **Un-fusing breaks correctness.** `GSPLAT_TT_FUSED_TILE=0` (two-program scaffold, split
   cull kernels) gives ≤42 dB even at spin=512 — the bucket reader is only validated against
   the fused cull's mask order.

Consequence: the settle is a **busy-wait on the sole reader (dataflow) core**, and each
tile's emit depends on that same tile's freshly-read masks, so it **cannot be overlapped**
(T4 double-buffering can hide a *latency*, but here elapsed work does not drain the read —
only the spin does, per probe 3). The architecturally clean removal — the cull SFPU writing
masks to a **core-local** L1 region consumed by the blend in the **same program** (no NoC
mask read at all) — is the `FUSE_BLEND` path, which **hard-faults at the kernel-config code
ceiling**: `Program size (92688) too large for kernel config buffer (70656) on TENSIX`
(merging the cull ~38 KB + blend ~22 KB SFPU into one compute kernel; +22 KB over). That is
the same iter-14 ceiling and is not trimmable without retuning the proven cull regime.

### Net effect today (why it stays gated)
Enabling `GSPLAT_TT_TILE_BUCKET` also pushes T1's synchronous 64 B record scatter into the
bin (sort 16.2 → 36.4 ms, +20 ms; a ring-buffered T1b would reclaim it) while blend stays
192 ms (gather replaced by bulk-load + L1 sort + the same mask spin → a wash). So the full
path is a **net +20 ms regression** vs the 192.8 ms / 63.85 dB production baseline. Correct
call: keep it gated OFF, banked for when the mask-handoff ceiling is solved.

### The unlock (next, when code-ceiling work is scheduled)
Bake the 32-bit mask **into the bucket record** during the cull pass (write `tile_recs[
rec_start + sorted_pos].mask`), so the blend's *existing* bucket bulk-load — which settles
cleanly via its barrier because `tile_recs` is written in the **prior sort Finish** — carries
the mask with zero extra NoC read and zero spin. This needs the cull pass to emit into
`tile_recs` (and to run in a Finish before blend), i.e. fold the mask into the same
producer→consumer settled-buffer discipline that already makes the record read spin-free.
This is the concrete path to the 109 ms blend (and removes the `cull_masks` buffer entirely).

---

## 7. iter 24 — UNLOCK 1 premise FALSIFIED + UNLOCK 2 is bandwidth-bound (both blocked)

Two more unlock attempts, both with decisive device runs. **Bucket path stays gated OFF.**

### UNLOCK 1 (mask-in-record) — the "spin-free because prior Finish" premise is FALSE
The hope was: a mask written by the cull and read by blend would be spin-free *if* it rode
the record's settled-buffer discipline. Decisive falsification — **a fully separate cull
*program* with its own `distributed::Finish` before blend still needs the spin**:

| path | spin=0 | spin=512 |
|---|---|---|
| `GSPLAT_TT_FUSED_TILE=0` (separate cull program + own Finish), standard blend | 37.6 dB / blend 110 ms | 63.85 dB / blend 194 ms |

So a cull-stage write is NOT made visible to a subsequent blend read by *any* `Finish`
(in-fused-call `BUCKET_FINISH`, OR a separate-program Finish). The records' spin-free read is
**specific to being written in the genuinely-separate sort-stage dispatch** (proj→ta→sort run
well before blend), not to "a prior Finish" or to co-location in a page. Co-locating the mask
in `tile_recs[k]` via a cull RMW would therefore inherit the cull-write settle problem on that
word, not the record's spin-free-ness. **The only write-source that reads back spin-free is the
sort stage** (or earlier). Net: UNLOCK 1 is only viable if the microblock mask is computed
**during the sort stage** and stored in the record there — but the sort program is currently
**dataflow-only** (`sort_bin` + `sort_radix_tile` + `sort_publish`, no SFPU compute kernel), so
this means adding the entire cull SFPU subsystem (reader+compute+writer, the box-origin ramps,
the ~38 KB conic/Mahalanobis compute) to the sort program — a major multi-kernel change with
its own 70656 B code-ceiling risk. Not attempted this iteration (large surface, and the proven
cull regime must not be retuned).

### FUSE_BLEND-resize fallback — re-measured, still 22 KB over
Re-ran `GSPLAT_TT_FUSE_BLEND=1`: still `Program size (92688) too large for kernel config
buffer (70656)`. The bucket path did NOT shrink it because FUSE_BLEND uses its own
`fused_tile_render.cpp` reader (untouched by the bucket work) and the overflow is dominated by
the **merged compute kernel** (cull ~38 KB + blend ~22 KB SFPU on trisc1), not the reader.
Shrinking by 22 KB requires retuning the cull math (prohibited) — not viable.

### UNLOCK 2 (pipeline T1's scatter) — implemented, but the scatter is BANDWIDTH-bound
Replaced `sort_bin`'s synchronous per-record scatter (read-barrier + write-barrier *per
record*) with a `REC_BATCH=16` ring (issue 16 reads → 1 barrier → inject depth → issue 16
writes → 1 barrier; `CB_REC` grown to 16 pages). Correctness held (63.85 dB) but **sort stayed
36.3 ms** (was 35–36 ms synchronous) — batching the barriers changed nothing. Conclusion: the
+20 ms is **not** barrier latency; it is the intrinsic **bandwidth** of materializing the
bucket — reading every kept record from `proj_m_blendrec` and writing it to `tile_recs`
(~2 × P_kept × 64 B ≈ 1.28 GB of NoC traffic). Pipelining hides latency, not bandwidth, so it
cannot recover the 20 ms. The only lever here is **T5 (quantize the record to 32 B)** to halve
the copy (~+10 ms), or fusing the scatter into the gather/project stage so the record is born
in tile-bucket order (no separate copy pass). The ring change is kept (it's the correct
structure and de-risks T1b) but is not a speedup on this input.

### Net status (unchanged): bucket path cannot beat baseline yet
- blend is pinned at ~193 ms by the per-tile mask settle spin (UNLOCK 1 blocked: only
  sort-stage writes settle; FUSE_BLEND overflows 22 KB).
- sort carries +20 ms of bandwidth-bound record copy (UNLOCK 2: not latency-bound).
- Concrete remaining paths, in order of expected payoff: (a) **SFPU microblock cull moved into
  the sort program** so the mask is a sort-stage (spin-free) write baked into the record —
  the real blend unlock, gated by the sort-program code ceiling; (b) **T5 quantize** to halve
  the scatter bandwidth; (c) born-in-bucket-order records from the project/gather stage to
  delete the copy pass entirely.

---

## 8. iter 25 — ROUTE A: the 70656 B kernel-config ceiling IS RAISEABLE (load-bearing)

**Verdict: RAISEABLE via a one-line tt-metal HAL constant. FUSE_BLEND now builds + runs, blend
drops to 116 ms — but the fused path has a separate correctness bug (29.99 dB) to fix.**

### Where the ceiling comes from (exact source)
- `tt_metal/impl/program/program.cpp:2282` `TT_FATAL(state.offset <= max_size, "Program size ...
  too large for kernel config buffer ...")`, where
  `max_size = get_ringbuffer_size(device, TENSIX)` (program.cpp:105):
  `= allocator_config.l1_unreserved_base − hal.get_dev_addr(TENSIX, KERNEL_CONFIG)`.
- `l1_unreserved_base` is **not** the HAL `DEFAULT_UNRESERVED` *base*; `device.cpp:440-459` sets
  `worker_l1_unreserved_start = L1_top − get_dev_size(TENSIX, DEFAULT_UNRESERVED)` — it uses the
  **SIZE** of `DEFAULT_UNRESERVED`.
- In `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp`:
  - `const uint32_t default_l1_kernel_config_size = 69 * 1024;` (line 43) → `= 70656`, the ceiling.
  - `DEFAULT_UNRESERVED` base `= ((MEM_MAP_END + default_l1_kernel_config_size − 1) | align) + 1` (l.69)
  - `DEFAULT_UNRESERVED` size `= MEM_L1_SIZE − DEFAULT_UNRESERVED_base` (l.91-92).
  So bumping `default_l1_kernel_config_size` raises the base **and** shrinks the size by the same
  amount → `worker_l1_unreserved_start` moves up → **the kernel-config ring grows 1:1**, at the cost
  of that many bytes of per-core worker L1. No env/runtime override exists (grepped rtoptions).

### The change (one constant) + how to rebuild the prebuilt tree
- `bh_hal_tensix.cpp:43`: `69 * 1024` → `128 * 1024` (ring 70656 → 131072; reserves +59 KB/core of
  the ~1.5 MB L1 — trivially affordable, bucket uses ~210 KB).
- The remote tt-metal at `/proj_sw/user_dev/smarton/tt-metal` is smarton-owned with source + a
  prebuilt `build_Release` (ninja). gsplat links `build/tt_metal/libtt_metal.so` via RUNPATH, so a
  **lib-only rebuild is picked up with no gsplat rebuild**. The prebuilt tree was configured with a
  cmake at `/usr/bin/cmake` that no longer exists here (now `/usr/local/bin/cmake`, v mismatch) so a
  cmake regen is unsafe; instead rebuild the three artifacts manually from ninja's own commands:
  recompile `bh_hal_tensix.cpp.o`, **re-archive `libhal_1xx.a`** (the .o is archived — relinking
  alone silently keeps the stale object; this was the gotcha), then relink `libtt_metal.so`.

### Result (`GSPLAT_TT_FUSE_BLEND=1`, full resident SFPU stack, views=1)
- **Ceiling gone**: no `Program size too large`; FUSE_BLEND JIT-compiles, binds, and **executes**.
- `[FUSED_TILE] upload+exec+readback=112.9 ms`; `[BLEND_DEVICE] resident_blend=116.1 ms`
  (vs ~193 ms baseline) — the spin-free single-program win, ≈ the predicted ~109 ms.
- **But `hero_vs_ref=29.99 dB`** (gate 63.6). The fused `fused_tile_render`/`fused_tile_writer`/
  `microblock_cull_compute` `FUSE_BLEND` path had never run before (always died at the ceiling), so
  this is its first on-device execution and it has a correctness bug (mask handoff / blend math) —
  independent of the ceiling. Debug next; ceiling itself is proven raiseable + fast.

---

## 9. iter 26 — FUSE_BLEND correctness debug: root cause = cull SFPU corrupts blend in one program (BANKED, gated OFF)

**Verdict: the 30 dB fused bug is an in-program datapath/DST hazard — the cull SFPU phase
non-deterministically corrupts the subsequent blend phase. The clean fix (full inter-phase SFPU
re-init) deadlocks; targeted resets do not fix it. This needs an LLK-level redesign, so FUSE_BLEND
stays gated OFF. Production (FUSE off) re-verified unregressed at 63.85 dB / 304.7 ms/view.**

### Evidence chain (device DPRINT A/B, build-ID verified each run)
1. **Inputs are bit-deterministic & correct.** Dumped the per-gaussian blend inputs the compute
   consumes (raw bits of cov_a, local mean-x, opacity, **and the microblock mask** = `row[10]`) from
   the fused reader for the first 6 gaussians of every core-(1,1) tile, across the warmup+timed
   passes. Under `sort -u` **every `(tile,p,gid)` collapses to a single line** → the mask + attribute
   handoff (cull→writer→`CB_TILE_MASKS`→reader→`CB_MB_COEFF`) is deterministic and matches the
   two-program path. The bug is NOT in data delivery or the mask handoff.
2. **Outputs are non-deterministic.** Dumped raw bf16 from `CB_COLOR_OUT` (`ABCOL`) at the writer.
   The **production** (two-program) path is deterministic (one line/tile under `sort -u`); the
   **fused** path prints **two different values per tile** across the two passes, and the magnitude
   of the run-to-run delta **scales with compositing depth L** (shallow L≈177 differs by ~1 unit;
   deep L≈5113 by ~40–60). A depth-compounding, run-to-run-varying error ⇒ a runtime hazard, not a
   static math/order/format mismatch (those were already audited identical).
3. **It is NOT the cross-gaussian SFPU accumulation pipeline.** Forcing each gaussian into its own
   `_llk_math_eltwise_unary_sfpu_start_/done_` bracket (full SFPU drain between gaussians) left the
   output non-deterministic (29.9→30.3 dB, still two values/tile). RAW separation on DR_R/G/B/T is
   not the cause.
4. **It is NOT an SFPU/MATH config-state leak.** Re-issuing `llk_math_hw_configure` +
   `llk_math_pack_sync_init` (MATH-only, no unpack/pack reconfig) before the blend phase: no change
   (29.93 dB, still non-deterministic).
5. **It IS the cull SFPU phase corrupting blend — pinned by a `CULL_LEVEL` sweep** (the cull's
   compile-time complexity knob, blend math untouched):
   - `CULL_LEVEL=0` (keep-all: only `fill_tile`+pack, **no `copy_tile`, no cull SFPU math**) →
     blend **DETERMINISTIC**, **44.13 dB** (blend-all image; slower 240 ms, expected).
   - `CULL_LEVEL=1` (bbox: `copy_tile(BOX)` + register-only `v_if` SFPU, **no DR_QV/QH, no
     exp/recip**) → blend **NON-deterministic**, 29.09 dB.
   - `CULL_LEVEL=3` (full conic/Mahalanobis) → **NON-deterministic**, ~30 dB.
   ⇒ The corruption is introduced by the **lightest** cull SFPU + `copy_tile` datapath activity
   (present at level≥1, absent at level 0), not by the heavy exp/recip/phasing. Having a `copy_tile`
   (UNPACK→DST) + SFPU op in **both** the cull and blend phases of one program is the trigger;
   `copy_tile` only in blend (level 0 cull) is fine.

### Why the two-program path is immune, and why the obvious fixes fail
The standalone blend program gets a **fresh** UNPACK/MATH/PACK + SFPU + DEST-sync configuration at
launch and never shares DEST with a cull phase. The fused single program reuses the same hardware
config and DEST bank across cull→blend per tile. Both unpack and pack formats **are** already
reconfigured at the cull→blend boundary (`copy_tile_to_dst_init_short(CB_XRAMP/YRAMP)` +
`pack_reconfig_data_format(CB_COLOR_OUT)`), yet non-determinism persists → the residual hazard is
below the format-reconfig layer (SrcA/DEST bank reuse / DEST full-sync ownership / SFPU LREG state
left by the cull datapath). The clean reset that a fresh program performs — `init_sfpu()` /
`unary_op_init_common()` mid-kernel before blend — **deadlocks** the compute pipeline (it re-runs
UNPACK/PACK `hw_configure` while CBs hold pending data; confirmed twice, wedged the device both
times). MATH-only reinit and per-gaussian SFPU drain (the non-deadlocking subsets) do not fix it.

### Decision: BANK, gate OFF (per the bounded-effort directive)
Making FUSE_BLEND correct requires an LLK-level inter-phase pipeline reset that fully drains
UNPACK/MATH/PACK and re-establishes SrcA/DEST/SFPU state **without** the `init_sfpu` deadlock — a
substantial, fragile change in tt-metal LLK territory. That is the "large redesign" boundary, so
FUSE_BLEND remains **default OFF** and the ~116 ms spin-free blend is not shipped this iteration.
- The ceiling raise (iter 25) stands and is independently valuable (proven raiseable + fast).
- Production two-program path re-verified **63.85 dB, blend 193.2 ms, 304.7 ms/view** with the
  reverted kernel — no regression from any debug edit.
- All debug scaffolding (`FUSE_AB` color/row DPRINT dumps in the fused + standalone reader/writer)
  is gated behind the `GSPLAT_TT_FUSE_AB` env/`FUSE_AB` define, default-off; disproven experiments
  (`FUSE_REINIT` per-gaussian bracketing / MATH reinit) were reverted; `CULL_LEVEL` restored to 3.

### Highest-payoff next paths (unchanged ranking, now better-informed)
- **ROUTE C — move the SFPU microblock cull into the SORT program** so the mask is a sort-stage
  (spin-free) write baked into the record. This sidesteps the fused-program hazard entirely (cull
  and blend stay in separate programs, each with fresh config) AND kills the spin — the cleanest way
  to capture the blend win without the in-program datapath hazard. Gated by the sort-program code
  ceiling (now known raiseable via the same HAL constant).
- **Alternatively**, isolate the exact LLK reset needed at the cull→blend boundary (drain + SrcA/DEST
  re-init without deadlock) — smaller code surface than ROUTE C if the right LLK call sequence is
  found, but requires LLK spelunking.
- T5 record quantization (64→32 B) + born-in-bucket-order records remain the scatter-cost recovery,
  to be folded in once a spin-free mask path lands.

---

## 10. iter 27 — ROUTE C: SFPU microblock cull moved INTO the sort stage, mask baked into the record (IMPLEMENTED + PROVEN spin-free; BANKED gated OFF — blocked by a PRE-EXISTING bucket-blend bug for Lb>64)

**Verdict: ROUTE C works exactly as designed — the per-microblock keep-mask is now produced by a
SORT-stage SFPU pass and baked into record word 10, and the L1-resident blend reads it back
SPIN-FREE (no `MB_CULL_SPIN`, no `cull_masks` DRAM round-trip on that path). This is PROVEN: with the
bucket path active, ROUTE C (baked recp[10]) matches/slightly beats the old cull_masks+spin path at
the SAME PSNR, so the mask handoff is correct. BUT the gated L1-resident bucket-blend path it feeds
has a PRE-EXISTING correctness bug (the documented "scatter-induced sort regression") that caps it at
~42 dB and blocks the gate. ROUTE C cannot ship as default until that bucket-blend bug is fixed.
Production (FUSED_TILE, no bucket) stays unregressed at 63.85 dB. Every device run went through
`devrun.sh`.**

### What was implemented (all gated behind `GSPLAT_TT_BUCKET_MASK`, production default OFF)
- **New sort-stage cull program** (`sort_device.cpp::build_program_bucket_cull`, launched in
  `sort_resident_pairs` after `launch_bin(1)`, gated `tile_bucket && bucket_mask_enabled() &&
  cull_built && P_kept>0`): a 3-kernel reader/compute/writer workload reusing the EXISTING
  `microblock_cull_compute.cpp` SFPU kernel.
  - `kernels/dataflow/reader_bucket_cull.cpp` (new): streams each LPT tile's dense records from
    `sort_tile_recs[rec_start..+count)` (SEQUENTIAL, not gathered), emits the 6-word cull coeff row
    + per-gaussian Mahalanobis `thr` (logf on the data-mover) — byte-identical coeff rows to
    `reader_microblock_cull.cpp`.
  - `kernels/dataflow/writer_bucket_cull.cpp` (new): packs the 32-bit microblock mask from `CB_KEEP`
    (same `perm(g,m)` unpack as `writer_microblock_cull.cpp`) and RMWs it into **record word 10** of
    the dense bucket. Preserves words 0..9.
- **`device_state` set/get_bucket_cull_params** (+ `pybind_module.cpp` publish before
  `sort_and_bin_tt`): hands `contrib_floor` + `cull_disabled` to the sort driver (the blend host args
  never reach the sort stage).
- **Blend reader** (`reader_alpha_blend_mb_devcull.cpp`, `MB_BUCKET_MASK` define from
  `blend_device.cpp`): on the bucket path reads `const uint32_t mask = recp[10]` (pure L1 load) and
  the `MB_BUCKET_PREFETCH_MASK` / `MB_CULL_SPIN` blocks are `#if`-excluded — NO spin, NO `cull_masks`.

### Device numbers (yyzo-bh-03, 1 view, build cpp#82 sha=470095f, all via devrun.sh)
| config | hero_vs_ref | blend ms | note |
|---|---|---|---|
| production (FUSED_TILE=1, no bucket) | **63.85** | ~193 | gate baseline, unregressed |
| ROUTE C: `TILE_BUCKET=1 BUCKET_MASK=1` (FIT=8192 default) | 42.84 | 178 | spin-free mask works; bucket-blend bug |
| control `TILE_BUCKET=1 BUCKET_MASK=0` (old cull_masks+spin) | 42.27 | — | SAME ~42 ⇒ bug is NOT the mask |
| `BUCKET_FIT=1` (≈all tiles → gather fallback) | **63.85** | ~110(spin) | gather path correct |
| `BUCKET_DBG_NOSORT=1` (bucket, skip L1 sort) | 16.05 | — | scatter order = random (sanity) |
| `BUCKET_FIT=16` (only Lb≤16 → bucket, insertion sort) | **63.85** | — | bucket plumbing + insertion sort correct |
| `BUCKET_FIT=64` (Lb≤64 → bucket, single-batch radix) | **63.85** | — | **L1 radix algorithm correct for Lb≤64** |

### Root cause of the residual 42 dB — PRECISELY localized (NOT a ROUTE C defect)
The bucket-eligible blend path is **correct for tiles with Lb ≤ 64 and wrong only for Lb > 64**:
- `BUCKET_MASK=0` also gives ~42 ⇒ masks are not the cause; ROUTE C's per-record mask baking is
  correct (matches/beats cull_masks).
- `NOSORT` = 16 dB but the L1 sort lifts it to 42 ⇒ the L1 sort does real, mostly-correct work.
- `FIT=16` and `FIT=64` both hit the full gate (63.85) ⇒ the bucket meta lookup, the bulk record
  load, the coeff emit, AND the stable LSD radix ALGORITHM are all correct (Lb∈[17,64] exercises the
  radix in a single 64-page load batch and is bit-correct).
- The only tiles that differ between FIT=64 (pass) and FIT=8192 (42) are those with **Lb∈(64, 8192]**.
  `id_start`/`cull_base` are read per-tile from metadata (not accumulated), so the bucket path's
  `continue` does NOT corrupt the gather-fallback tiles — the corruption is in the Lb>64 bucket tiles
  themselves.
⇒ **Next lever: the dense-bucket record ASSEMBLY for tiles whose records exceed one 64-page load
batch / span multiple binning cores.** The reader's multi-batch load loop
(`while(i<L){end=min(i+64,L); 64 reads; barrier;}`) is logically correct, so suspect the
SORT-stage dense scatter in `sort_bin.cpp` (BIN_EMIT_REC): `brec_page = recrowp[t] + curp[t]` with
the host-computed per-(core,tile) `recbase` prefix. A tile whose kept gaussians are split across
multiple binning cores (which is exactly the Lb>64 / dense regime) likely has its records written
with a cross-core base/offset mismatch (overlap, gap, or a different cross-core concatenation order
than the canonical (key,id) layout `sort_radix_tile` consumes), so the contiguous
`tile_recs[rec_start..+Lb)` the blend reads back is not the tile's true record set. Verify by dumping,
for a known Lb>64 tile, `bucket_meta(rec_start,Lb)` vs the canonical `(id_start, id_end-id_start)` and
the set of `recp[9]` keys vs the canonical sorted keys; then audit the host `recbase` prefix-sum
(per-core dense bases) against `bin2d`.

### Decision: BANK, gate OFF (bounded-effort directive)
- ROUTE C is **default OFF** (`GSPLAT_TT_BUCKET_MASK` unset). Production unchanged + re-verified
  63.85 dB. The sort-stage cull program is only built/launched when the flag is set
  (`bucket_mask_enabled()`), so it adds zero work to the production sort.
- The spin-free sort-stage mask mechanism is DONE and reusable: the moment the Lb>64 bucket-blend
  assembly bug is fixed, flipping `GSPLAT_TT_TILE_BUCKET=1 GSPLAT_TT_BUCKET_MASK=1` should give the
  full-gate, spin-free, fully-L1-resident blend (target ~110 ms) with no further mask work.
- Cost of the sort-stage cull pass itself: `[BUCKET_CULL] exec≈74 ms` (steady-state) — this runs in
  the separate sort dispatch and is the price of moving the cull off the blend critical path; it is
  NOT yet overlapped/hidden and is a follow-up tuning item once correctness lands.

---

## 11. iter 28 — Lb>64 bug RE-ROOT-CAUSED: it is a FAST-PRODUCER timing race in the bucket emit→blend-compute pipeline, NOT cross-core record assembly and NOT the mask value

**Verdict: §10's prime suspect (cross-core `recbase`/dense-record ASSEMBLY for Lb>64) is REFUTED. The
dense bucket records are bit-perfect, the baked `recp[10]` mask is bit-perfect, and the L1 sort order
is correct — for ALL tile sizes incl Lb>64. The 42 dB is a producer/consumer TIMING race: the
L1-resident bucket reader emits coeff rows into `CB_MB_COEFF` faster than `alpha_blend_compute_mb.cpp`
consumes them, and the fast feed corrupts the blend. Slowing the reader per-record (≈2000 cyc) OR
making it do real per-record work (inline cull) recovers the FULL gate (63.85). Production (FUSED_TILE)
re-verified 63.85 dB / 303 ms/view, unregressed. Every device run via `devrun.sh`.**

### Decisive experiments (yyzo-bh-03, 1 view, all via devrun.sh; FUSED_TILE=0 TILE_BUCKET=1, FIT=8192)
| config | hero_vs_ref | what it proves |
|---|---|---|
| `BUCKET_MASK=1` (baked recp[10], fast emit) | **42.87** | the bug |
| `BUCKET_MASK=1` + `BUCKET_FORCE_INLINE=1` (RMW still runs; blend recomputes mask inline from L1 record, ignores recp[10]) | **63.85** | RMW does NOT corrupt records; words 0..9 perfect; bug is the recp[10] *read path*, not the record |
| `BUCKET_MASK=1` + `BUCKET_MASK_DEBUG=1` (still blends with `mask=recp[10]`, adds per-record inline ref compute + occasional DPRINT) | **63.85**, `BSUM mism=0` for all Lb>64 tiles (L up to 7616) | recp[10] in memory EQUALS inline ref (baked mask correct); the only delta vs the 42 dB run is per-record LATENCY |
| `BUCKET_MASK=1` + `BUCKET_EMIT_SPIN=200` | 37.34 | tiny throttle is NOT monotonic (race, not a settle) |
| `BUCKET_MASK=1` + `BUCKET_EMIT_SPIN=2000` | **63.85** (blend 388 ms) | ≈ one SFPU-gaussian time of per-record throttle fixes it |
| `BUCKET_MASK=1` + `BUCKET_EMIT_SPIN=8000` | **63.85** (blend 1182 ms) | confirms throttle, far too slow |
| `BUCKET_MASK=1` + `CB_MB_COEFF` depth 1 / 2 / 4 / 8 | 33.3 / 33.9 / 43.3 / 42 | depth does NOT fix it; deeper (reader runs further ahead) is WORSE → producer-ahead race, not wrap/visibility |

### What this rules OUT (vs §10)
- **Cross-core record assembly / `recbase` prefix**: REFUTED. `FORCE_INLINE` reads `recp[0..8]` and the
  sort key `recp[9]` from the SAME dense-bucket records at the SAME sorted indices and renders 63.85
  for Lb>64. If the assembly were wrong, those reads would be garbage too. (The §10 `BUCKET_VERIFY`
  key-multiset probe also reported `mism=0` for dense tiles.)
- **The baked mask value**: REFUTED. `MASK_DEBUG` shows `recp[10] == inline-ref` (`mism=0`) for every
  Lb>64 tile, yet the *fast* run that uses exactly that value renders 42 dB. Same bytes, different
  result ⇒ timing, not data.
- **CB depth / wrap-stomp / write-visibility**: REFUTED. Deeper `CB_MB_COEFF` is worse, not better; a
  store-buffer flush would be ~tens of cycles, but the fix needs ~2000.

### Current best hypothesis (for the next worker)
The bucket emit loop (`reader_alpha_blend_mb_devcull.cpp`, the `Lb≤MB_BUCKET_FIT` branch, ~line 730)
is a TIGHT pure-L1 loop: `recp` read → write `CB_MB_COEFF` row → `cb_push_back`, with NO NoC ops
between pushes. The gather fallback uses the identical row-write + push but is naturally throttled by
its per-chunk `noc_async_read_barrier()`s, so it never exposes the race (which is why FIT=64 — almost
all tiles on the gather path — passes). When the bucket reader floods `CB_MB_COEFF`, the compute
(`process_tile_gaussians` → `dispatch_blend_guarded`, one `MATH()` SFPU call per masked microblock under
a single `_llk_math_eltwise_unary_sfpu_start_/done_` per tile) produces wrong output. CB backpressure
*should* prevent this, so the residual hazard is suspected to be an LLK/CB ordering issue between a
fast NCRISC producer and the TRISC SFPU consumer (analogous in spirit to the §9 fused-program
DEST/SFPU-state hazard) — i.e. the compute reading/blending a row before the producer's effect is
coherent, or an SFPU-pipeline state issue that only manifests at full feed rate.

**Suggested next steps (ranked):**
1. Add a CHEAP, correct per-tile (not per-record) reader↔compute fence on the bucket path instead of a
   blind spin — e.g. a real producer→consumer semaphore handshake, or interleave the existing harmless
   `noc_async_read_barrier()` cadence the gather path has. Goal: throttle WITHOUT the 2000-cyc spin tax.
2. Instrument the COMPUTE (`alpha_blend_compute_mb.cpp`) under fast feed: `MB_COEFF_DEBUG` dumps the
   first rows — compare `row[*]` the compute reads against what the reader wrote for a fixed dense tile
   at SPIN=0 vs SPIN=2000. If the compute reads a stale/garbage row only at SPIN=0, it is a CB data-
   coherence bug (producer write not visible at consume); if the row is identical but output differs,
   it is an SFPU/DEST pipeline-state bug at full rate.
3. If (1) lands the gate spin-free, the fully-L1-resident blend is unblocked: flip
   `GSPLAT_TT_TILE_BUCKET=1 GSPLAT_TT_BUCKET_MASK=1`. Note even the throttled-correct run is currently
   ~388 ms blend in the FUSED_TILE=0 split-pass config (separate 90 ms cull pass + sort-stage RMW), so
   hitting the ~110 ms target also needs the cull pass folded/hidden — a separate perf item.

### Diagnostic knobs left in place (all env-gated, production-safe, default OFF)
- `GSPLAT_TT_BUCKET_FORCE_INLINE` → `MB_BUCKET_FORCE_INLINE`: bucket path recomputes mask inline
  (ignores recp[10]); gives 63.85 and is the current WORKING (if slow) fully-L1-resident proof point.
- `GSPLAT_TT_BUCKET_EMIT_SPIN=<n>` → `MB_BUCKET_EMIT_SPIN`: per-record reader throttle (diagnostic).
- (pre-existing) `GSPLAT_TT_BUCKET_MASK_DEBUG` → `MB_BUCKET_MASK_DEBUG`: per-record baked-vs-inline
  `BSUM` comparison.

### Decision: BANK, ROUTE C + bucket path remain default OFF
Production (FUSED_TILE, no bucket) re-verified **63.85 dB, blend 192.3 ms, 303.3 ms/view** — unregressed
by any of the diagnostic edits. The bug is now precisely characterized as a fast-producer timing race
(not the assembly/mask the prior handoff assumed), which redirects the fix to the reader↔compute
handshake rather than `sort_bin.cpp`/`recbase`.

## 12. iter 29 — RESOLVED: HYPOTHESIS A proven; root cause = UNPACK frees the CB slot before the MATH direct-L1 read; FIX = MATH→UNPACK back-pressure ack + MATH cache-invalidate (gated `GSPLAT_TT_BUCKET_CB_FENCE=1`)

**Verdict: HYPOTHESIS A (stale/torn row read), not B (SFPU pipeline hazard). The bucket fast path now
hits 63.85 dB WITHOUT EMIT_SPIN, gated behind `GSPLAT_TT_BUCKET_CB_FENCE`. Production unregressed.
Every device run via `devrun.sh`; device not wedged.**

### The A/B probe (gated `GSPLAT_TT_BUCKET_AB_PROBE`)
Reader stamps `row[11]=k` (sorted record index) and `row[12]=XOR(row[0..10])`; the compute MATH thread
re-checks them right where it loads the blend coeffs. At full feed the probe showed widespread
mismatches and `chk_ok=0` (TORN rows) with PSNR stuck at 43–44 dB ⇒ the compute reads INCOHERENT row
bytes, not correct-bytes-wrong-math. That decides **A over B**. (Caveat: the probe's `seqmm`/`chkmm`
counters are NOT a clean pass/fail signal — `k` is the sorted index, not the per-tile `g`, and the extra
words perturb timing — so the IMAGE PSNR is the ground truth. Even the final fixed run shows
`seqmm≈rows` while rendering a perfect 63.85 dB.)

### The mechanism (found by reading `tt_metal/hw/inc/api/compute/cb_api.h`)
`alpha_blend_compute_mb.cpp` reads each `CB_MB_COEFF` row from L1 **directly on the MATH thread**.
`get_tile_address(cb,0)` only computes the address on UNPACK (from its `fifo_rd_ptr`) and **mailboxes**
it to MATH/PACK. But `cb_wait_front`/`cb_pop_front` are **UNPACK-only macros**: UNPACK pops (frees the
slot) as soon as it has mailboxed the address — it does NOT wait for the slow, SFPU-bound MATH read. On
the throttle-free bucket feed the NCRISC producer then refills the freed slot (depth 8) with a LATER
record before MATH loads it ⇒ torn/stale rows. The gather/production paths survive only because their
per-chunk `noc_async_read_barrier()`s throttle the producer (and `EMIT_SPIN≈2000` faked that delay).
A token-in-`row[15]` data-ready poll did NOT work (44 dB): the slot is reassigned to a *different*
record, so an exact-token wait never matches.

### The fix (two cheap, bounded, NON-spin parts — both in `alpha_blend_compute_mb.cpp`, gated `MB_BUCKET_CB_FENCE`)
1. **MATH cache-invalidate** (`invalidate_l1_cache()` == `asm("fence")`) before the row loads — Blackhole
   L1 is a write-through cache and MATH may hold a stale line for the recycled (depth-8) slot address.
2. **MATH→UNPACK back-pressure ack** over the hardware mailbox: after MATH has loaded every coeff word
   (`mailbox_write(UnpackThreadId, g+1)`), UNPACK blocks on `mailbox_read(MathThreadId)` **before**
   `cb_pop_front`. UNPACK can therefore never free a slot the producer would overwrite before MATH read
   it. An `asm volatile("" ::"r"(a)…:"memory")` pins the non-volatile coeff loads into registers *before*
   the ack (else the compiler could sink them past it). Reader side just keeps a store-ordering fence
   before `cb_push_back`. Cost: one mailbox round-trip + one fence per gaussian (tens of cycles), vs the
   2000–8000 cyc `EMIT_SPIN`.

### Results (yyzo-bh-03, 1 view, all via devrun.sh)
| config | hero_vs_ref | blend ms | ms/view |
|---|---|---|---|
| bucket fast path, FENCE **off** (the bug) | 42–44 | — | — |
| bucket fast path + `BUCKET_CB_FENCE=1` (the fix) | **63.85** | **178.9** | 372.9 |
| PRODUCTION (FUSED_TILE default, fix gated off) | **63.85** | **192.9** | **303.9** |

The fixed bucket blend (178.9 ms) is actually FASTER than production's fused blend (192.9 ms). The
remaining ms/view gap (372.9 vs 303.9) is the **separate ~80–90 ms cull pass** in the FUSED_TILE=0
split-pass config — folding/hiding that cull pass is the **next lever** to reach the ~110 ms blend target.

### Decision: BANK, gated OFF (default), production unregressed
`GSPLAT_TT_BUCKET_CB_FENCE` stays default OFF; the bucket fast path is correctness-complete behind it but
not yet a ms win at the frame level until the cull pass is folded. Production path untouched.

## 13. iter 30 — the L1-resident path BEATS production: the "cull pass" was a REDUNDANT second cull, not the cull itself (commit, gated OFF default)

**Verdict: the L1-resident bucket blend now renders 63.85 dB at 298.8 ms/view vs production 303.4 — a
verified net win (−4.6 ms single-view, −9.3 ms / 4.9% at 8-view steady-state) — with NO new kernel and
NO regression. Root cause of the +69 ms gap was NOT "the cull is expensive": it was that the
`BUCKET_MASK` (ROUTE C) config ran TWO full SFPU cull passes per frame. Dropping the redundant one is the
whole win. Production (FUSED_TILE) re-verified 63.85 dB / 303.4 ms/view on the same binary, unregressed.
Every device run via `devrun.sh` (yyzo-bh-03, build cpp#93 bin=dd34548033e587b3). Gated OFF by default.**

### Where the +83 ms "sort+cull" actually went (localized via [MB_TIMING] device logs, not host wall)
The iter-29 L1 config (`FUSED_TILE=0 TILE_BUCKET=1 BUCKET_MASK=1 FIT=8192 CB_FENCE=1`) logs BOTH:
- `[BUCKET_CULL] exec≈74 ms` — the ROUTE-C sort-stage SFPU pass that RMWs the keep-mask into record
  word 10 (folded into the SUMMARY `sort` stage → `sort=99.4`).
- `[CULL_SPLIT] exec≈80 ms` — the *production* standalone SFPU microblock-cull (`cull::process_frame`,
  reader/compute/writer_microblock_cull) that fills the `cull_masks` DRAM buffer. It ALWAYS runs when
  `GSPLAT_TT_SFPU_CULL=1` (folded into the SUMMARY `blend` stage along with the ~95 ms real blend →
  `blend=179`).

So the `BUCKET_MASK` path computed the keep-mask TWICE: `recp[10]` (used by bucket tiles) AND `cull_masks`
(used only by the Lb>FIT gather-fallback tiles). The bucket tiles read `recp[10]` and **ignore**
`cull_masks` — the entire 74 ms `BUCKET_CULL` pass is pure overhead on top of the cull the frame already
pays for. That is the +69 ms vs production, almost exactly.

### The fix (config + one tiny correctness-ergonomics code change)
**Drop `BUCKET_MASK`.** With `bucket_mask_enabled()==false` the sort-stage `BUCKET_CULL` pass is never
built/launched, and the bucket blend reader reads the keep-mask from the **shared** `cull_masks` (bulk-
loaded once per tile into L1, indexed `cull_base+k` — the depth-sorted layout the stable LSD radix
reproduces). Fallback tiles read the same `cull_masks`. **One** cull pass (`CULL_SPLIT`) now serves the
whole frame.
- Code change (`blend_device.cpp`): `MB_BUCKET_CB_FENCE` is now **DEFAULT-ON whenever the bucket path is
  active** (it is the proven-necessary iter-29 fast-producer-race fix; without it the no-mask bucket path
  also caps at ~42 dB). Opt out with `GSPLAT_TT_BUCKET_CB_FENCE=0`. The define is only ever set inside
  `if (tile_bucket)` (reader) / gated on `tile_bucket` (compute), so production's FUSED_TILE path is
  untouched. This makes the shippable L1 config a clean 3-flag gate that is correct without remembering
  the fence flag.

### Device numbers (yyzo-bh-03, 1 view unless noted, all via devrun.sh, build cpp#93)
| config | hero / min@8v dB | ms/view (1v) | ms/view (8v) | proj / ta / sort / blend (1v) |
|---|---|---|---|---|
| Production (FUSED_TILE=1) | 63.85 / 36.57 | **303.4** | 191.5 | 58.5 / 36.4 / 15.7 / 192.6 |
| L1 OLD (BUCKET_MASK, 2 cull passes) | 63.85 | 372.9 | — | 58.6 / 35.7 / **99.4** / 179.0 |
| **L1 WIN (no BUCKET_MASK, 1 cull pass)** | 63.85 / **60.82** | **298.8** | **182.2** | 58.7 / 35.4 / **26.2** / 178.3 |
| Option 1 (inline soft-float cull in reader) | 63.85 | **536.3** | — | 58.5 / 34.0 / 26.6 / **416.9** |

- The L1 WIN's `min_vs_ref` across 8 views (60.82) is HIGHER than production's own 8-view min (36.57), so
  the bucket path introduces no per-view regression — 60.82 is the scene's natural tt-vs-ref spread.
- **Option 1 (fold the cull inline into the blend reader, like production's
  `reader_alpha_blend_mb_devcull`) is a TRAP on the L1-resident path:** it blows blend to 416.9 ms.
  Production's inline soft-float Mahalanobis cull is "free" ONLY because it hides in the random-gather
  latency shadow (the mover is ~85 % idle waiting on DRAM). With L1-resident records the mover is no
  longer stalled, so the same soft-float cull becomes fully-exposed serial work. ⇒ A pre-computed mask
  (SFPU `cull_masks`) is *essential* once the gather shadow disappears; do not re-fold cull into the
  L1-resident reader.

### Decision: BANK gated OFF, do NOT flip default yet
The win is verified and clean, but the margin is thin (1.5 % single-view) and the bucket path is so far
validated only on the bicycle scene + relies on the recent CB_FENCE handshake and the Lb>FIT gather
fallback. Flipping the trusted production default for ~1.5–4.9 % is not yet worth the risk. **Banked gated
OFF** (`GSPLAT_TT_FUSED_TILE=0 GSPLAT_TT_TILE_BUCKET=1 GSPLAT_TT_BUCKET_FIT=8192` → 298.8 ms/view, 63.85 dB).

### Next lever (the single biggest remaining cost is now `CULL_SPLIT`)
After folding, the L1 blend stage (178.3 ms) is still ≈ `CULL_SPLIT` (80 ms) + real blend (95 ms) +
readback. The 80 ms standalone SFPU cull is now the #1 target:
1. **Overlap it (Option 2):** `CULL_SPLIT` is serialized proj→ta→sort→CULL_SPLIT→blend because
   `cull_masks` is written in the sorted (`cull_base+k`) layout. If the cull is re-indexed to the
   pre-sort candidate position (or the mask is carried through the sort scatter), the SFPU cull could run
   concurrently with the sort stage and leave the critical path → another ~80 ms.
2. Fusing `CULL_SPLIT`+blend into one program is still blocked by the §9 iter-26 cull-SFPU↔blend-SFPU
   DEST hazard, so keep them in separate programs.
3. Validate the bucket path across all 30 views + the other scenes before any default flip.

## 14. iter-31 — CULL_SPLIT overlap investigation: it's irreducible device SFPU, not a host bubble

**Goal:** hide the ~80 ms `CULL_SPLIT` SFPU pass (toward ~220 ms/view). **Outcome:** banked a gated,
gate-clean latency change (`GSPLAT_TT_CULL_PIPELINE`, ~1 ms) and proved the 80 ms is genuine device SFPU
compute that cannot be hidden without restructuring the sort. Below is the evidence trail.

### 14.1 Measure-first: what is the 80 ms?
Three independent signals all say **device-side SFPU compute**, not dispatch/Finish latency or mask writeback:
- **ROUTE-C comparison (§ earlier):** the sequential-record cull (no random gather) was 74 ms vs the
  random-gather `CULL_SPLIT` at 80 ms — only 6 ms is gather. The remaining 74 ms is the Mahalanobis SFPU math
  over 3.37 M candidates. So Option C ("gather is the cost, move it off SFPU") is false: the SFPU *is* the cost.
- **Pipeline probe (this iter):** I gated the cull program's own `distributed::Finish` behind
  `GSPLAT_TT_CULL_PIPELINE`. The cull and resident blend share ONE in-order CQ, so dropping that Finish only
  removes a HOST bubble (lets the blend's ~110-core host/runtime-arg setup overlap the cull's device window).
  Net effect: **300.0 → 298.8 ms/view (~1 ms)**, 63.85 dB unchanged. If the 80 ms were dispatch/Finish latency,
  this would have recovered far more. It didn't ⇒ the 80 ms is device execution.
- **Profiler:** the cull-like stage runs NCRISC (mover) ~93 % busy and TRISC (SFPU) ~84 % busy over ~89 ms —
  both cores saturated; this is a compute/movement-bound kernel, not an idle dispatch gap.

### 14.2 Why A/B/C don't cheaply land the win
- **Single shared resource.** cull-SFPU (80 ms) and blend-SFPU (~93 ms) both run on the *same* SFPU cores.
  Even on separate command queues they serialize: 80 + 93 = 173 ms of irreducible SFPU on the critical path.
  Multi-CQ gives no concurrency when the bottleneck resource is shared.
- **Option A (overlap with sort) ceiling is only 26 ms.** The cull can in principle run under the sort's
  shadow, but the sort is just 26 ms — so even a *perfect* overlap hides at most 26 of the 80 ms, and only by
  carrying/re-indexing the keep-mask **through the depth-radix scatter** (the cull writes `cull_masks` in the
  post-sort `cull_base+k` layout; computing pre-sort means re-threading every mask bit through the radix
  permutation). That is sort-pipeline surgery (add an SFPU compute stage to the sort program + radix mask
  carry + cross-program sync) for ≤ 26 ms — high risk, low ceiling. Banked, not attempted.
- **Option B (bake mask into scatter record)** is the same radix-carry problem and risks reintroducing the
  exact ROUTE-C `recp[10]` duplicate-cull that 2d1f93f just removed.
- **Fusion** stays blocked by the §9 iter-26 cull-SFPU↔blend-SFPU DEST hazard.

### 14.3 What landed (gated, default OFF)
`GSPLAT_TT_CULL_PIPELINE=1` skips the cull program's standalone host `Finish` so the blend's host-side program
setup overlaps the cull's device window. Correctness unchanged (same in-order CQ; the blend's first blocking
readback still drains the cull before any host read). **L1-resident: 63.85 dB, 298.8 ms/view** (proj 58.7,
ta 35.4, sort 26.2, blend ~177). **Production (FUSED_TILE default): 63.85 dB, 304.8 ms/view — unregressed.**
This is within noise of the 298.8 §13 baseline; it is banked as the gate-clean attempt + the evidence that the
cull is device-bound, NOT as a material win.

### 14.4 Next lever
The only path to materially hiding the cull is **Option A done properly**: move the microblock-cull SFPU compute
INTO the sort program so it runs concurrently with the radix, and carry the per-candidate keep bit through the
radix scatter so `cull_masks` lands in post-sort order "for free" under the 26 ms sort shadow. Ceiling ~26 ms,
not 80. To go below that you must reduce SFPU *work*: lower `CULL_LEVEL` math cost (cheaper keep test) or cull
fewer candidates earlier (tighter tile-assign / opacity prefilter) so the 3.37 M-candidate SFPU pass shrinks.
That — shrinking the SFPU candidate count — is the higher-leverage direction than overlap.

## 15. iter-32 — tile→core LOAD BALANCE for the SFPU cull+blend stages: already LPT, already near-optimal

**Task:** diagnose tile→core load imbalance for the SFPU-bound cull (~80 ms) and blend (~178–193 ms) stages
and, if imbalanced, ship LPT. **Outcome:** the imbalance is ALREADY SOLVED — LPT (longest-processing-time)
tile→core assignment is shipped and active on BOTH the production and L1-resident paths, and it packs the
work to within **1.3 % of perfect** (theoretical) / **~7 % realized on-device**. There is no LPT win left
to capture for these stages. Banked the diagnosis + the chosen next lever (candidate-count shrink) with
hard numbers; production re-verified unregressed.

### 15.1 The current tile→core scheme — it is already LPT (not round-robin, not tile-id order)
Both stages read a single resident assignment built once per frame in the sort stage:
- `sort_device.cpp::build_lpt()` (and the mirror `blend_device.cpp::compute_lpt_assignment()`): **sort all
  non-empty tiles by DESCENDING candidate count, then greedily place each onto the currently
  least-loaded core.** Cost proxy = per-tile kept-candidate count (`tile_pad[t]` / `ids_off[t+1]-ids_off[t]`),
  which is exactly what drives cull (one 32-lane SFPU mask op per candidate) and is a good proxy for blend.
- Published resident as `sort_lpt_tile_ids` (concatenated per-core tile lists) + `sort_lpt_meta`
  (per-core `(offset,count)`). The cull pass (`cull::process_frame`) and the blend reader both index their
  tiles through this same LPT layout — no host re-derivation, no round-robin, no tile-id order.
- Empty tiles are filtered out entirely (correctness-critical: dispatching the CB pump for thousands of
  empty tiles deadlocks ≥16 cores).

### 15.2 Hard numbers — LPT is at its packing optimum (gated `GSPLAT_TT_LPT_STATS`, device, cpp#95)
One device run (production FUSED_TILE, bicycle hero, via devrun.sh) with the new host-only
`GSPLAT_TT_LPT_STATS` dump on `build_lpt`:

```
[LPT_STATS] tiles_nonempty=1024 cores=110 total_cand=4306640 mean/core=39151
            median/core=39344 max/core=39664 min/core=38512 heaviest_tile=26544
            | makespan/mean=1.0131  makespan-mean=513 cand (=1.3% headroom)  heaviest/mean=0.6780
```

- **Theoretical LPT headroom = 1.3 %** (busiest core 39,664 vs mean 39,151 candidates; only 513 over mean).
- **The heaviest single tile = 26,544 candidates = 0.678× the per-core mean (39,151).** Because the
  biggest tile is *well below* the mean per-core load, NO single item forces the makespan above the mean
  → LPT's lower bound `makespan ≥ max(mean, heaviest_tile)` reduces to `mean`, and LPT achieves it to 1.3 %.
  This is the structural reason there is no win: the workload has no monster tile to split.
  (`total_cand` here is the page-padded count 4.31 M; the dense `P_kept` is 3.37 M — padding inflates the
  cost slightly but the balance conclusion is unchanged.)

### 15.3 Realized on-device per-core SFPU imbalance (device-zone Tracy, `profile_summary_l1.json` + CSV)
Per-core busy time WITHIN each stage window (110 cores), parsed from the raw device-profiler CSV:

| stage (SFPU/TRISC-bound) | TRISC max/mean | makespan−mean | read |
|---|---|---|---|
| **cull** (TRISC busy ~74.6 ms) | **1.067** | ~5.0 ms | well balanced |
| **blend** (TRISC busy ~71–76 ms) | **1.067–1.072** | ~5.1 ms | well balanced |

The realized 6.7–7.2 % (vs the 1.3 % candidate-count theoretical) is the residual that LPT-by-candidate
**cannot** capture: per-tile cost also depends on each candidate's microblock-touch count `k` (a dense
foliage tile costs more per candidate than its candidate count implies). But it is only ~5 ms on a ~75 ms
stage — too small to chase, and not addressable by re-balancing tile→core (it would need a `Σk`-weighted
cost proxy, whose gain is bounded by these ~5 ms).

For contrast, the **non-LPT stages ARE imbalanced**: project (contiguous N-page split, `split_pages`) shows
TRISC max/mean **1.56** (~33 ms makespan−mean) and tile_assign (BRISC) **1.54** (~19 ms). Those are real but
(a) out of scope for the SFPU cull/blend target and (b) not tile→core problems (project runs before tiles
exist; its imbalance is variable per-chunk survivor counts, fixable only by finer chunking / work-stealing).

### 15.4 Verdict + next lever
**No LPT load-balance win exists for cull/blend — it is already shipped and within 1.3 %/~7 % of optimal.**
The dominant cull+blend cost is therefore NOT imbalance; it is (a) the cull-SFPU↔blend-SFPU serialization
(§14: 173 ms on the shared SFPU cores, fusion blocked by §9 DEST hazard) and (b) the total SFPU candidate
count. The single highest-leverage plan-aligned lever remains **shrinking the SFPU candidate count**
(cheaper `CULL_LEVEL` keep-test, tighter tile-assign / opacity prefilter) — it cuts both the cull and blend
makespan *and* shrinks the heaviest tile, the only thing that could ever re-introduce LPT headroom.
A secondary banked finding: **project's ~33 ms contiguous-split imbalance** is a separate, real load-balance
opportunity (finer/work-stealing chunking), but off the cull/blend critical path.

The `GSPLAT_TT_LPT_STATS` dump is host-only, default OFF, zero effect on the assignment — kept in-tree so
the LPT balance can be re-measured any frame without re-parsing the profiler. Production (FUSED_TILE
default) re-verified **63.85 dB / 303.5 ms/view** (proj 58.5, ta 36.4, sort 16.0, blend 192.4) — unregressed.
