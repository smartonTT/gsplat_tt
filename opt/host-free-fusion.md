# Host-free fused cull/blend — deep analysis + staged spec

Status: **analysis/design only** (bh-30 firmware-wedged). No device run, build, or
kernel edit was made to produce this. All claims are read from the actual kernels
(`reader_alpha_blend_mb_devcull.cpp`, `alpha_blend_compute_mb.cpp`,
`writer_alpha_blend.cpp`, `microblock_cull_compute.cpp`, the `fused_tile_*`
kernels) and `blend_device.cpp`, plus the established prior findings
(`opt/sfpu-cull-next.md`, `opt/stage-c2-payload-impl.md`,
`opt/plan-high-utilization-pipeline.md` §0.5-Q4/Q5/Q9, §4, §7).

> **UPDATE 2026-06-01 (iter 14, commit e204f63) — single-program fusion is OFF THE TABLE.**
> Implementing §8.4 as ONE merged cull+blend program hard-faults at `EnqueueMeshWorkload`:
> `TT_FATAL: Program size (92688) too large for kernel config buffer (70656) on TENSIX`.
> The 70656B per-core kernel-config ring-buffer ceiling (= `l1_unreserved_base − KERNEL_CONFIG`)
> is a HARD limit, not tunable without rebuilding tt-metal. Merged compute trisc1 = 59260B
> (cull 38152 + blend 21668); the cull is `noinline` uint-bit-ABI templates that can't be
> cheaply shrunk, and the cull regime must not be retuned. **Do the L1 mask handoff as TWO
> SMALL programs** (cull, then blend) sharing a fixed RESIDENT-L1 mask region, with a `Finish()`
> between enqueues to replace the per-candidate `MB_CULL_SPIN` — sidesteps the ceiling entirely.
> The single-program scaffold is gated off (`GSPLAT_TT_FUSE_BLEND=1`) and kept in-tree.

> **UPDATE 2026-06-01 (iter 15) — TWO-PROGRAM L1 MASK HANDOFF: the spin is NOT a DRAM-settle artifact (§1.2/§1.4 premise FALSIFIED).**
> Implemented the two-small-program L1 handoff: `cull_masks` allocated as an
> L1-interleaved buffer (same 64B page layout + per-tile page-aligned base), cull
> writer writes masks to L1, a host `Finish()` between the cull and blend enqueues,
> blend reader pops from L1. Gated `GSPLAT_TT_L1_MASKS=1` (DEFAULT OFF; baseline
> 63.85 dB / 191 ms preserved). On-device (yyzo-bh-03, 1 view):
> - Masks land **bit-identically** in L1 (`GSPLAT_TT_SFPU_CULL_DEBUG`: 0 CULLMM
>   mismatches, full **63.85 dB** — proves L1 transport + the Finish are correct).
> - But removing the spin gives **30 dB**, and a spin sweep shows the gate is met
>   ONLY at spin≈512, where blend == baseline:
>   `spin 0→30.1dB/136ms, 256→40.4/143, 384→43.7/167, 448→50.9/178, 512→63.85/190.9`.
> - So the per-candidate `MB_CULL_SPIN` is a **reader read-completion window**
>   (the freshly `noc_async_read_tile`'d mask page is consumed too soon after
>   `noc_async_read_barrier()`), **not** a DRAM-bank write-settle artifact. It is
>   UNCHANGED by moving masks DRAM→L1. The `cull_masks` DRAM traffic (~0.4 GB) was
>   already hidden behind the spin + the 1.9 GB random attr gather, so the
>   round-trip removal yields **no blend win** (L1 spin-512 190.9 ms ≈ DRAM 191.4).
> - **Decision: REJECT/gate off.** Baseline held. **Next lever: Stage C2** — the
>   contiguous per-tile payload (`GSPLAT_TT_BLEND_PAYLOAD` scaffold, §3a) makes the
>   blend reader a pure sequential stream (kills the 1.9 GB random gather AND lets
>   masks ride the same stream), which is the actual blend bottleneck — not the
>   mask transport. The spin's true nature (read-completion) should be re-examined
>   there (a sequential payload read may not need the per-candidate barrier+spin at
>   all).

Ground truth carried in (do not re-litigate):
- Blend stage ≈ **196 ms**, reader-bound on one mover RISC (NCRISC).
- SFPU compute is **100% hidden**: `NOBLEND` exec == full exec (the
  `MB_DEBUG_PROF_NOBLEND` variant in `alpha_blend_compute_mb.cpp` proves the MATH
  RISC is starved by the reader, not the bottleneck). Compositing roofline ≈ **7 ms**
  for this scene (plan §7).
- `P_kept ≈ 3.2 M` (gaussian,tile) candidate pairs / **1024 tiles** ⇒ ≈ **3.1 k
  candidates/tile**, ≈ **24.6 k candidates/core** over ~130 cores under LPT.
- Failed attempts (all rooted in the DRAM `cull_masks` round-trip): count-first
  two-pass **27.77 dB** and mask-first gather-skip **28.04 dB** (masks read BEFORE
  the attr-gather barrier ⇒ stale); chunk-level spin reduction **29.10 dB**
  (per-candidate spin is load-bearing for DRAM settle); full L1-fused cull+blend
  **DEADLOCKED** at `CB_MB_COEFF` depth 8 (count pushed AFTER coeff emit).

---

## 1. Blend roofline gap (§8.8 prep) — static breakdown of the ~196 ms reader

The reader-bound blend is `reader_alpha_blend_mb_devcull.cpp`, `MB_RESIDENT` +
`MB_SFPU_CULL` path (the resident pipelined gather, lines 549–705). Per kept
candidate the single mover RISC does the following — **all integer / NoC, no
float** (the cull math already moved to the SFPU pass):

### 1.1 Per-candidate bytes moved and NoC transactions

`issue_chunk_reads()` (lines 113–141) issues, per gaussian `g`:

| read | source SoA | pages | bytes fetched | bytes used | random? |
|------|------------|------:|--------------:|-----------:|---------|
| cov_a | `proj_m_a[g]` | 1 | 64 | 4 | **random** (page `g>>4`) |
| cov_b | `proj_m_b[g]` | 1 | 64 | 4 | **random** |
| cov_c | `proj_m_c[g]` | 1 | 64 | 4 | **random** |
| mean_x | `proj_m_px[g]` | 1 | 64 | 4 | **random** |
| mean_y | `proj_m_py[g]` | 1 | 64 | 4 | **random** |
| opacity | `proj_m_opacity[g]` | 1 | 64 | 4 | **random** |
| colors r/g/b | `proj_m_colors[3g..3g+2]` | 1–2 | 64–128 | 12 | **random** |
| **gather subtotal** | | **7–8** | **~480–576** | **36** | |
| mask window | `cull_masks[base+p]` | 2 | 128 | 4 | seq within tile, but DRAM round-trip |
| **per-candidate total** | | **~9–10** | **~600–700** | **~40** | |

So per candidate the mover issues **~9–10 NoC read transactions** and moves
**~600 B** of which only **~40 B (≈6–7%)** is used — a **~16× page-granularity
over-fetch** because each random 64-B page contributes one used lane (`g & 0xF`,
lines 618–633). Amortized reads add ≈0.06 ids-page reads/candidate
(`load_ids_chunk`, 1 page / 16) plus per-tile `sort_tile_ranges` (1–2) and
`cull_mask_base` (1, line 528), all negligible.

Frame totals (3.2 M candidates):
- **~29 M** gather NoC reads + **~6.4 M** mask NoC reads ≈ **~35 M NoC
  transactions/frame** (≈ **270 k/core**).
- **~1.9 GB** random gather + **~0.4 GB** mask = **~2.3 GB DRAM read traffic/frame**
  to deliver only **~115 MB** of useful payload (3.2 M × 36 B).

### 1.2 The `MB_CULL_SPIN=512` cost

Lines 640–646: per candidate, after the dedicated mask-page barrier, a
`for (volatile int _s = 0; _s < 512; ++_s) {}` busy-wait. `volatile` forces a
load+store+compare+branch each iteration (~4–6 mover cycles) ⇒ **~2.0–3.0 k mover
cycles per candidate** spent doing nothing but letting freshly-written
`cull_masks` DRAM pages settle. This is **per candidate** (×3.2 M), and it is on
the critical path of the one mover RISC. Relative to the ~9–10 NoC issues
(~30–50 cyc each ⇒ ~360 cyc) plus the 16-word coeff emit + `cb_reserve/push`
(~60–80 cyc, lines 683–701), the **spin is the single largest per-candidate mover
cost** in the SFPU-cull path. It is also mandatory (see 1.4) — which is precisely
why removing it requires removing the DRAM round-trip, not tuning the count.

### 1.3 Why SFPU compute (~7 ms) is fully hidden

The compute kernel (`alpha_blend_compute_mb.cpp`, `process_tile_gaussians`, lines
268–309) consumes one `CB_MB_COEFF` row per candidate
(`cb_wait_front`/`cb_pop_front`, lines 278/306) and dispatches `k` SFPU
microblock-blends (`dispatch_blend_guarded`, lines 253–263; `k` = popcount of the
mask). The producer (reader) takes ~600 B + ~10 NoC issues + 512-spin per
candidate; the consumer takes `k`·(~18 SFPU vec-ops) ≈ a few hundred cycles. The
reader is **strictly slower per candidate**, so the MATH RISC blocks in
`cb_wait_front(CB_MB_COEFF)` most of the time. `CB_MB_COEFF` depth 8
(`blend_device.cpp:816`) is a small sliding window; it fills and the SFPU drains
it faster than the reader refills. Hence `NOBLEND == FULL` and the **7 ms SFPU
roofline is entirely buried under the ~196 ms reader**. The bottleneck is
**NoC/issue traffic + spin on one mover**, not compute.

### 1.4 Random vs coalescable, and the mask-vs-prefetch ordering ↔ spin coupling

- **Random (uncoalescable):** the 7–8 SoA gather reads/candidate. Depth-sorted
  `gid`s are uncorrelated with SoA page layout, so each is an independent
  random 64-B DRAM page on a different bank/buffer. This is the **dominant
  traffic** and the thing **Stage C2 eliminates**.
- **Coalescable / sequential:** the `cull_masks` window is sequential within a
  tile (`cull_base + p`, p increasing) and the ids pages are sequential — but the
  masks still pay a **full DRAM round-trip** (written by the cull pass, read back
  by the blend).

**Ordering constraint (lines 584–598).** The mask read must be issued **after**
the attr-gather barrier and given **its own** `noc_async_read_barrier()`. Folding
the 2 small mask reads into the *shared* prefetch barrier that also covers ~144
attr page reads did **not** reliably land them under fast timing → the prefetched
mask returned stale/partial bits → microblocks dropped → **PSNR 30 < keep-all
43.76**. So masks are read strictly after the attrs for the chunk are in L1.

**Why this couples to the spin (and why reorder/reduce broke PSNR to ~28–29 dB).**
Both the ordering rule and the spin are artifacts of `cull_masks` living in
**DRAM** across two programs within one `Finish()`. `noc_async_read_barrier()` on
the blend consumer guarantees the *read* completed, but **not** that the cull
writer's fresh DRAM *write* to the same page has fully settled in the bank
(producer→DRAM→consumer write-after-write/RAW across the round-trip). The
per-candidate `MB_CULL_SPIN` is the empirically-tuned settle window:
- reading masks **before** the attr barrier (count-first / mask-first
  gather-skip) ⇒ the small mask reads ride a barrier that returns early ⇒ stale
  masks ⇒ **27.77 / 28.04 dB**;
- **chunk-level** spin (one spin per ~16 candidates instead of per candidate) ⇒
  insufficient/uneven settle for each freshly-read page ⇒ **29.10 dB**.

So the spin must be *per candidate* and the mask read *after* the attr barrier —
both load-bearing **only because the masks round-trip through DRAM**. The L1 mask
handoff (§3b) removes the round-trip, which removes **both** the ordering hazard
and the spin at once.

---

## 2. Deadlock root cause + the exact ordering invariant that fixes it

### 2.1 The working (non-deadlocking) producer/consumer contract

Today the blend is two kernels handshaking over `CB_MB_COUNTS` (depth 2,
`blend_device.cpp:817`) and `CB_MB_COEFF` (depth 8, `:816`):

Reader `reader_alpha_blend_mb_devcull.cpp`, per tile:
1. push the **candidate count** first — lines 533–538:
   `cb_reserve_back(CB_MB_COUNTS,1); cnt_ptr[0]=L; cb_push_back(CB_MB_COUNTS,1);`
   where `L = id_end - id_start` from `sort_tile_ranges` (lines 484–521).
2. **then** emit exactly `L` coeff rows, **one reserve/push per candidate** —
   `cb_reserve_back(CB_MB_COEFF,1)` … `cb_push_back(CB_MB_COEFF,1)` (lines
   683/701). Every candidate emits one row even if `mask==0` (comment lines
   531–532); the compute dispatches nothing for a zero mask.

Compute `alpha_blend_compute_mb.cpp`, per tile:
1. `cb_wait_front(CB_MB_COUNTS,1)` → read `num_g` (lines 383–389), `cb_pop_front`
   at line 431.
2. loop `num_g` times: `cb_wait_front(CB_MB_COEFF,1)` … `cb_pop_front(CB_MB_COEFF,1)`
   (lines 278/306).

Because **count is pushed before any coeff row**, the consumer learns `num_g`
immediately and then drains `CB_MB_COEFF` one row at a time as the producer fills
it. The depth-8 CB is a sliding window; `L` can be arbitrarily large with no
hang. This is the contract the fused path must preserve.

### 2.2 The deadlock (full L1-fused cull+blend, count pushed AFTER coeff emit)

The fused attempt deferred the count: it culled/emitted the per-tile coeff rows
**first** and pushed a (post-cull "keeper") COUNT only **after** the emit loop.
With `CB_MB_COEFF` depth 8 and any tile where `L > 8` (essentially every tile —
avg 3.1 k):

- **Producer** fills slots 0–7, then blocks in `cb_reserve_back(CB_MB_COEFF,1)`
  for slot 8 — the CB is full.
- **Consumer** blocks in `cb_wait_front(CB_MB_COUNTS,1)` — the count has not been
  pushed yet, so it has not begun popping `CB_MB_COEFF`.
- The count is only pushed *after* the producer finishes emitting all `L` rows —
  which it can never reach because it is stuck at slot 8.

**Circular wait** ⇒ deadlock. Depth 8 just sets the trip point (`L > 8`); the bug
is the **count-after-coeff ordering**, not the depth.

### 2.3 The exact invariant (code-line precision)

> **Per tile, `cb_push_back(CB_MB_COUNTS,1)` MUST strictly precede the first
> `cb_reserve_back(CB_MB_COEFF,1)`, and the pushed count MUST equal the number of
> coeff rows subsequently emitted (= `L = id_end - id_start`, every candidate
> emits one row). Reserve/push `CB_MB_COEFF` ONE row per candidate (never reserve
> `L` up front). The consumer reads the count, then pops one row per candidate.**

Key enabling fact: **the count is knowable before culling.** `L` is the candidate
count from `sort_tile_ranges` (reader lines 484–521 / cull reader
`fused_tile_render.cpp:205–211`), and the design emits **one row per candidate**
regardless of mask (mask==0 ⇒ consumer dispatches nothing). So there is **no need
to defer the count to a post-cull "kept" total** — that deferral was the entire
bug. Concretely, the fused kernel must replicate the working reader order:
push `CB_MB_COUNTS`(=L) for tile `t`, *then* run cull+emit over `t`'s candidates,
one `CB_MB_COEFF` reserve/push each.

Corollary for the in-tree fused scaffold: the cull half already obeys this — the
cull reader pushes `CB_CULL_COUNTS`(=L) before emitting `CB_CULL_COEFF` rows
(`fused_tile_render.cpp:213–220` then `:245–307`), and the cull compute reads the
count first (`microblock_cull_compute.cpp:455–462` then `:480–490`). The fused
**blend** half must do the same with `CB_MB_COUNTS`/`CB_MB_COEFF`. `CB_MB_COUNTS`
depth 2 lets tile `t+1`'s count be staged while `t` drains; keep it ≥2.

---

## 3. Implementation-ready staged spec (each step gate-safe + independently verifiable)

Order matters: **(a) is independent and cannot deadlock; (b) requires the fused
single program; (c) depends on (b).** Each step keeps the §2.3 invariant.

### (a) Stage C2 — per-tile depth-ordered CONTIGUOUS blend payload

**What:** after sort, build one contiguous payload row per candidate so the blend
reader **streams** it instead of doing the 9-page random gather. Cull/`cull_masks`
**unchanged**. Already scaffolded in tree (`payload_pack.cpp`,
`GSPLAT_TT_BLEND_PAYLOAD=1`; `opt/stage-c2-payload-impl.md`).

**Row layout (~36 B used, padded to a 64-B DRAM page):**

| field | bytes | source |
|-------|------:|--------|
| cov_a, cov_b, cov_c | 12 | `proj_m_a/b/c` |
| mean_x, mean_y | 8 | `proj_m_px/py` |
| opacity | 4 | `proj_m_opacity` |
| cr, cg, cb | 12 | `proj_m_colors[3g..3g+2]` |
| (pad / optional `mask`) | →64 | DRAM-aligned page |

This is **byte-identical to the row the current reader emits** into `CB_MB_COEFF`
(lines 685–700: `cov_a/b/c`, `mx_local`, `my_local`, 0, opacity, cr, cg, cb, mask,
0…). Storing `mx_local = mean_x - tx_tile` can be deferred to the reader (keep the
payload tile-independent) or baked per tile.

**Storage:** one contiguous DRAM slab per tile; base = exclusive prefix-sum of
`round_up(per_tile_count[t], 16)` (page-aligned to 64 B), exactly mirroring
`cull_mask_base`. Register `blend_payload` + `blend_payload_base` in
`device_state` (grow-on-demand like `cull_masks`,
`blend_device.cpp:1772–1779`).

**Blend reader change:** replace `issue_chunk_reads` (9 random pages/candidate)
with a **sequential** page stream of the tile's slab (double-buffered, depth 2 —
see §4). Removes ~1.9 GB random traffic ⇒ ~115 MB streamed at near-full DRAM
efficiency, and **~9 NoC reads/candidate ⇒ ~1** (or fewer if multiple rows/page).

**CBs/DST:** unchanged. Reader still feeds `CB_MB_COEFF` (depth 8); compute DST
budget unchanged at **6 fp32 tiles** (R/G/B/T slots 0–3 + X/Y ramps 4–5 ≤ 8).
**§2.3 invariant unchanged** (count pushed before coeff).

**Gate-safety:** payload pass is **additive** (blend keeps using the devcull
reader until the reader variant lands). The reader variant changes only *how* the
identical 36 B/row is fetched (random→sequential), not the blend math ⇒ PSNR is
unchanged by construction. Verify: `GSPLAT_TT_PAYLOAD_VERIFY=1` byte-compares the
payload vs host `build_mb_payload` for one tile; then `a003_verify.py` ×2 at
hero_vs_ref **≥ 63.85 dB**. **Cannot deadlock** — it is a DRAM producer pass plus
a sequential read; no cross-kernel CB cycle is altered.

### (b) L1 mask handoff — single fused program, masks via `CB_TILE_MASKS`

**What:** in one fused program where a core runs cull→blend, hand the packed
32-bit masks from the cull/writer to the blend reader through an **L1 circular
buffer `CB_TILE_MASKS`** instead of writing/reading them through DRAM
`cull_masks`. This kills the `cull_masks` DRAM round-trip **and** the
`MB_CULL_SPIN` (§1.2) **without** the stale-mask hazard (§1.4).

**Producer:** the cull compute already packs per-batch keep flags
(`microblock_cull_compute.cpp` → `CB_KEEP`, `pack_tile` line 542) and the
fused writer assembles the 32-bit mask per gaussian
(`fused_tile_writer.cpp:236–243`, `mask |= 1<<m`). Instead of
`noc_async_write` to `cull_masks` (`:315–320`), `cb_push_back` a **mask page** to
`CB_TILE_MASKS`. Pack **one 32-mask page per BATCH=32** (128 B = 32 × u32), which
matches the cull batch granularity and the depth-sort order.

**Consumer:** the blend reader pops one `CB_TILE_MASKS` page per 32 candidates and
reads `mask[p & 31]` for candidate `p` (replacing `load_mask_page` + the dedicated
barrier + the spin, lines 594–646). Because it is an **L1 CB** with
`cb_wait_front`/`cb_pop_front` fencing, the masks are **present and committed by
construction** — no DRAM settle, **no `MB_CULL_SPIN`**, and the "read after attr
prefetch lands" rule is satisfied automatically (the per-candidate emit reads the
mask at the point where that candidate's attrs are already in L1).

**Ordering invariant within the fused program (per tile):**
1. cull all of tile `t`'s candidates → push `⌈L/32⌉` mask pages to
   `CB_TILE_MASKS`;
2. push `CB_MB_COUNTS`(=L) **before** the first `CB_MB_COEFF` row (§2.3);
3. blend emits `L` coeff rows, popping a mask page every 32 candidates.

Sequence cull-then-blend **per tile** (not all-cull-then-all-blend) so
`CB_TILE_MASKS` need only buffer a tile's masks, not the whole frame.

**CBs + depths (fused program):**

| CB | page | depth | role |
|----|------|------:|------|
| `CB_BOX_OX/OY` (0,1) | 4 KB | 1 | constant box ramps (cull) |
| `CB_CULL_COEFF` (2) | 64 B | 32 | cull coeff rows (reader→cull-compute) |
| `CB_CULL_COUNTS` (3) | 64 B | 2 | per-tile [L,tx,ty] |
| `CB_KEEP` (16) | 4 KB | 4 | cull-compute→writer keep tiles |
| **`CB_TILE_MASKS`** (new) | 128 B | **4–8** | **L1 mask handoff, writer→blend reader** |
| `CB_MB_COUNTS` (3 or new) | 128 B | **≥2** | blend count (§2.3) |
| `CB_MB_COEFF` | 64 B | **8** | blend coeff rows (reader→blend-compute) |
| `CB_OUT` (16) | bf16 tile | 6 (mult. of 3) | R/G/B out |

`CB_TILE_MASKS` depth 4–8 covers cull→blend skew within a tile (the cull runs a
few batches ahead of the blend). If cull and blend share one compute kernel, note
`CB_KEEP`(16) and `CB_OUT`(16) currently collide on CB id 16 — give the fused
program **distinct CB ids** for keep vs color out.

**DST budget (≤8 fp32 tiles):** cull phase uses DR_BOX_OX(0), DR_BOX_OY(1),
DR_KEEP(2), DR_QV(3), DR_QH(4) = **5**; blend phase uses R(0),G(1),B(2),T(3),
X(4),Y(5) = **6**. Run **cull-then-blend per tile** so the slots are *reused*, not
co-resident ⇒ **max 6 live ≤ 8**. (If a future variant overlaps them, it must
stay ≤ 8 — there is no headroom to keep both QV/QH and R/G/B/T live at once, so
do not overlap.)

**Data layout:** masks in `CB_TILE_MASKS` are dense per tile in depth-sort order
(page `b` holds candidates `32b..32b+31`); no `cull_mask_base` page-alignment
needed since it never hits DRAM.

**Gate-safety:** the masks are bit-identical to the DRAM-cull masks (same SFPU
cull math, same `perm(g,m)` pack); only the transport changes (DRAM→L1 CB). With
CB fencing there is no stale read, so dropping the spin is correct by
construction. Verify in two steps: (i) keep `cull_masks` DRAM write **and** the
`CB_TILE_MASKS` handoff, assert they match per candidate (debug compare) on a few
tiles; (ii) remove the DRAM path + spin, `a003_verify.py` ×2 ≥ **63.85 dB**.
Expected: blend drops from ~196 ms toward the gather/issue floor (then §4).

### (c) Drop the cull→blend `Finish()`

Once (b) makes cull+blend a **single program** per core (masks never touch DRAM),
there is **no cross-program data dependency** to fence. Remove the inter-program
barrier:
- legacy path runs `cull::process_frame` with its own `Finish()`
  (`blend_device.cpp:1902`) then the blend program — this whole split disappears;
- the fused path already uses a **single** `Finish()` for the cull+blend
  workloads (`:2188–2190`), but they are still **two programs**; collapsing them
  into one program removes the second `EnqueueMeshWorkload` and leaves exactly one
  `Finish()` at frame end before readback (`:2190–2194`).

**Gate-safety:** safe **only with (b)** — the L1 CB ordering inside the single
program (`CB_TILE_MASKS`, `CB_MB_COUNTS`→`CB_MB_COEFF`, §2.3) now enforces every
ordering the DRAM `Finish()` used to. Dropping the barrier *without* (b) would
re-expose the stale-mask hazard. Verify PSNR unchanged ×2 and that frame time
drops by the removed barrier bubble.

---

## 4. Path to the ~7 ms roofline (post-fusion profiling + strips)

After (a)+(b)+(c), the reader's random gather and spin are gone; the SFPU should
become the bottleneck (the *correct* place). Remaining overheads to profile and
strip:

1. **Double-buffer the payload read (depth-2).** Make the C2 sequential payload
   read depth-2 so the DRAM read of chunk `K+1` overlaps the SFPU composite of
   chunk `K` (the resident reader already double-buffers attrs, lines 568–612;
   carry that into the payload-streaming variant). Target: reader never stalls the
   MATH RISC.
2. **Bound the 32-way microblock unroll to the AABB sub-range (§0.5-Q4).** The
   cull already computes `mx_lo..mx_hi, my_lo..my_hi`
   (`compute_microblock_mask` lines 215–222). Carry the bbox range into the blend
   and use **templated range dispatch** so `dispatch_blend_guarded`
   (`alpha_blend_compute_mb.cpp:253–263`) emits bit-tests only for the few
   microblocks in the box, not all 32. **Measure avg `k` = popcount(mask) first**
   — if `k≈2–6`, ~26–30 of the 32 bit-tests/candidate are dead.
3. **LPT load balance.** With tiles (1024) ≫ cores (~130) the greedy
   decreasing-count work queue is near-optimal; the residual is the single
   heaviest tile. Confirm per-core idle waiting on the max tile; chunk only a
   dominant/L1-overflow tile (last resort — loses transmittance early-out, plan
   §0.5-Q3).

**On-device measurements to capture once bh-30 is back:**
- avg/95p **`k`** (microblocks touched per kept candidate) — popcount histogram of
  the masks (instrument the cull writer or blend reader);
- **DRAM read bytes/frame** before vs after C2 (expect ~2.3 GB → ~0.12–0.2 GB) and
  **NoC reads/candidate** (~9–10 → ~1);
- **blend ms** with the spin removed, and PSNR (must hold 63.85 dB);
- **MATH-RISC busy %** and `NOBLEND` vs `FULL` divergence — once `FULL > NOBLEND`,
  the SFPU is finally the bottleneck (approaching the 7 ms roofline);
- **LPT skew**: per-core candidate-count distribution, max/avg ratio, idle time on
  the heaviest tile;
- **per-stage exec**: C2 payload-pack ms, fused cull+blend ms, the removed-barrier
  bubble (§c).

---

## Appendix — program boundaries, shared DRAM, env gate (carried from scaffold)

| Kernel | Role |
|--------|------|
| `fused_tile_render.cpp` (RISCV_1) | per tile: ranges → ids → SoA gather → cull coeff rows; **v2**: after cull batches, stream blend coeff rows from **L1 masks** (no DRAM `cull_masks`) |
| `microblock_cull_compute.cpp` (TRISC) | SFPU cull math → `CB_KEEP` (v2: fold blend SFPU into the same compute, cull-then-blend per tile) |
| `fused_tile_writer.cpp` (RISCV_0) | **v1**: pack masks → `cull_masks` DRAM (scaffold); **v2**: push masks → `CB_TILE_MASKS` (L1) + commit per-tile bf16 RGB → `res_out` |

**Shared resident DRAM:** inputs `proj_m_{a,b,c,px,py,opacity,colors}`,
`sort_sorted_ids`, `sort_tile_ranges`; scratch `res_xramp/yramp`, `res_out`,
`tile_ids`, (`cull_masks`/`cull_mask_base` removed by §3b),
**`blend_payload`/`blend_payload_base`** added by §3a.

**Env / gate:** `GSPLAT_TT_MB_DEVCULL=1`, `GSPLAT_TT_RESIDENT_BLEND=1`,
`GSPLAT_TT_SFPU_CULL=1`, `GSPLAT_TT_FUSED_TILE=1` (+ `GSPLAT_TT_BLEND_PAYLOAD=1`
for §3a). Promote each step only after hero_vs_ref **≥ 63.85 dB ×2** on 30-view
(`a003_verify.py`), recorded in `opt/metal-iters.jsonl`. Device hygiene and the
bh-30 reset procedure: `opt/sfpu-cull-next.md`.

---

## UPDATE 2026-06-01 (iter 17, commit 3b54b2e) — §3a payload path: 11→43 dB partial fix, BANKED gated-OFF

**Status: `GSPLAT_TT_BLEND_PAYLOAD` defaults OFF.** Baseline restored and verified:
`hero_vs_ref=63.85 dB`, `blend=191.0 ms`, `ms/view=391.4` (gate PASS, DoD DONE).
The payload scaffold (`payload_pack.cpp` + `reader_alpha_blend_mb_payload.cpp`) and
the sizing fix below stay in-tree but are inert unless the env var is set. All
diagnostic probes (rowck/rdump in `alpha_blend_compute_mb.cpp`, the `GSPLAT_TT_ROWCK`
wiring + payload-coverage readback in `blend_device.cpp`) were removed; the
`GSPLAT_TT_DIFF_DUMP` reporting helper in `a003_verify.py` stays but is env-gated.

When ON, the payload path runs `blend≈26 ms` (the ~7× win, near the ~7 ms roofline)
but plateaus at **~42–43 dB** (was **11.23 dB** before this fix). Not landable yet.

### What was FIXED (11 → 43 dB): payload buffer was undersized vs the PADDED id space

Root cause was **not** a reader/writer tile-sync / push-pop mismatch (those match the
proven devcull reader exactly — per-tile `num_g` is identical, verified via an
order-independent compute-side checksum). It was a **DRAM buffer-sizing OOB**:

- The pack and reader index `blend_payload` by the **global candidate index**
  `id_start[t] + p`, where `id_start` comes from resident `sort_tile_ranges`.
- The resident sort (`sort_device.cpp`) publishes a **page-aligned PADDED layout**:
  `padded_cursor += round_up(cnt, 16)` per tile, so `max(id_end) == padded_cursor`,
  which exceeds the dense kept count `sort_P_kept[0]` by up to `~num_tiles*15`.
- `ensure_payload_buffer` sized the buffer to `round_up(P_kept,16)` (even +4096 pad),
  leaving it short by the per-tile padding sum. Pack then wrote payload pages PAST
  the buffer end; via the interleaved DRAM accessor those OOB pages landed in the
  next-allocated buffers (`res_xramp`/`res_yramp`/`res_out`), corrupting the
  coordinate ramps **every timed frame** → the original 8-px vertical stripes +
  bottom-block garbage. (Confirmed: forcing a per-frame ramp re-upload masked it
  10.85→26.73 dB; the real fix removed the OOB entirely.)
- **Fix** (`blend_device.cpp` `ensure_payload_buffer`, ~L2125): size `blend_payload`
  to exactly one 64B row per `sort_sorted_ids` element
  (`rows = buf_ids->size()/sizeof(uint32_t)`), which is allocated to exactly
  `padded_cursor` — i.e. the true index space. 26.73 → **42.09 dB**, ramps clean,
  no per-frame re-upload needed.

### What the REMAINING ~43 dB residual is

- **Spatial pattern changed**: no longer global vertical stripes. Now a **diffuse,
  density-correlated error with faint radial streaks** spread across many/most tiles
  (`diff10x.png` via `GSPLAT_TT_DIFF_DUMP=1`). Error is broad, not localized to a
  band — consistent with a *systematic* per-candidate data difference, not OOB.
- **Coverage is complete**: payload readback found `read_sentinel=0`,
  `bad_tiles=0/1024` — pack writes every page in each tile's `[id_start,id_end)`
  read range. So no missing/unwritten rows.
- **Counts match**: per-tile `num_g` (the `L` the reader streams) is identical to
  devcull for every tile.
- **Values diverge for EVERY tile**: an order-independent (commutative-XOR) per-tile
  checksum of the coefficient row data (`row[0..9]`) AND of the mask (`row[10]`)
  **differs** between the payload and devcull paths on **all 1024 tiles** — i.e. it
  is not an ordering shift, the actual packed values differ everywhere.
- **But most candidates are byte-identical**: a per-row dump of tile 0 (`RDUMP`)
  showed `g=0,2,3` **byte-identical** to devcull, while `g=1` had divergent
  cov/mean/op (`row[0..6]`, e.g. `a=53.4` vs `18558.4`). (Caveat: single-tile dump,
  some run-to-run noise observed in which candidate lands at a given `g`.)
- Ruled out: SFPU-vs-RISC mask math (devcull with `SFPU_CULL=0` RISC masks = 63.85,
  so `compute_microblock_mask` is fine); pack double-buffer overlap (`CB_SCR_ATTR`
  is correctly sized to `2*16*GATHER_FIELDS`=288 pages=18432 B in `blend_device.cpp`
  ~L2069, exactly the ping-pong footprint); sort-publish race (single shared CQ).

### Single best hypothesis + exact next experiment

**Hypothesis:** pack's read window into `sort_sorted_ids` is offset from the devcull
reader's window by the **padded-vs-dense index convention** — the same padded-layout
sharp edge that caused the sizing bug. "Count matches but values differ on every
tile" is exactly the signature of two paths reading **different (but equally long)
slices** of `sort_sorted_ids`: if pack uses the page-aligned padded `id_start` from
`sort_tile_ranges` while the devcull reader effectively indexes the dense cursor (or
vice-versa), then for every tile past tile-0 the accumulated 16-pad offset shifts
pack's gids to a neighbor's candidates — valid-looking gaussians with wrong
cov/mean/op, producing the diffuse density-correlated error.

**Next experiment (decisive, ~1 iter):** DPRINT the raw gid pack actually packs
(`gids[cur][j]` in `payload_pack.cpp`, first 4 of tile 0 AND tile 1) and compare to
the gid the **devcull reader** gathers for those same two tiles. If they match on
tile 0 but diverge on tile 1 by tile-0's `round_up(cnt,16)-cnt` padding, the index
conventions are mismatched → reconcile pack to read `sort_tile_ranges`/
`sort_sorted_ids` with the identical convention the devcull reader uses (confirm
whether `id_start` is meant to be the padded page base or the dense cursor, in
`sort_device.cpp`'s publish). If gids match on both tiles, fall back to: write pack's
assembled `row[0..10]` for tile-0 candidates to a scratch DRAM buffer and host-diff
against the devcull reader's row for the same candidates, field by field, to localize
the systematic divergence to a specific `row[]` slot.
