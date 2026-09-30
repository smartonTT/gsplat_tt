# Assessment — `kinginu/tt-splat` (matrix-native 3DGS) vs our pipeline

**Written 2026-09-30.** Read-only study; no rendering code changed.
Source studied: `github.com/kinginu/tt-splat` @ `e518b1a` (README, `docs/bh-port-math.md`,
`spike/` PyTorch oracle, `tools/` ttnn device path, `outputs/bench/*.json`).
Our side: `docs/CONTEXT.md`, `opt/FINAL-REPORT.md`, `opt/sort-l1-resident-plan.md`,
`render/` sources.

---

## 1. TL;DR

tt-splat rewrites the *math* of 3DGS so the hot path is a matmul: the Mahalanobis field
becomes an exact GEMM `Q[P×G] = Φ[P×6]·θᵀ[6×G]`, `exp` becomes the polynomial
`w = o·(1−Q/k)₊²`, and depth-sorted alpha becomes order-independent Weighted-Sum Rendering
`C = Σwc/Σw`. Forward and backward collapse to `GEMM → pointwise → GEMM`.

**Their headline optimisation targets a resource we are not bound on.** Their thesis is
"SFPU is 64× slower than MVMUL, so get off the SFPU". Our measured profile is the opposite:
TRISC/SFPU is **~52 ms/view and off the critical path**, while **NCRISC dataflow is the
saturated pole at ~125-131 ms** (`docs/CONTEXT.md` §"Per-stage cost"). A pure
`exp`→polynomial / SFPU→FPU swap therefore buys us ~0 ms of frame time and costs us the
pre-trained `.ply` (their own README: "pre-trained public `.ply` assets do not transfer").

**What is genuinely valuable to us is their *data structure*, not their arithmetic.** Two
tt-splat design choices attack exactly the two blockers that killed the only remaining
structural lever in `opt/FINAL-REPORT.md` (the Stage-2 cross-stage per-tile pipeline, iter-141
NO-GO):

| our blocker (iter-141) | tt-splat's answer |
|---|---|
| the gaussian→tile transpose is gated by a **global prefix-sum barrier** → 0 ms pipelinable per-tile | **allocate-once fixed-capacity per-tile bucket, offsets from an atomic increment** — no prefix sum, no global barrier |
| the fused double-buffered kernel fits L1 (1.5 MB) only on a **~4% knife-edge** and blows it on overflow tiles | **tile-local coordinates make the per-pair payload bf16-safe** → halve the record, double the L1 headroom |

That reopens the lever the project froze on. It is the reason this repo is worth reading.

**Honest scoreboard.** Their `playroom` run is 1264×832 = 1.051 Mpx at **14.6 fps
(≈68 ms/view)**, G-independent `O(P·K)`, quality 20.11 dB held-out vs gsplat's 29.24.
Ours is 1024×1024 = 1.049 Mpx at **173.3 ms/view (5.8 fps)** at full bicycle (~6.1 M
gaussians, `scenes/bicycle.ply` = 1.52 GB), bit-identical to golden. Same pixel budget,
they are 2.5× faster with a ~9 dB quality deficit and a 60× smaller model — and they got
there in pure `ttnn` Python ops (no Metalium kernels anywhere in the repo; `generic_op`
appears only in the README as future work). The structural lesson is not their absolute
number, it is the **shape**: per-view cost `O(P·K)`, independent of G. Ours is
`O(pairs) = O(G_visible × fan-out)`.

---

## 2. What tt-splat actually is (and is not)

| claim | status in their repo | reusable by us? |
|---|---|---|
| `Q = Φθᵀ` exact GEMM (monomial expansion) | derived + verified in `spike/forward.py::quad_form_tilelocal`, fp64-exact | **yes, algebra only** — no retraining needed |
| tile-local pixel coords (fold tile origin into μ) mandatory for bf16; error 219 → 0.04 | verified `spike/tests/test_tile_local.py` | **yes** — precision argument transfers directly |
| polynomial splat `(1−Q/k)₊²` instead of `exp` | trained against | **model change** — needs retraining, or a fitted approximation |
| Weighted-Sum Rendering (no depth sort) | trained against | **model change**, and it loses occlusion |
| fixed-capacity per-tile budget K, overflow dropped, allocate-once | implemented (host + dense ttnn) | **yes, design only** |
| single-destination scatter (R=1) + neighbourhood gather, "lossless at R=1, K=128" | measured on their scenes | **design, with a caveat** (see §5.2) |
| NoC-atomic fixed-capacity scatter kernel | **not implemented** — README WIP | design only |
| device-resident training, MCMC density control, Adam on-card | implemented | **out of scope** (we are inference-only) |
| their own on-device binning | **loses to host binning** (287 vs 201 ms/it) because it is dense `O(T·G)` | do **not** copy the implementation |

Everything they ship is `ttnn` op-graph Python (`ttnn.matmul` ×117, `ttnn.embedding` ×27,
`ttnn.execute_trace` ×47). We are a hand-written Metalium renderer. **No code transfers.
Only ideas.**

Also worth noting: their quality numbers are matched-config and honest, and they are bad
for the solid-geometry case — lego `−17.9 dB`, playroom `−9.1 dB` vs gsplat at matched
resolution and gaussian count. bicycle is a real outdoor scene with hard occluders. WSR as
shipped is not a viable blend for our reference workload.

---

## 3. Which techniques are adoptable incrementally vs need retraining

### 3.1 Adoptable now, no retraining, algebra-preserving
1. **`Q = Φθᵀ` reassociation.** It is an identity. Our blend already computes
   `power = A·dx² + B·dx·dy + C·dy²` from a pre-folded conic
   (`render/kernels/compute/alpha_blend_compute_mb.cpp:132-134`, conic hoisted per-gaussian
   in `pfwc_compute` at iter-111). Rewriting it as `Φ(x,y)·θ(g)` with `Φ` precomputed once
   per microblock lets the whole `[pixels × gaussians]` Q field be produced by
   `matmul_tiles` on the FPU instead of the SFPU. Not bit-identical (different summation
   order / fp32 dst accumulation), but no model change.
2. **Tile-local-coordinate precision argument → shrink the per-pair record.** Their measured
   bound is the missing justification for a change we previously rejected: the 32 B PACK2
   record keeps covariance in **full fp32** because "the blend recomputes the conic via
   `det = a*c - b*b`, which loses too much to fp16"
   (`render/kernels/dataflow/sort_bin.cpp:392-400`). **That reason no longer exists** — the
   det/reciprocal was hoisted out per-gaussian at iter-111. The record now carries the
   already-inverted conic `A,B,C` plus a tile-local mean `mx_local,my_local` (already
   tile-local, exactly tt-splat's form). Those are all O(1)-scaled quantities in tile-local
   space, so fp16 is defensible. We already have exact fp32→fp16 RNE conversion in the same
   file (`sort_bin.cpp:49`).
3. **Fixed-capacity bucket with atomic-increment offsets.** Pure data-structure change.
   Our kernels today deliberately avoid atomics — the prefix-sum exists precisely so
   "cores never share a slot — no race, no atomics"
   (`render/kernels/dataflow/sort_bin.cpp:493`). Swapping to a NoC atomic fetch-and-add per
   tile counter removes the global barrier. **Can stay bit-identical**, because the per-tile
   depth radix (`sort_radix_tile`) re-canonicalises order afterwards — provided the sort key
   is made a *total* order (depth_key, gaussian_id) so ties do not depend on arrival order.
4. **SoA/tilized θ operand layout** instead of AoS 32 B records. Prerequisite for (1); its
   independent value is fewer, larger NoC transactions and no scalar unpack in the data mover.

### 3.2 Needs retraining (new `.ply`, new golden, no comparison against standard assets)
- **Polynomial splat as a model** (`w = o·(1−Q/k)₊²`). Their gaussians are *fit* to it.
  Applying it to a standard-trained `.ply` changes appearance.
  - *Middle path that does not need retraining:* keep the model, replace the **evaluator** —
    fit a polynomial/rational approximation to `o·exp(−Q/2)` over the 3σ support and hold the
    quality gate. We currently use a 21-bit-accurate `_sfpu_exp_21f_bf16_`; we only need
    ~50 dB on 8-bit output. This is an approximation, not a model change.
- **Weighted-Sum Rendering.** Requires retraining *and* accepts the occlusion loss. Their own
  matched numbers (−9.1 dB playroom, −17.9 dB lego) rule it out for bicycle as-is. Their WIP
  "revealage" term (lego 17.37 vs sorted 17.25, sort-free) is the only version worth watching,
  and it is unproven at high resolution.
- **MCMC fixed-count density control**, on-device training, Adam: training-only, out of scope.

### 3.3 Not adoptable
- **16×16 tiles** (`P=256`). They use them to keep `px² ≤ 225` and to fit the GEMM. Our 32×32
  tile matches the Tensix output tile; halving it quadruples tile count (1024 → 4096) and
  multiplies our fan-out, which is our bound.
- **Dense `O(T·G)` binning.** Their own measurement says it loses to CPU binning. Our on-device
  tile assign is already sparse and balanced.

---

## 4. Expected effect on NCRISC data movement, and on fidelity

### 4.1 NCRISC
The pole is NCRISC at ~125-131 ms/view. Its named zones (iter-131 Tracy, per-view makespan):

| zone | ms | character | scales with |
|---|---:|---|---|
| `sort_bucket_emit` | 30.6 | **store-bound, on the critical path**; "per-pair 32 B L1 store floor" | pairs × record bytes |
| `tile_blend_load` | 28.1 | ~98% SFPU-backpressured (**shadowed = effectively free**) | pairs × record bytes |
| `rd_l1_bulk` | 27.9 | ~98% backpressured (shadowed) | pairs × record bytes |
| `tile_l1_cull_rd` | 22.0 | ~98% backpressured (shadowed) | pairs × record bytes |
| `sort_subchunk_mat` | 19.6 | overflow gather + permute, partly shadowed | pairs × record bytes |
| `ta_bucket_scatter` | 12.2 | tile_assign K2 store | pairs |
| `ta_gauss_aabb` | 9.7 | per-gaussian AABB | G_visible |

Two things follow, and they set the whole ranking:

- **Only the *store* side of the pair traffic is on the critical path.** ~78 ms of NCRISC
  reads are already hidden behind SFPU backpressure. So **moving work from stores to reads is
  free, and halving read bytes is worth ~0 ms.**
- **Every store term is `pairs × bytes_per_pair`.** Both factors are attackable, and tt-splat
  supplies one idea for each: fp16 tile-local payload (bytes) and single-destination
  scatter + neighbourhood gather (pairs).

Expected direction (all unmeasured — see §6):
- record 32 B → 16 B: `sort_bucket_emit` is at its 32 B store floor, so ~**−15 ms** there
  plus a few ms of `sort_subchunk_mat`; reads gain nothing (shadowed). Frame ~173 → ~155.
- pairs cut by the fan-out factor: `sort_bucket_emit` and `ta_bucket_scatter` scale down
  proportionally; the cost lands on the cull/blend **read** path (shadowed) and on SFPU cull
  (52 ms, with ~70 ms of headroom before it becomes the pole). Upside depends entirely on
  the mean fan-out, which we have never measured (we only know the **median is 4**, iter-117).
- atomic bucket append: 0 ms by itself; it unblocks pipelining up to ~44 ms of NCRISC
  (`sort_bucket_emit` ~32 + materialize overflow ~12) under the ~51 ms cull+blend SFPU
  shadow. `opt/FINAL-REPORT.md` bounds that at the ~125 ms NCRISC pole (~20% best case).

### 4.2 Fidelity — where our bar has to move, stated plainly
Today's gate is `hero_vs_ref = 100.00 dB`, i.e. **bit-identical** to
`tests/fixtures/hero/hero_golden_8bit.png`. That bar cannot survive this line of work, and
pretending otherwise would stall it.

- **Survives bit-identical:** atomic bucket append (with a total-order sort key), tilized
  operand layout, pure scheduling/pipelining changes, and a two-tier R=1 gather *if* the
  neighbourhood is provably sufficient for every contributor it must find.
- **Breaks bit-identical, small and boundable:** fp16 per-pair payload, `Φθ` reassociation,
  a fitted `exp` replacement. These change the last bits of `power`/`alpha`. tt-splat's
  measured tile-local bf16 error is ~0.04 absolute on an O(1) `Q` (≈4% weight error) —
  which is why the recommendation is **fp16 (10-bit mantissa), not bf16** (~0.5%).
- **Breaks it structurally:** WSR, polynomial splat as a model, any retraining.

The project's own policy already has the escape hatch: *"If a future iteration legitimately
changes output and is accepted, refreeze the golden and add a new `NEW REF` divider row"*
(`opt/ttw/knowledge/context.md`). The recommendation is to **refreeze once, deliberately, at
the start of this line of work**, and to re-gate on the existing `hero_vs_cpu` float PSNR
(63.95 dB today) as the real fidelity anchor — because that metric measures divergence from
the CPU oracle, which is what fp16/reassociation actually perturbs, whereas
bit-identical-to-golden only measures "did anything change at all".

---

## 5. Ranked candidate optimisations

Ranked by (expected ms on the **pole**) × (confidence) ÷ (risk). Every number is an
expectation to be measured on the p150 IRD reservation with the 30-view bicycle bench —
nothing here has been measured.

### 5.1 — Shrink the per-pair record 32 B → 16 B (fp16 tile-local payload)
- **Mechanism.** tt-splat's tile-local-coordinate result says an O(1)-scaled conic + local
  mean is safe in half precision; our fp32 requirement was justified by a per-pair
  `det = a·c − b·b` inversion that **no longer runs** (hoisted per-gaussian at iter-111,
  `alpha_blend_compute_mb.cpp:121-124`). New record: `A,B,C,mx_local,my_local` as fp16
  (10 B) + depth_key (4 B) + op/color UNORM16 (already packed, drop to one word) → 16 B.
  PACK2 becomes PACK4 (4 records per 64 B page).
- **Moves.** `sort_bucket_emit` 30.6 ms (explicitly at the "per-pair 32 B L1 store floor"),
  `sort_subchunk_mat` 19.6 ms, and the L1 footprint of every bucket CB.
- **Secondary, and possibly the bigger prize:** it converts the Stage-2 pipeline's L1
  "~4% knife-edge that blows the budget on overflow tiles" into ~2× headroom, i.e. it
  removes one of the two iter-141 NO-GO blockers.
- **Risk.** Low, localised (`sort_bin.cpp` pack + the three readers' unpack). Not
  bit-identical → refreeze golden, gate on `hero_vs_cpu`.
- **Expected:** −10 to −18 ms frame; L1 per-tile footprint halved.

### 5.2 — Single-destination scatter + neighbourhood gather (cut the pair count)
- **Mechanism.** tt-splat scatters each gaussian to **one** cell and has each tile gather a
  fixed budget from a radius-R neighbourhood ("measured lossless at R=1, K=128"). For us
  this trades an on-critical-path **store** for an off-critical-path, 98%-backpressured
  **read** — exactly the direction our profile wants. Two-tier: gaussians with 3σ radius
  ≤ one tile take the R=1 path; the large tail keeps today's AABB fan-out.
- **Moves.** `sort_bucket_emit` 30.6 and `ta_bucket_scatter` 12.2 scale with the removed
  pairs. Cost lands on `tile_l1_cull_rd`/`rd_l1_bulk` (shadowed) and SFPU cull (52 ms, with
  ~70 ms of headroom to the pole).
- **Blocking unknown.** Pairs are dominated by *large* gaussians (a splat spanning 10×10
  tiles emits 100 pairs), and we have only ever measured the **median** fan-out (4), never
  the mean or the tail. **Prerequisite: a pairs-per-gaussian histogram** (same DPRINT
  technique as the iter-139 overshoot histogram). If the mean fan-out is ~4, this is a
  ~20 ms lever; if pairs are tail-dominated, it collapses to a few ms and should be dropped.
- **Risk.** Medium. Can be bit-identical if the two-tier split is conservative.
- **Expected:** −0 to −25 ms frame, decided by one cheap measurement.

### 5.3 — Atomic fixed-capacity bucket append (delete the global prefix-sum barrier)
- **Mechanism.** Replace the prefix-sum-derived, race-free slot allocation
  (`sort_bin.cpp:493`, `sort_bin_emit.cpp` Pass-2 checkpoints) with a per-tile NoC atomic
  fetch-and-add into an allocate-once fixed-capacity bucket (we already have
  `kBucketFit = 8192`, `render/host/config.h:36` — only the *offset derivation* changes).
  Overflow handling stays as today rather than tt-splat's "drop".
- **Moves.** 0 ms directly. It converts `sort_bucket_emit` from "**0 ms pipelinable**,
  global transpose behind a prefix-sum barrier" (the exact iter-141 NO-GO finding) into a
  per-tile-completable stage, which is the precondition for hiding ~44 ms of heavy NCRISC
  under the ~51 ms cull+blend SFPU shadow. iter-140 already proved the NCRISC↔SFPU overlap
  mechanism (fused makespan = max, not sum, 0 hangs).
- **Risk.** Medium on correctness (needs a total-order `(depth_key, gaussian_id)` sort key to
  stay deterministic), low on quality, and it is the one candidate that can remain
  bit-identical. Confirm the exact tt-metal atomic API on-device (tt-metal is not vendored
  in this tree).
- **Expected:** 0 ms alone; enables a bounded ~20% (≈−35 ms) with 5.1 supplying the L1
  headroom. **5.1 + 5.3 together are the reopening of the frozen lever.**

### 5.4 — `Q = Φθᵀ` as `matmul_tiles` on the FPU, with a tilized θ operand
- **Mechanism.** Precompute the six monomials `Φ = [x², y², xy, x, y, 1]` once per microblock
  (our X/Y ramps are already resident in dst: `DR_X`, `DR_Y`), fold `A,B,C,mx,my` into the
  per-gaussian 6-vector `θ`, and produce the whole Q field with `matmul_tiles` (contraction 6,
  padded to 32 — tt-splat notes this fat-shallow shape runs at ~3-6× elementwise, not the
  full 16×). The alpha-compositing recurrence stays serial on the SFPU, reading Q instead of
  computing it: ~10 of ~18 SFPU ops per (gaussian × microblock) disappear.
- **Moves.** `tile_blend_sfpu` 28.8 + `tile_mb_mask` 22.4 — **both off the critical path
  today.** Its near-term value is the *operand layout*: a tilized `[32×32]` θ face is one
  ~2 KB NoC transaction per 32 gaussians instead of 32 record reads, and the data mover stops
  doing scalar unpack.
- **Why it is ranked 4th, honestly.** Shrinking SFPU is currently *counterproductive* for
  5.3, which wants a large SFPU shadow to hide NCRISC under. This is the **endgame** lever:
  it only pays once NCRISC has been driven below ~52 ms by 5.1-5.3. It is also the direct
  route to tt-splat's real prize — a per-view cost of `O(P·K)` that is independent of G.
- **Risk.** High effort (blend restructure), low quality risk.
- **Expected:** ~0 ms now; the main lever once the pole flips to SFPU.

### 5.5 — Replace `exp` with a fitted polynomial (evaluator, not model)
- **Mechanism.** tt-splat's `(1−Q/k)₊²` is a *model* change. The no-retraining version is to
  fit a low-degree polynomial or rational approximation to `o·exp(−Q/2)` over our 3σ support
  (`kKCap = 3.0`, `render/host/config.h:60`) and evaluate it with FPU muls/adds instead of
  `_sfpu_exp_21f_bf16_`. We need ~50 dB on 8-bit output, not 21-bit accuracy. Compact support
  also makes the 3σ truncation exact rather than an approximation.
- **Moves.** One SFPU transcendental per (gaussian × microblock) inside `tile_blend_sfpu`
  (28.8 ms, off-path).
- **Risk.** Low, self-contained, easy to bound (sweep degree vs `hero_vs_cpu`).
- **Expected:** ~0 ms frame now. Do it only as a prerequisite for 5.4 / the all-FPU blend.

### 5.6 — Order-independent blending (research spike only)
- **Mechanism.** WSR, or tt-splat's WIP "revealage" variant, deletes the depth sort and with
  it the per-tile radix, the depth key in the record, and the ordering constraint that forces
  the gaussian→tile transpose to be global in the first place.
- **Moves.** Potentially `sort_tile_depth` 7.2 + the depth key (4 of 16 B) + the entire
  ordering dependency — the largest structural prize on the list.
- **Why last.** It requires **retraining bicycle against our own renderer**, produces a `.ply`
  no standard viewer can display, and their own matched measurements show −9.1 dB (playroom)
  to −17.9 dB (lego) against sorted alpha. Our reference workload is a real outdoor scene with
  hard occluders. This is a "watch their revealage work, do not port WSR" item.
- **Expected:** not a 2026-H2 lever. Revisit if their revealage result holds at high resolution.

---

## 6. What must be measured before acting (all on the p150 reservation, 30-view bicycle)

1. **Pairs-per-gaussian distribution** (mean, p50, p95, max) — gates 5.2 entirely. Cheap
   DPRINT histogram, same pattern as iter-139.
2. **Re-baseline 173.3 ms on a p150.** Every number in this document traces to Blackhole
   **P100** (`yyzo-bh-07`). p150a is a different part (120 Tensix vs P100's grid); the
   NCRISC/SFPU balance that drives this whole ranking may shift.
3. **fp16 payload quality sweep** — `hero_vs_cpu` for fp16 vs bf16 vs fp32 payload, before
   committing to a golden refreeze.
4. **Confirm the tt-metal NoC atomic fetch-and-add API** on the device host (tt-metal is not vendored in this tree).

## 7. Files and links

- tt-splat: `github.com/kinginu/tt-splat` @ `e518b1a`; math: `docs/bh-port-math.md`;
  oracle: `spike/forward.py`, `spike/arms.py`; viewer: `github.com/kinginu/tt-splat-viewer`.
- WSR paper: arXiv 2410.18931. MCMC density control: `ubc-vision/3dgs-mcmc`.
- Ours: `docs/CONTEXT.md` (baseline + zone table), `opt/FINAL-REPORT.md` (the freeze),
  `opt/sort-l1-resident-plan.md` (refuted-premises ledger),
  `render/kernels/dataflow/sort_bin.cpp` (record pack),
  `render/kernels/compute/alpha_blend_compute_mb.cpp` (blend inner loop),
  `render/host/config.h` (`kBucketFit`, `kKCap`).
