# Blend + Cull speedup plan (SFPU dependency-stall)

Status: PARKED behind fusion. Owner: supervisor. Evidence: on-device
ablation captured 2026-06-08 (see "Evidence" below).

> **READ FIRST — frame priority as of iter 132:** the frame is **BRISC-FW /
> host-dispatch bound (~161.7 ms)**; **TRISC (all SFPU) is ~52 ms, OFF the critical
> path**. Blend SFPU micro-opt therefore **cannot move the frame** until program
> fusion + a transmittance-early-out blend flip the bottleneck onto SFPU. The
> prioritized roadmap, the 1 ms north-star answer, parked levers, and the
> refuted-premises ledger (incl. "cull+blend SFPU ~175 ms is the killer" — REFUTED
> iter 116, RE-TEST if the blend algorithm changes) live at the top of
> **`opt/sort-l1-resident-plan.md`**. This blend plan stays valid as the design for
> roadmap step #3 (blend reader-cluster / phasing rework), to execute AFTER fusion.

## TL;DR

Blend is **100% SFPU-bound and dependency-stall-bound**, not reader-co-limited
and not throughput-bound. The single lever is **instruction-level parallelism
(ILP)**: restructure the blend inner loop into a *phased* dispatch (the pattern
the cull kernel already uses), so independent work fills the SFPU pipeline
instead of stalling on a serial dependency chain. Reader/bandwidth/exp-cost
optimizations are proven dead ends.

## Evidence (ablation, ttw render_clean, 30 views, 2026-06-08)

Per-view `tile_blend_sfpu` makespan under selective neutralization:

| config   | what it removes                    | blend ms/view | meaning |
|----------|------------------------------------|---------------|---------|
| baseline | nothing                            | 30.5          | production blend |
| NOMATH   | all SFPU dispatch                  | **0.63**      | reader floor — data movement is ~free |
| NOREAD   | NoC payload reads                  | 19.6          | SFPU still dominates (stale masks ⇒ lower bound) |
| NOEXP    | the SFPU exp (~12 of ~45 instr)    | 29.2          | exp is only 1.25 ms (4%) |

Conclusions:
- **Reader is ~98% idle.** Kill SFPU math ⇒ whole blend = 0.63 ms. The reader's
  "29.6 ms" is `cb_reserve_back` backpressure-wait. Faster reads / wider
  transfers / double-buffer tuning buy **nothing**.
- **Blend = 30 ms is pure SFPU.**
- **SFPU is stall-bound, not throughput-bound.** Removing 27% of the
  instructions saved 4% ⇒ most cycles are spent stalled on the per-pair
  dependency chain, not issuing. Cheaper exp won't help until ILP is fixed.

Harness: `/localdev/smarton/prof_ablation.sh` (edit kernel → capture → save CSV
→ restore from pristine backup; never `git checkout` the device tree — its kernels
are an uncommitted rsync of Mac HEAD `dc6b38c`).

## Evidence (Track-2 PHASING ablation, ttw render_clean, 30 views, 2026-06-11)

Measured the iter-112 phased kernel (HEAD `aba575c`) against a bit-identical
UNPHASED reconstruction (full per-microblock `power→exp→alpha→RGBT` chain
back-to-back; same instructions, same `DR_SCR` spill, only the SFPU issue ORDER
differs → isolates Track 1's ILP effect). Harness
`opt/prof/prof_ablation_phasing.sh` (on device `/localdev/smarton/prof_ablation_phasing.sh`),
same capture method as the baseline ablation; device `yyzo-bh-07`.

| config        | tile_blend_sfpu ms/view | hero md5 (bit-identity) |
|---------------|-------------------------|-------------------------|
| **phased** (HEAD, Track 1) | **45.58** | `e3fefb11…` |
| **unphased** (pre-Track-1) | **34.11** | `e3fefb11…` (≡ phased) |
| noexp-on-phased            | 47.10 | `2f786db3…` (exp removed) |
| nomath (reader floor)      | 0.63  | — |
| noread (sfpu floor, stale) | 61.01 | — |

**Verdict: Track 1 phasing REGRESSED blend by +11.47 ms/view (+33.6%).** phased
45.58 vs unphased 34.11, with **bit-identical pixels** (phased.md5 == unphased.md5,
so the measurement is valid). The frame-time drift corroborates: unphased 30-view
avg = 220.1 ms vs phased 231.5 ms (~11 ms, matching the blend-zone delta), and the
unphased 34.11 ≈ the original 2026-06-08 baseline 30.5 (the old baseline WAS the
unphased kernel; the small residual is sort-path drift from iters 113–114).

**Why it hurt — fan-out limited (K≈1).** Phasing only pays off when a gaussian
covers ≥2 live microblocks (K≥2) so independent per-microblock ops can fill the
SFPU pipeline. With K≈1 there is no cross-microblock ILP to expose, and phasing
just adds **4× the mask-scan traversal** (four 32-deep `blend_phase_*` recursions
per gaussian = 128 branch checks vs the unphased single 32-deep walk = 32) on top
of the identical `DR_SCR` spills — pure overhead, zero payoff. **exp is NOT the
lever**: noexp-on-phased (47.10) is within noise of phased (45.58), i.e. removing
the SFPU exp saves ~0 ms — blend is still **dependency-stall-bound on the per-pair
chain**, exactly as the original ablation concluded.

**Recommendation:**
1. **REVERT Track 1** (iter 112) — it is a pure ~11.5 ms/view regression with no
   upside on this workload. Restores blend to ~34 ms/view, recovering the frame
   drift (211→~200 ms/view).
2. **Track 3 fan-out histogram FIRST** to confirm K≈1 quantitatively before any
   further ILP work. If confirmed, the microblock axis is dead for ILP — the
   parallel axis MUST be **batch B gaussians** (Track 3's microblock-major alpha
   batching: compute B independent `alpha`s phased, then the short serial T/color
   recurrence) for a fixed ILP width independent of fan-out. Do NOT pursue a
   cheaper exp (proven a no-op here).

## Why this is an ILP problem, and the divergence question

Concern: "how can microblocks run in parallel if they need different numbers of
iterations (different gaussian counts / early-out depths)?"

Answer — pick parallel axes that have **no** divergence:

1. **SIMD (32 lanes) = PIXELS within one microblock**, not microblocks. All 32
   pixels of a microblock share the same gaussian list and early-out depth ⇒ no
   intra-vector divergence. Two microblocks with different depths are different
   vectors processed at different times; their unequal lengths never share a lane.
2. **ILP (pipelining) = independent chains in flight**, not lockstep loops. The
   divergent dimension (gaussians-per-microblock, T-saturation depth) stays a
   **serial outer loop**. We pipeline the *divergence-free* unit: one gaussian's
   contribution to the K microblocks it covers — each is exactly ONE blend
   (uniform, length-1, independent accumulators). We never force unequal-length
   loops into lockstep.

Current state:
- **Cull is already phased** (`cull_phase_thr → _fx → _fy → _combine`: all 32
  batch lanes of one op before the next dependent op). This is why cull (21 ms)
  is healthier than blend.
- **Blend is NOT phased**: `dispatch_blend_guarded` runs the full
  `power→exp→alpha→at→RGBT` chain for microblock 0, then microblock 1, … Each
  chain is serial and issued back-to-back ⇒ the SFPU stalls on op latency.

## Tracks

### Track 1 — Phase the blend dispatch (PRIMARY) — DONE (iter 112), REGRESSED, REVERT
**Track-2 measured it: phased 45.58 vs unphased 34.11 ms/view (+33.6% SLOWER),
bit-identical. Fan-out K≈1 ⇒ no ILP to expose; phasing is pure 4×-mask-scan +
spill overhead. REVERT it. See the Track-2 Evidence section above.**
Restructure `process_tile_l1_blend` / `dispatch_blend_guarded` in
`render/kernels/compute/alpha_blend_compute_mb.cpp` so each gaussian's K covered
microblocks are processed in phases instead of full-chain-at-a-time:
1. Phase A: `power = A·dx² + B·dx·dy + C·dy²` for all K → DEST scratch.
2. Phase B: `weight = exp(power)` for all K.
3. Phase C: `alpha = min(op·weight, 0.99)` for all K.
4. Phase D: `at = alpha·T; R/G/B += at·c; T·=(1-alpha)` for all K (independent
   across microblocks — separate accumulators — so it pipelines too).
- Process in chunks of ~4 microblocks to fit the SFPU LReg budget; spill
  intermediates to DEST between phases (cull does exactly this).
- R/G/B/T/X/Y for all 32 microblocks are already DEST-resident → no extra
  accumulator traffic.
- Expected: ~2× IPC where fan-out K≥2 ⇒ blend **30 → ~16–20 ms**. Bounded by
  average per-gaussian microblock fan-out.
- Risk: DEST read-after-write hazards (cull hit this; the phasing IS the fix).
  Keep PSNR ≥ 60 dB gate.

### Track 2 — Validate with the ablation harness — DONE (iter 115, 2026-06-11)
Added `opt/prof/prof_ablation_phasing.sh` (phased / unphased / noexp-on-phased /
nomath / noread); ran the 30-view capture flow on `yyzo-bh-07`. **Result: phasing
REGRESSED blend +33.6% (45.58→34.11 unphased), bit-identical; exp removal saves
~0 (noexp 47.10 ≈ phased 45.58) ⇒ still stall-bound, NOT throughput-bound; a
cheaper exp does NOT matter. → REVERT Track 1; do Track 3's fan-out histogram
next.** See the Track-2 Evidence section above.

### Track 3 — Raise effective fan-out (amplifies Track 1)
Phasing only helps when a gaussian covers ≥2 live microblocks.
- First: instrument a per-gaussian microblock-fan-out (mask popcount) histogram
  to size the chunk width to reality.
- If fan-out-limited: **batch B gaussians for the alpha-compute** (microblock-
  major): compute B independent `alpha`s (the expensive exp+quadratic) phased,
  then the short serial T/color recurrence (respects front-to-back order; only
  the cheap recurrence is serial). Fixed ILP width B regardless of footprint.
  More complex; only if Track 1 is fan-out-limited.

### Track 4 — Fuse cull + blend into one pass (structural, higher risk)
Today cull and blend are two SFPU passes over the same sorted subchunks (cull
writes the mask into slab word3; blend reads it). Fuse per-gaussian: compute
coverage → immediately blend live microblocks. Eliminates one full pass + shares
the dx/dy/conic setup. Potential cull+blend **51 → ~35 ms**. Gate behind Track 1.

### Track 5 — Offload the quadratic form to the matrix/FPU engine (speculative)
The matrix unit is idle during blend. `A·dx²+B·dxdy+C·dy²` is a small quadratic
form a FPU MAC could compute, freeing SFPU cycles for exp+accumulate. Prototype
only; the SFPU↔FPU DEST handoff may eat the savings.

## Execution order
Track 1 → Track 2 (measure) → Track 3 if fan-out-limited → Track 4 if more is
needed → Track 5 speculative. Cull is already phased; its near-term win is Track
4 (fusion) + a cheaper conservative coverage test, not re-phasing.

## Relationship to the sort plan (`opt/sort-l1-resident-plan.md`)
The two plans are coupled and should land together:
- Sort Stage 3 (O2) emits records **directly in this kernel's SFPU vector layout**
  (field-SoA in `cull_perm(g,m)` order). Decide Track 1's DEST field order here;
  the sort emit writes into it ⇒ blend does pure contiguous vector loads, no
  AoS→lane.
- Once records are **L1-resident** (sort tile-stationary handoff), the random-
  gather latency shadow that hides today's inline cull is **gone** → the cull must
  stay a **precomputed SFPU mask** pass (lessons §13: inline cull on resident
  records blew blend to 416 ms). This constrains Track 4 (cull+blend fusion):
  fuse the *math*, but keep the mask precompute.
- Sort Stage 5 (trace/host-free) folds this kernel into the single replayed
  frame; keep the blend compute free of host-set per-tile runtime args (read
  counts/flags from the device CB, as the reader already does).

## Guardrails
- PSNR gate ≥ 60 dB (`ttw.toml`); hero `hero_vs_ref` must hold.
- Measure on device with `prof_ablation.sh`-style captures; never trust
  wall-clock zone duration alone (it conflates busy vs semaphore-stall — that is
  what produced the wrong "co-limited" conclusion originally).
- Device kernels are an uncommitted rsync of Mac HEAD; restore from file backup,
  never `git checkout` on the device.
