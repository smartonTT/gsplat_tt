# Blend + Cull speedup plan (SFPU dependency-stall)

Status: PLANNED (not yet executed). Owner: supervisor. Evidence: on-device
ablation captured 2026-06-08 (see "Evidence" below).

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

### Track 1 — Phase the blend dispatch (PRIMARY)
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

### Track 2 — Validate with the ablation harness
Add a `phased` config to `prof_ablation.sh`; re-run the 4-capture flow; compare
`tile_blend_sfpu` makespan. Then run NOEXP-on-phased: if phasing converts blend
to throughput-bound, a cheaper exp finally matters (chain Track via A4).
Evidence-gated, same method as the baseline ablation.

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
