# Tile-Assign ("ta") Stage — Bottleneck Analysis & Optimization Plan

Analysis worker, 2026-06-01. HOST-ONLY read-and-document pass (no device runs).
All claims are grounded in the source + existing profiling logs cited inline.

Project per-stage budget (hero `bicycle`, 1 view, all-resident gates on):
`proj 54 · ta 129 · sort 16 · blend 191` ms/view ≈ 390. Gate `hero_vs_ref ≥ 63.6`,
baseline 63.85 dB. ta is the **2nd-biggest stage and essentially unoptimized**.

---

## TL;DR (read this first)

- **ta is 100% soft-float-compute-bound on a single data-mover RISC per core.**
  It is NOT mover/DRAM-bandwidth-bound, NOT host/`Finish`-bound, NOT scan-bound.
  Every ta kernel runs on `RISCV_1` (a NoC data-mover with **no FP hardware**),
  so all the floating-point math compiles to scalar soft-float library calls.
- The 129 ms decomposes (measured) as **K4 per-pair Mahalanobis cull 66 ms (51%)
  + K3 per-Gaussian `m2_thresh` log 37 ms (29%) = 103 ms (80%)**, plus K1 AABB
  10.5 ms, K2 scatter 13.8 ms, on-device scans ~1.6 ms, host bridges ~0.
- **The killer fact: that 103 ms of soft-float removes only 4.6 % of pairs**
  (`P=3,369,033 → P_kept=3,213,120`, see `.ttw/logs/verify-latest.log:76`). The
  ta-stage cull is nearly a no-op on pair count while costing 80% of the stage.
- **Best first experiment (no kernel rewrite): drop the ta-stage cull (K3+K4)**
  and let blend's existing microblock cull do the rejection. Projected ta
  129 → ~26 ms (−103 ms) if PSNR holds. Must measure PSNR on device.
- If the ta cull must stay for PSNR: move K4 to the SFPU (reuse
  `microblock_cull_compute.cpp`) and fold K3's log into projection. But both are
  larger lifts and K4-on-SFPU has a known **unresolved correctness bug**
  (`opt/sfpu-cull-next.md` iter 7).

---

## 1. What the ta stage is and how data flows today

Env flags ON in the verify command (`.ttw/logs/verify-latest.log:6`):
`GSPLAT_TT_DEVICE_TILE_ASSIGN=1 GSPLAT_TT_RESIDENT_TA_IN=1 GSPLAT_TT_RESIDENT_PAIRS=1
GSPLAT_TT_TA_DEVICE_SCAN=1` (+ resident project/gather/sort/blend). So ta runs the
fully-resident path: inputs read over NoC from `proj_m_*`, pairs left resident in
DRAM for sort, prefix-sum on-device, **zero host round-trips**.

Host driver: `src/gsplat_tt/tile_assign_device.cpp::tile_assign_tt`
(`383`–`1024`). It launches **seven** programs, each a single-RISC dataflow
kernel on `DataMovementProcessor::RISCV_1` over the whole compute grid
(`num_cores = grid.x*grid.y`; P100 Blackhole, harvest mask Tensix `0xc0`):

| # | kernel (file) | unit | what it computes | resident I/O |
|---|---|---|---|---|
| K1 | `kernels/dataflow/tile_assign_bbox.cpp` | per-Gaussian (M) | AABB in tile space → `tiles_per_gaussian[m] = w*h` | reads `proj_m_px/py/rx/ry`; writes `tpg` |
| scan_reduce | `tile_assign_scan_reduce.cpp` | per-core | sum of `tpg` over core's range → `core_total[core]` | int only |
| scan_bases | `tile_assign_scan_bases.cpp` | core 0 | exclusive scan of per-core partials → `core_base[]` + publishes `P` to `ta_pairs_P` | int only |
| scan_add | `tile_assign_scan_add.cpp` | per-Gaussian (M) | seeded exclusive prefix-sum → `offs[M+1]` | int only |
| K3 (m2thr) | `tile_assign_m2thr.cpp` | per-Gaussian (M) | `m2t = -2·log(floor/op)` (or −1 sentinel) | reads `proj_m_opacity`; writes `m2thr` |
| K2 | `tile_assign_scatter.cpp` | per-pair (P) | binary-search owning Gaussian; emit `(gid, tid)` page-aligned | reads `offs`+means/radii; writes `gids/tids` |
| K4 (cull) | `tile_assign_cull.cpp` | per-pair (P) | constrained-min Mahalanobis vs tile rect → `keep[p]∈{0,1}` | reads pairs+cov+means+`m2thr`; writes `keep` |

Output handoff (resident-pairs, `tile_assign_device.cpp:917`–`936`): registers
`ta_pairs_gid / ta_pairs_tid / ta_pairs_keep` + `ta_pairs_P` in `device_state`;
sort bins them on-device and compacts implicitly via `keep[]` — **no D2H of pairs,
no host compaction** (those host bridges, `compact`/`d2h`, measure 0.00 ms below).

Algorithm parity: K1/K2/K4 reproduce `gsplat_cpu::tile_assign` Phases 1/3/4
bit-exact (`src/gsplat_cpu/tile_assign.cpp:48`–`365`). K3 mirrors the host
`m2_thresh` precompute (`tile_assign.cpp:171`–`197`); the −1 sentinel folds the
opacity-floor drop (`tile_assign_cull.cpp:178`–`187`).

### Data volume (hero, from logs)
- M = 1,883,905 visible Gaussians; P = 3,369,033 AABB pairs ≈ **1.79 pairs/Gaussian**
  (overlap factor; radii are tight 3σ boxes — see §3 why the cull barely fires).
- K1: reads 4×M×4B ≈ 30 MB, writes M×4B ≈ 7.5 MB.
- K2: writes 2×P×4B ≈ 27 MB; reads `offs` + per-Gaussian-page-cached attrs.
- K3: reads M×4B 7.5 MB, writes 7.5 MB.
- K4: reads 2×P×4B pairs (27 MB) + per-Gaussian-page-cached cov/means/m2thr;
  writes P×4B keep ≈ 13.5 MB.
- Total DRAM traffic ≈ **~150–200 MB/view**. On GDDR6 (hundreds of GB/s–TB/s
  aggregate) that is **< 2 ms** — so bandwidth is not the limiter (see §3).

---

## 2. Measured sub-stage breakdown (the smoking gun)

`GSPLAT_TT_TA_TIMING=1` steady-state line, `.ttw/logs/ta-timing-20260531-231952.log`
(matches `ta-timing-after-…:` — warmup dropped):

```
[TA] M=1883905 P=3369033 resident_in=1 resident_pairs=1 cull=1 dev_scan=1 |
 k1=10.53 scan1=0.60 d2h_tpg=0.04 prefix=0.10 scan2=0.92 h2d_offs=0.00
 k2=13.78 k3=37.49(c=37.49 h2d=0.00) k4=65.85 publish=0.01 compact=0.00
 d2h=0.00 total=129.32ms
```

| sub-stage | ms | % | what dominates the cost |
|---|---:|---:|---|
| **K4 per-pair Mahalanobis cull** | **65.85** | **51%** | soft-float quadratic form over P=3.37M pairs on the mover |
| **K3 per-Gaussian `m2_thresh`** | **37.49** | **29%** | soft-float `std::log` over M=1.88M on the mover |
| K2 scatter | 13.78 | 11% | int binary-search + div/mod + NoC writes (recomputes AABB) |
| K1 AABB count | 10.53 | 8% | 4 soft-float divides + clamps over M |
| scans (1+bases+2) | 1.62 | 1% | integer, on-device |
| d2h_tpg (read P, 64 B) | 0.04 | ~0 | one control page |
| publish / compact / d2h | ~0 | ~0 | **eliminated** in resident-pairs mode |
| **total** | **129.32** | | |

Production (no `TA_TIMING`) overlaps K3 with the scans+K2 by removing the host
barriers (`k3_pipeline`, `tile_assign_device.cpp:490`,`620`), but K3/scans/K2 still
execute serially on the same RISC in the CQ, so wall-clock ≈ the same 129 ms
(verify summary: `ta=128.7`, `verify-latest.log:87`). Overlap removes host
launch bubbles, not device compute.

---

## 3. Bottleneck diagnosis (mover vs compute vs scan/host) + roofline

**Diagnosis: ta is soft-float-compute-bound on the data-mover RISC.** Reasoning:

1. **It's compute, not DRAM/NoC.** Every kernel is `RISCV_1` only
   (`tile_assign_device.cpp:146`,`170`,`196`,`221`,`248`,`274`,`298`). The two
   biggest sub-stages do heavy FP per element: K4 a full constrained-min
   quadratic form with 2 divides + ~10 mul/add per pair
   (`tile_assign_cull.cpp:195`–`234`), K3 a `std::log` per Gaussian
   (`tile_assign_m2thr.cpp:88`). The data-mover RISCs have **no FP unit**, so each
   of these is tens–hundreds of cycles of soft-float (the exact failure mode
   `plan-high-utilization-pipeline.md §1.1`/§2 names). Total DRAM traffic is
   ~150–200 MB (§1) → < 2 ms at GDDR6 rates, two orders below the 103 ms spent.
   So the cost is the scalar FP, not byte movement.
2. **It's not host/`Finish`-bound.** In resident mode `d2h/h2d/compact/publish`
   all measure 0.00 ms; the only host read is one 64 B page of `P` (0.04 ms).
3. **It's not scan-bound.** On-device scan total ≈ 1.6 ms.
4. **It's not load-imbalance (LPT) bound.** Work is split by contiguous page
   ranges evenly across cores (`split_pages`, `tile_assign_device.cpp:356`); K4
   is per-pair uniform, so cores finish within ~1 page of each other. (Per-tile
   skew bites sort/blend, not ta.)

**Roofline — what ta SHOULD cost.** Using the plan's §7 method (vec-ops ÷
cores·clock):
- K4: P=3.37M pairs × ~30 fp32 ops ÷ 32 SFPU lanes ≈ 3.2e6 vec-instr; at
  ~120 cores × 1.35 GHz × 1 vec-op/cyc ≈ 1.6e11/s → **~0.02 ms of pure ALU**.
- K3: M=1.88M logs ÷ 32 lanes × (~10 op SFPU log) ≈ 6e5 vec-instr → **~0.004 ms**.
- The real floor is then set by **NoC/DRAM movement (~1–2 ms) + scan (~1.6 ms) +
  the integer scatter K2 (~14 ms, already near its int/NoC roofline)**.

So the **order-of-magnitude target for ta is ~15–20 ms** (dominated by the
integer scatter + scans + movement), with the FP sub-stages K1/K3/K4 collapsing
from ~114 ms to a few ms once they leave the soft-float mover. The current 129 ms
is **~7× over roofline**, entirely because of FP-on-mover.

**The decisive structural fact.** K4 removes only **155,913 of 3,369,033 pairs =
4.63 %** (`P` vs `P_kept`, `verify-latest.log:76`). The AABB boxes are already
tight 3σ rectangles (radii from projection), so the ellipse-corner false
positives the cull targets are rare. **80 % of the stage (K3+K4, 103 ms) buys a
4.6 % pair reduction** — and blend runs its *own* per-microblock Mahalanobis cull
(`GSPLAT_TT_SFPU_CULL` / `MB_DEVCULL`, `microblock_cull_compute.cpp`) that rejects
at finer granularity downstream. This makes the ta-stage cull a prime candidate
for **removal**, not just acceleration.

---

## 4. Ranked optimizations

Ranked by (expected ms saved ÷ risk·effort). "rebuild?" = needs the host `.so`
(`build-tt`) rebuilt via the loop's `iterate.sh` recipe.

### R1 — Drop the ta-stage cull (K3+K4); let blend's microblock cull reject. ★ BEST
- **Change:** in `tile_assign_tt`, add an env gate (e.g. `GSPLAT_TT_TA_NO_CULL=1`)
  that skips K3 (`enqueue_k3_device`) and K4 (`wl_cull`), and publishes the
  resident pairs with an **all-kept** mask so sort passes every AABB pair through.
  Minimal host change: in the `resident_pairs_active` block
  (`tile_assign_device.cpp:917`) register `ta_pairs_gid/tid`, set
  `P_kept = P`, and either (a) make sort treat a missing/absent `ta_pairs_keep`
  as all-ones, or (b) cheaply fill `buf_keep` with 1s (one tiny kernel or a memset
  buffer) — confirm which sort expects (`sort_device.cpp` reads `ta_pairs_keep`).
- **Mechanism of win:** deletes 103 ms of soft-float (K3 37 + K4 66). ta
  129 → **~26 ms**.
- **Downstream cost:** sort/blend process 4.6 % more pairs (P 3.37M vs 3.21M):
  sort ≈ +0.7 ms (bin/kernel scale ~linearly, `verify-latest.log:79`), blend's
  microblock cull rejects the extra corners cheaply (it already runs over all
  candidates). Net frame likely **−100 ms**.
- **Risk:** PSNR. The blend microblock cull must reject the same empty-corner
  pairs the ta cull would have. Plausible (finer-grained, same Mahalanobis), but
  **must be measured** (gate ≥ 63.6 dB). Medium risk, tiny effort.
- **Rebuild:** yes (small host edit; possibly a 5-line keep-fill kernel).
- **Expected: −103 ms** ta (→ ~26 ms), pending PSNR.

### R2 — Fold K3 `m2_thresh` out of ta (into projection or precompute-once).
- **Change:** `m2_thresh = -2·log(floor/op)` depends only on per-Gaussian opacity
  and the constant `contrib_floor`. The **project compute kernel already touches
  opacity per visible Gaussian on the compute RISC** and writes `proj_m_opacity`;
  have it also emit `proj_m_m2thr` (one extra SFPU `log` + store, ~free relative
  to projection's existing per-splat work). ta then reads `proj_m_m2thr` over NoC
  instead of running K3.
- **Mechanism:** removes K3 (37 ms) from ta; the log moves onto the SFPU inside a
  stage that's already paying for the per-Gaussian sweep.
- **Risk:** low–medium. Must keep bit-faithful `log` vs host (`-ffp-contract=off`,
  see `tile_assign_cull.cpp:43`) so K4's threshold compare doesn't flip; verify
  PSNR. Edits `project_device.cpp` + its compute kernel.
- **Rebuild:** yes. **Expected: −37 ms** ta. (Subsumed by R1 if R1 lands.)

### R3 — Move K4 cull to the SFPU compute path.
- **Change:** reimplement the per-pair constrained-min Mahalanobis on the SFPU,
  reusing the machinery in `kernels/compute/microblock_cull_compute.cpp` (it
  already evaluates the same quadratic form per microblock). Pack pairs SoA,
  vectorize 32 pairs/op, keep `keep` packing on the mover.
- **Mechanism:** 66 ms soft-float → ~6–10 ms (the blend cull saw 422 → 37 ms,
  ~11×, `opt/sfpu-cull-next.md:244`).
- **Risk: HIGH.** The SFPU cull has an **unresolved correctness bug**: a
  codegen/execution defect for mid-range covariance flips masks (29.6 dB),
  documented through iter 7 (`opt/sfpu-cull-next.md:139`–`170`). Don't build on it
  until that's fixed. Large effort.
- **Rebuild:** yes. **Expected: −56 ms** ta — only if R1 is rejected on PSNR and
  the SFPU-cull bug is fixed first.

### R4 — Halve K1 by moving AABB divides to SFPU / fusing K1 into K2.
- **Change:** K1's 4 soft-float divides/Gaussian (`tile_assign_bbox.cpp:126`–`129`)
  are recomputed verbatim in K2 (`tile_assign_scatter.cpp:160`–`165`). Either
  (a) have K1 also store `min_x/min_y/w` (3 int arrays) so K2 skips the recompute,
  or (b) move the divides to the SFPU. K1's `(px±rx)/tile_size` is a per-Gaussian
  elementwise op (multiply by `1/tile_size`, a constant).
- **Mechanism:** removes duplicated AABB float math; K1 10.5 → ~3–5 ms.
- **Risk:** low (integer tid math unchanged; preserve truncation semantics of
  `(int)` cast for bit-exactness). **Rebuild:** yes.
- **Expected: −5 to −7 ms** ta. Lower priority; only matters after R1/R3.

### R5 — (Algorithmic, later) coarse pre-reject before scatter.
- A cheap per-Gaussian early-out (e.g. opacity-floor drop applied in K1 so dead
  Gaussians never expand pairs) trims P upstream. With opacity static across
  views this is essentially free. Small win on this scene (most M survive); note
  for completeness. **Rebuild:** yes.

---

## 5. Single best first experiment

**Run R1: disable the ta-stage cull and measure PSNR + per-stage ms.**

Why first: it's the cheapest change (host-only, no new FP kernel), it directly
tests the load's most surprising fact (the cull removes 4.6 % for 80 % of the
stage), and its outcome decides the whole branch:
- **If PSNR holds ≥ 63.6 dB** → ship it; ta drops 129 → ~26 ms (~−100 ms/view,
  the single biggest available win) and R2/R3/R4 become unnecessary for ta.
- **If PSNR regresses** → the corner pairs matter; fall back to R2 (free K3
  removal, −37 ms, low risk) and queue R3 (SFPU K4) **after** the
  `sfpu-cull-next.md` codegen bug is fixed.

Concrete recipe for the implementation worker:
1. Add `GSPLAT_TT_TA_NO_CULL` gate in `tile_assign_tt`; when set, skip
   `enqueue_k3_device`/`wl_cull`, publish pairs with `P_kept=P` and an all-ones
   keep (check `sort_device.cpp`'s `ta_pairs_keep` expectation first).
2. Rebuild `build-tt`; run `a003_verify.py --views 1` ×2 with the full resident
   env from `verify-latest.log:6` plus the new flag; confirm `hero_vs_ref` and
   read the new `ta=` from the SUMMARY line.

---

## 6. Uncertainties that require an on-device run (measure, don't assume)

1. **R1 PSNR** — does blend's microblock cull reject the 4.6 % corner pairs the
   ta cull drops? Only a device verify answers this. (Hypothesis: yes, finer
   granularity, same math.)
2. **Sort's `keep` contract** — confirm whether `sort_device.cpp` requires
   `ta_pairs_keep` present or can treat its absence as all-kept; dictates whether
   R1 needs a keep-fill kernel. (Read-only TODO: inspect `sort_device.cpp` binning
   before editing — not done here as it's the sort stage.)
3. **Production K3 overlap** — `TA_TIMING` serializes K3; the true overlapped
   contribution of K3 in the no-timing path should be confirmed by toggling K3
   alone (it appears non-overlapping in wall-clock, but verify).
4. **R2/R3 bit-fidelity** — moving `log` (R2) or the quadratic form (R3) onto the
   SFPU changes rounding; the threshold compare is boundary-sensitive
   (`tile_assign_cull.cpp:43` documents fast-math flipping decisions). PSNR must
   be re-gated after either.
5. **Exact core count** — `num_cores = grid.x·grid.y` on this harvested P100
   (Tensix mask `0xc0`); the roofline used ~120. The absolute floor scales with
   the real count but the diagnosis (FP-on-mover, ~7× over roofline) does not.

---

## 7. File:line index (for the implementation worker)

- Host driver / orchestration: `src/gsplat_tt/tile_assign_device.cpp`
  - K3 device enqueue + pipeline gate: `:493`–`507`, `:490`, `:620`–`642`
  - on-device scan path: `:610`–`696`
  - K2 scatter launch: `:745`–`783`
  - cull (K3 precompute + K4) + resident-pairs publish: `:820`–`936`
  - env flags: `:427` (RESIDENT_TA_IN), `:437` (RESIDENT_PAIRS), `:454` (DEVICE_SCAN)
- Kernels (all `RISCV_1` data-mover): `src/gsplat_tt/kernels/dataflow/`
  - `tile_assign_bbox.cpp` (K1 AABB), `:126`–`132` the float divides
  - `tile_assign_scatter.cpp` (K2), `:160`–`165` recomputed AABB
  - `tile_assign_cull.cpp` (K4), `:195`–`234` quadratic-form cull
  - `tile_assign_m2thr.cpp` (K3), `:88` the `std::log`
  - `tile_assign_scan_{reduce,bases,add}.cpp` (on-device prefix-sum)
- CPU reference (bit-exact target): `src/gsplat_cpu/tile_assign.cpp:48`–`365`
- SFPU cull reuse target (+ its open bug): `kernels/compute/microblock_cull_compute.cpp`,
  `opt/sfpu-cull-next.md`
- Profiling: `.ttw/logs/ta-timing-20260531-231952.log`,
  `.ttw/logs/verify-latest.log:76`,`:84`–`:89`
- Pipeline context: `opt/plan-high-utilization-pipeline.md` §1, §2, §4-B, §7, §8
