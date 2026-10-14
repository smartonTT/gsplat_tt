# Structural plan 4: from 8.06 ms/view to the 1.83 ms/view Faster-GS target

Task #663, 2026-10-13 (PDT). A plan only: no device runs and no code. This file supersedes the
draft of the same path on ttp/t603 (@bf425d81), which predates iters 226-229.

Labels: **M** = measured on our hardware, **P** = published (GPU numbers; never measured by us),
**E** = estimated (arithmetic or a model; the basis is given next to it).

## 0. Where we are and what the target is

| item | value | label |
|---|---|---|
| Headline: best-iter-229 (3702fbb5, buildid 7b7ace57), bh-33, p100, 110 cores (11x10), 1024x1024, 30 bicycle views, b2b | **8.063 ms/view (124.0 FPS)** | M |
| 2 chips, AFR, bh-50 (2-board p150), pre-227 build | 4.226 ms/view b2b | M |
| AFR viewer page FPS, bh-50 (#615) | 113.3 FPS on 2 chips vs 71.2 on 1 | M |
| Target: Faster-GS inference renderer, RTX 4090, bicycle 1237x822 (Faster-GS, CVPR 2026, Table 6) | **1.83 ms/view (547.1 FPS)** | P |
| Other GPU references, RTX 4090, same scene and resolution | TileGS 3.979, gsplat 4.395 (TileGS arXiv 2609.03613 Table 5); INRIA 3DGS 6.18 (Faster-GS Table 6) | P |

Resolution: 1237x822 = 1.017 Mpx vs our 1024x1024 = 1.049 Mpx (3% more for us).

**Caveat on the target (new finding, needs a check).** The Faster-GS 547 FPS row renders the
Faster-GS *trained* bicycle model, not the INRIA checkpoint we render. The Faster-GS README lists
"informed pruning from Speedy-Splat" and MCMC densification in its default training (P). Its
Gaussian count for bicycle is not in the main paper text. Our dashboard row G5 calls it "full
unpruned model". If that model has far fewer Gaussians than our 5.56M, the 3.4x gap between the
INRIA 3DGS (6.18 ms) and Faster-GS (1.83 ms) rows is partly a model-size effect. Follow-up F1 checks
the count (supplemental Table) and fixes the G5 label. The verdict below does not depend on it.

## 1. Stage breakdown at iter 229 with hardware floors

Method: t630 (FOLD=1 traced, best-iter-227) windows, scaled by 0.89 (traced 9.48 ms sum vs 8.454 ms
untraced b2b at 227), then moved for iter 228 (L-A removed the 0.224 ms mat+blend->sort gap) and
iter 229 (SWAP_FILL, -0.163 ms measured, all in blend). The per-view programs run serially on one
in-order CQ, so their windows add up.

| stage (program) | ms/view at 229 | label | what bounds it | hardware floor (arithmetic) | floor label |
|---|---|---|---|---|---|
| pfwc (project+cull over N = 5.56M) | ~1.38 (1.55 traced x 0.89) | E | NCRISC/BRISC movers set the program end; TRISC close behind (t614: end is 1:1 on b2b, ds150 +0.122, ds400 +0.376 M) | DRAM: ~334 MB/view / 420 GB/s = 0.80 ms (p150 at 512 GB/s: 0.65) | E (bytes from ledger, BW M) |
| sort (one-launch counting scatter by tile) | ~1.33 (1.49 x 0.89); emit 1.1-1.2 | E | scalar movers, ~164 core-cycles/pair, NoC scatter | 2.614M pairs x 8 B = 21 MB; 2 passes r+w = 84 MB / 420 GB/s = 0.20; scalar 40 cycles/pair on 220 movers = 0.35 | E |
| mat phase (mask on TRISC1 + per-tile radix on movers), serial before blend | ~1.33 critical core incl. drain (1.49 traced) | E | both sides: movers 1.36 traced mean (sort_subchunk_mat), TRISC1 1.08 mean real cull work (#595 M) | per-tile radix at ~20 cycles/pair: 0.32; mask: no floor sized | E |
| blend SFPU (critical core) | ~3.79 (4.44 x 0.89 - 0.163) | E | TRISC1 SFPU issue: 54 slots per Gaussian x 32-px microblock (U9 M + SWAP_FILL) | see below: ~2.1 at 30 slots | E |
| end spread, drain, gaps | ~0.2 | E | blend imbalance max-mean 0.24 traced; view-boundary gap ~0 after L-A | 0 | E |
| **sum** | **~8.03** | E | | | |
| **measured** | **8.063** | M | | | |

Host-side cross-check (iter 229 B2B_STAGES, host waits, M): sort 0.80, blend 6.97 (includes
pfwc(N+1) device time), xview 0.32, view_total 8.16.

**Blend arithmetic.** Critical core: 3.79 ms x 1.35 GHz = 5.1M cycles. Pairs per core: 2.614M / 110
= 23.8k, so ~215 cycles per pair, or ~4.0 microblock evaluations of 54 slots per pair (t630 counted
~3.7 live microblocks per pair; the rest is loop overhead). Per-slot value: SWAP_FILL cut 4 slots
and gained 0.163 ms (M), so **one slot is worth ~0.041 ms/view** (E, from M). The body is ~28 math
slots (exp included) and ~26 load/store/nop/swap slots. A body of ~30 slots would make blend ~2.1
ms (E). The math itself is near minimal: dx, dy, the quadratic form, exp, the opacity clamp, the
1/255 test, three colour MADs and the T update.

**Compute ceiling vs the GPU (E).** SFPU: 110 cores x 32 lanes x 1.35 GHz = 4.75 T lane-ops/s.
RTX 4090: 82.6 TFLOPS fp32 (P, spec sheet), about 41 T FMA/s, so ~9x the lane throughput. Our blend
issues ~523M SFPU slots per view (2.614M x 3.7 x 54), i.e. ~16.7 G lane-ops; that alone is 3.5 ms
at full SFPU issue. A 4090 does the same arithmetic in well under 1 ms.

**Ideal single-chip floor for this model (E).** Blend at 30 slots 2.1 + mask ~0.5 + pfwc 0.80 +
sort 0.35 = **~3.75 ms** run serially. If pfwc and sort ran fully hidden under blend on the movers
and DRAM: max(SFPU 2.6, movers/DRAM ~1.2) = **~2.6 ms**. Both are above 1.83 ms.

Missing data, turned into tasks: the mask's own floor (R3 step 0), where SIB's +2.7 ms in blend went
(R2), and the pfwc reader's DRAM-wait share of the mover end (R7 step 0).

## 2. GPU techniques (Faster-GS, TileGS, Speedy-Splat, HiGS) mapped to Tensix

| GPU technique (P) | Tensix mapping | status here | est. saving (E) | basis |
|---|---|---|---|---|
| Fused preprocessing (activations, projection, cull in one kernel) | pfwc fused project+cull+record write | landed (pfwc fuse, REC32 iter 221) | 0 | done |
| Exact splat/tile intersection (Speedy-Splat SnugBox/AccuTile, TileGS) | B2 ellipse-vs-tile test in K2 | #659 re-measure in flight | 0.1-0.15 | #538 M -0.116 at 9.2 ms |
| Warp-level culling in blend (Faster-GS) | per-pair 32-px microblock mask (mat_cull_mask) | landed; ~3.7 of 32 microblocks live per pair | 0 | done; its cost (mask 1.08 ms) is R3 |
| Early ray termination (T < eps) | T_EPS skip in blend | landed; cutting it costs PSNR (#580) | 0 | M |
| Global radix sort on 64-bit (tile, depth) keys (CUB onesweep) | counting scatter by tile + per-tile adaptive radix in mat | landed | see R3 / R6 | sort-related time is ~2.7 ms/view (sort 1.33 + mat movers ~1.3) vs a ~0.55 floor |
| Tile grouping: sort at coarse tiles, rasterize fine (GS-TG) | same shape as ours (sort by 32x32 tile, cull per microblock) | landed | 0 | done |
| Load balancing (per-tile work queues) | static LPT deal; dynamic stealing not built | R5 | ≤0.2 | end spread 0.23 traced |
| Compact fp16 scene layout (HiGS path in gsplat) | bf16/fp16 pfwc input records | R7 | 0-0.3 | only if reads set pfwc's mover end |
| Model-side pruning (Speedy-Splat, Faster-GS training) | LOD / merge at load (U16) | R9 study | 0.5-1.5 if PSNR holds | changes the image; user decision |
| Higher SM count / clock | p150 (120 cores, 512 GB/s) | R8 headline | 0.3-0.6 | core-bound share ~6.5 ms x (1-110/120) |

## 3. Structural redesigns

| redesign | bound (E) | risk | effort | verdict |
|---|---|---|---|---|
| Persistent kernel (L4) | program boundaries only, ≤0.38 (#591) | code is 2.37x the kcfg limit (#579) | high | not ranked: #591 sized the remaining boundary gain at ≤0.38 with the only fitting design at +0.07 |
| Zero-host pipeline | ≤0.03 after the pfwc enqueue (t614 M); boundary idle taken by L-A | low | | done in effect (L-A, iter 228) |
| Sort reuse across views | 0 on the bench (#529); 0.2-0.4 in the viewer | | | viewer only (U18) |
| Sort under blend (SIB: sort(N+1) on blend(N)'s idle movers) | up to ~1.0-1.3 (reader idle 4.47 ms, writer idle ~2.1 ms in blend, t630 M) | **measured +2.84 slower** (#573: 11.505 vs 8.670, blend 7.18 -> 9.88) | high | shelved on M. Re-open only through R2 (cause of the slowdown) |
| Sub-device core split (pfwc on its own cores) | ≤0.4; modelled +0.07 (#591) | | high | excluded (#591) |
| Different tiling: 16x16 screen tiles | more pairs (each Gaussian covers more tiles); per-pair sort, K2 and mask cost rise; blend microblocks barely change | | high | not ranked (E: net loss); microblock mask already gives 32-px culling |
| FPU blend (dense quadratic form, t111/t511/t148) | measured +0.45 worse | | | excluded (M) |
| FPU blend, log-domain variant (L = log(1-a), prefix sum by triangular matmul, T = exp) | ≤ the colour MADs and T update, ~5 of 28 math slots | exp per element stays on SFPU and log(1-a) adds SFPU work | high | excluded (E): moves ~5 slots to the FPU and adds ~8 (log), no new evidence against t511 |
| Blend body redesign (loop order, fewer loads/stores per microblock) | 0.4-0.8 at 44-34 slots (0.041 ms/slot) | LREG pressure (8 LREGs) | medium | **ranked R1** |
| Mat phase: cut both the mover radix and the TRISC mask | 0.3-0.6 | md5 must hold | medium | **ranked R3** |

## 4. Multi-chip (reported as "2 chips", never as the headline)

| mode | ms/view | label | note |
|---|---|---|---|
| AFR, 2 chips (bh-50), pre-227 build | 4.226 b2b | M | #661 is fixing a chip-1 bit mismatch before it lands in the viewer |
| AFR, 2 chips, at the 229 build | ~4.03 | E | 8.063 / 2; host phase B (~1.15 ms/view per chip, t614 M) runs in its own thread per chip |
| Split frame, 2 chips (tile halves) | latency model 6.2; b2b 5.6-6.7 | E (#616) | pfwc does not halve (1.50 -> 1.32); worse than AFR for b2b. #627 measures it (queued) |
| AFR, 4 chips, at 229 | ~2.0 | E | linear AFR; host needs 4 feeder threads at ~60% load. No 4-chip host is charter-named |
| AFR, 4 chips, after R1-R5 (~6.8 single) | ~1.7 | E | the first configuration that beats 1.83 ms/view on paper |

More chips scale AFR throughput almost linearly while the host keeps up (phase B ~1.15 ms per view
per chip). Latency per frame does not improve with AFR; only split frame cuts it.

## 5. Ranked tasks (each above 0.1 ms/view b2b)

Device tasks run on bh-33 under `ttp lock p100`, b2b 30 views, 4 rotated rounds with a same-day A/A
arm, `python3 opt/ab_jit_check.py` on both arms (a JIT-HASH FAIL voids the A/B), hero screenshot on
device + diff vs benchmarks/reference_v2/hero.png, PSNR ≥42.4 dB (margin 0.45), md5 077caaba unless
the task says otherwise, latency only as a secondary column.

| rank | task | size (ms/view b2b) | basis | device | depends on | parallel with |
|---|---|---|---|---|---|---|
| R1 | Blend body: write a minimal TRISC1 schedule on paper (count slots, LREG use), then build the best variant | 0.4-0.8 (54 -> 44..34 slots) | E from M: 0.041 ms/slot (SWAP_FILL); U9 slot count | step 0 no, then yes | — | R2, R3 step 0, R9 |
| R2 | SIB slowdown root cause: one traced run of the #573 SIB build (c33ab7f7) with BZ_PROF counters, split blend's +2.7 ms into TRISC sharing (town compute), NoC/DRAM contention and L1 | profiling (gates R6, up to ~1.0-1.3) | #573 M (+2.84); t630 M (reader 4.47 / writer ~2.1 ms idle in blend) | yes | — | R1 step 0, R9 |
| R3 | Mat phase: step 0 counts at 229 (mover radix cycles per pair, mask SFPU slots per pair); then cut both sides (fewer radix passes on movers; cheaper conservative mask on TRISC1) | 0.3-0.6 | E: mat 1.33 crit; movers 1.36 vs TRISC 1.08 traced (#595 M) | step 0 no, then yes | — | R1 (different TRISC code path, same device: serialize the A/Bs) |
| R4 | B2 ellipse cull at 229 (#659, in flight) | 0.1-0.15 | M -0.116 at iter 225 | yes | — | — |
| R5 | Dynamic blend tile stealing (NoC semaphore counter instead of static LPT) | ≤0.2 | E: end spread 0.23 traced x 0.89 | yes | after R1 (imbalance changes with the body) | R3 |
| R6 | Sort under blend v2 (mover-only emit inside blend(N), no TRISC town) | 0.5-1.2 if R2 clears it | E: sort 1.33 minus contention | yes | **R2** must show the +2.7 ms came from TRISC sharing or a fixable contention | — |
| R7 | pfwc input bytes: bf16 means/covariance inputs. Step 0: read the reader's DRAM-wait share of the mover end from existing pfwc_pr counters (#629/t484); kill if <0.2 ms. Step 1: CPU PSNR check of the quantised scene | 0-0.3 | E: pfwc 1.38 vs DRAM floor 0.80; #629 put the read pattern at ≤0.05 | step 0-1 no | — | all |
| R8 | p150 120-core headline re-measure at 229 (not a code lever) | 0.3-0.6 (headline only) | E: ~6.5 ms core-bound x (1 - 110/120) + DRAM 448 -> 512 GB/s | yes, needs a free non-viewer p150 via `opt/ird/renew.sh measure` | coordinator: only the measurement task reserves | all |
| R9 | LOD / merge study on CPU (U16): PSNR vs Gaussian count against benchmarks/reference_v2/hero.png | 0.5-1.5 if PSNR ≥42.4 | E: pfwc ∝ N, pairs ∝ visible N | no | user decision: does a reduced model count as "full model"? | all |

**Projected single-chip ms/view (E, cumulative, p100):** 8.06 (M) -> R4 7.95 -> R1 7.35 -> R3 6.9
-> R5 6.75 -> R7 6.65. With R6 (only if R2 clears it): ~5.7-6.0. On a p150 (R8): about 0.5 less,
~5.2-6.2. With R9 (only if PSNR holds and the user accepts a reduced model): another 0.5-1.5.

## 6. Excluded levers and why (standing rule #299/#300: no re-queue without new evidence)

| lever | task | result | label |
|---|---|---|---|
| pfwc mover transfer pattern (U1) | #629 | ≤0.05 | M/E |
| Morton reorder + chunk cull; u64 unit gather (U3) | #470/#479/#493, #631/#637/#644 | net loss +0.18..+0.41; u64 10.34 vs 8.45 | M |
| Sort key width / passes (U10) | #632 | 0-0.08 | E |
| Absolute means | #634/#647/#654 | within A/A noise | M |
| Overlap mat with blend | #645 | ≤0.298 traced (= spec B, killed) | E from M |
| Sort in blend (SIB) | #573 | +2.84 ms/view slower | M (re-open only via R2) |
| B4 cross-frame / sub-device pfwc (U17) | #591 | +0.07 modelled | E |
| L4 persistent kernel | #579 | 2.37x kcfg limit; boundary gain ≤0.38 | code size |
| FPU blend (quadratic form, dense GEMM), incl. log-domain variant | t111/t511/t148 | +0.45 worse; exp stays on SFPU | M / E |
| Early pfwc enqueue | t614 | ~0, hidden by queued sort | M |
| Cross-view sort reuse on the bench | #529 | 0 (views not coherent) | E; viewer only |
| Precision cuts, contrib_floor > 1/255, 16 B record | #580, it152-204, #23 | PSNR below 42.4 | M |
| RECFAST / VIS_FAST | #599/#606 | 0.071 / bundle 0.070 | M |
| pfwc compaction A / A' | #596/#611 | killed (model on FOLD=0 windows) | E, flagged weak |
| pfwc cov2d on SFPU, cov_cam on FPU | it197, plan 3 | TRISC is off pfwc's critical path (t614) | M |
| K2 into sort_ol, emit pack on TRISCs, emit cursors, p150 emit table | #298, #165, #215, #358 | slower or ≤0.1 | M / E |
| Metal trace replay | #526 | ≤0.083 | E |
| 16x16 screen tiles | — | not ranked: more pairs, per-pair costs rise (§3) | E |
| Scene pre-filter fitted to the 30 bench views | plan 3 | gaming the bench | judgement |

## 7. Ideas not yet sized

- Viewer-only: cross-view sort reuse (U18, 0.2-0.4 page-side).
- K2 pairs pushed straight into the owner core's L1 (no DRAM scatter), using the ~834 KB of L1
  that is dead during blend (#543). Same family as SIB: wait for R2.
- Blend writer pack (65 µs/tile on scalar BRISC) moved to the packer for the last tile only: drain
  0.25 traced on the critical core; #165 (all tiles) lost, a last-tile-only variant is unsized
  (likely <0.1).

## 8. Reachability verdict

- **1.83 ms/view on one chip with the full bicycle model is not reachable (E).** The ideal floor of
  this design on one p100 is ~2.6-3.75 ms (§1), set by SFPU issue rate in blend. The RTX 4090 has
  ~9x our SFPU lane throughput (P spec vs E arithmetic). The known levers project to ~6.7 ms
  (R1-R5, R7), ~5.7-6.0 with R6, and ~5.2-5.5 on a p150 (all E).
- **INRIA 3DGS on a 4090 (6.18 ms, P) is reachable on one chip (E):** R1+R3 alone give ~6.9.
- **TileGS (3.979 ms, P) and gsplat (4.395 ms, P) are reachable on 2 chips now (E: AFR ~4.03 at
  229; 4.226 M pre-227).**
- **1.83 ms/view needs 4 chips with AFR after R1-R5 (~1.7 ms, E)**, or 2 chips plus a reduced model
  (R9) that holds 42.4 dB, if the user accepts one. Neither is in the charter's resources today:
  a 4-chip host is not named, and the model change needs the user's yes.
- The project should keep working R1-R5 (about 1.3 ms of E gains, each above the 0.1 bar) and R2
  (it decides the largest remaining lever). Stop rule stays as in the ledger: stop when R1-R7 come
  back under 0.1 each or fail the gate.
