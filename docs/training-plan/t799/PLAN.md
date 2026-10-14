# Trainer ms/view after T2 split + LPT, and the next levers (task #799, 2026-10-14)

Refresh of the #773 breakdown (docs/training-plan/t773/PLAN.md). Target: 1-2 ms per training view
on one p150, bicycle images_4 at 1237x822. Labels: **M** measured on device (source hand-off),
**E** estimate or Mac model, **P** published. No device was used for this plan.

## 1. Hero training step today (one view, _DSC8679, N = 6.13M, SH0 unless noted)

| stage | ms/view | label | source / note |
|---|---:|---|---|
| forward, train mode (GSPLAT_TT_BLEND_TRAIN=1), orbit b2b | 14.70 | M | #778 (dataset cameras 13.86 M); render-only is 7.30 |
| fused L1 + D-SSIM loss, K=4/VF=1 | 0.89 | M (synthetic size) | #753, now default |
| T1 blend backward, LPT whole-tile plan | 51.95 | M | #785 (82.41 -> 51.95, bit-identical; busiest/mean 1.655 -> 1.043) |
| T2 projection+SH backward + sparse Adam, heavy-row split kcap 16 | 9.29 | M | #784 (98.42 -> 9.29; A 2.06 + B 7.27; kcap 8: 8.85) |
| T1 -> T2 chain overhead | ~0 | M | #784 chain 90.52 vs 81.37 + 9.29 |
| **device total** | **~76.8** | E (sum of M) | was ~196 in #773 |
| host plans if left on the critical path | +~8.8 | M | #784: T2 plan 6.6-7.7 CPU + 1.1 upload; #793 moves it off |
| t2sh pre/post (SH bands 1-15) on the hero | not measured | - | #756: 0.019/0.028 ms on a 64x64 crop |
| densification, dump-free loop | not built | - | #778: T1/T2 inputs still come from Python prep of dumps |

Caveats: every hero number above comes from a dump without the 3DGS Jacobian clamp. #797 adds the
clamp; the trainer is re-baselined after it (follow-up F6). The sh3 dL/dmeans fix (#795) passes
the host gate (1.4e-5); its device rerun is #798. Published GPU steps for scale (P): 3DGS 55.9,
Faster-GS 8.3, FastGS 3.84, LiteGS ~1.1 ms/iter (PLAN-single-chip §2).

T1 is now 68% of the device time, forward 19%, T2 12%.

## 2. New Mac evidence (pair_model.py, hero view, bicycle.ply, 1,809,729 visible rows)

`PYTHONPATH=$PWD CAMS=<cameras_test.json> python3 docs/training-plan/t799/pair_model.py <dir with docs/t685/hero.py + jclamp>`
writes pair_model.json. Pairs are (Gaussian, 32x32 tile). "tight" = the Gaussian's largest alpha
anywhere in the tile (exact minimum of the conic over the tile rectangle) is >= 1/255, so at least one
pixel can use it.

| pair list | pairs | rows k>16 | heavy slots (k>16) | light planes, kcap 16, rows in order | same, rows grouped by k |
|---|---:|---:|---:|---:|---:|
| today: 3-sigma rect, no jclamp | 4,490,511 | 2,990 | 1,246,250 | 191,492 | **51,449** |
| opacity-aware rect, no jclamp | 4,007,978 | 2,365 | 1,059,549 | 160,272 | 46,668 |
| tight alpha, no jclamp | 3,162,197 | 1,510 | 537,960 | 127,458 | 41,391 |
| 3-sigma rect + jclamp 1.3 (#797) | 3,300,478 | 1,658 | 56,377 | 187,102 | 51,112 |
| opacity-aware rect + jclamp | 2,982,001 | 1,188 | 34,269 | 155,656 | 46,362 |
| tight alpha + jclamp | 2,637,695 | 506 | 13,879 | 122,172 | 41,132 |

Findings (E, Mac model):
1. jclamp removes 26.5% of hero pairs and 95% of T2 heavy slots (1.25M -> 56k), but the longest
   row stays 1,014 slots in every config, so the phase-A floor row (#793) stays.
2. The model's 191,492 light planes matches #784's device plan (181,694) within 5%. Light planes are
   set by the longest row in each group of 64 consecutive rows; **grouping rows by capped k first
   cuts them 3.7x (191k -> 51k)** at the same slot order inside each row.
3. 20% of jclamp-rect pairs (3.30M -> 2.64M) and 30% of today's pairs cannot reach alpha 1/255 at
   any pixel of their tile.
4. From runs/2188/t1_dead.npz (#776, pre-jclamp): cutting each tile's list after its last pair with
   a live vector keeps 2,209,295 of 4,478,225 pairs (49.3%) and 51% of vector visits; tile median
   3,120 pairs, last-live median 785.

## 3. Ranked levers not covered by open tasks

Open tasks already cover: #780 (T1 UNPACK staging), #787 (T1 dead-vector skip A/B), #788 (train
forward: fused program, SCHED, T early-out, fast exp), #792 (T2 kcap 8, default), #793 (T2 long-row
split + host plan off the critical path), #796 (LPT default in prep), #797 (jclamp), #798 (sh3 device
rerun, hero loss oracle). Bounds below are on top of those unless stated. Cycle costs from #772:
T1 setup 172 + finish 113 cyc/pair at ~1.34 GHz = 0.21 us/pair; 110 cores.

| # | lever | stage, ms now | bound (E) and reasoning | lossless | device-free prep | ledger |
|---|---|---|---|---|---|---|
| 1 | **T2 k-grouped row order**: sort light rows by min(k, kcap) before cutting 64-row groups; slot order inside each row unchanged | T2 phase B 7.27 | **-5 ms**: phase B scales with planes (#784: 3.2/2.0/3.35 us per plane per core); 191k -> 51k planes gives 7.27 -> ~2.0 | yes, bit-equal (per-row sums unchanged; rows are independent; Adam is per row) | yes: Python mirror + host plan code + unit test of plane count and per-row order | new; not in any task or shelf |
| 2 | **T1 per-tile stop at the last contributor**: T1 runs each tile's list only up to the last pair any pixel used (3DGS last_contributor) | T1 51.95 | **-4 to -5 ms** on top of #787: 2.27M pairs x 0.21 us setup+finish / 110 cores = 4.4 ms; more if #787 skips less than modelled. jclamp may shrink the dead tail | yes: pairs after the last contributor get zero gradient in the oracle too | yes: stop index from the forward T plane (bench: from dumps), T1 reader count change, LPT weights from truncated counts, unit test on t1_dead.npz | new; #788 stores a per-pixel stop for the forward, this consumes its tile max |
| 3 | **Exact per-pair alpha test in the training pair list** (B2 for training) | T1 + T2 + sort | **-1 to -2 ms**: 20% of jclamp pairs; part overlaps lever 2 (dead tails) | yes: dropped pairs have alpha < 1/255 at every pixel, the blend skips them anyway | yes: CPU reference + TA flag + test that image and T0 gradients are identical | B2 shelved for render (#538, #666: 6.8% dead, ~0.05 ms). **New evidence:** in training each pair costs ~285 cyc in T1 plus a T2 slot, and the training list (3-sigma rect) has 20-30% dead pairs |
| 4 | Opacity-aware rect (3DGS extent) for training lists | same | -0.5 to -1 ms; 3.30M -> 2.98M pairs | yes (same argument, weaker cull) | yes | cheaper subset of 3; take only if 3 is too costly in TA |
| 5 | **Dump-free trainer loop** (forward -> loss -> T1 -> T2 on device buffers, T1/T2 plans built from device outputs) | whole step | no ms gain by itself; it is what makes `train step ms/view` a real M number and exposes hidden host gaps | yes | partly: host-side plan builders and buffer wiring with unit tests on dumps; the loop itself needs the device | PLAN-single-chip M1; no open task builds it |
| 6 | **Budgeted densification / final N** (T8b) | all stages | largest lever: forward 7.9 -> 3.2 ms render at 1M (D1, M); T1/T2 scale with pairs (pair ratio ~0.39 at 1.2M, D1) | n/a: changes the model; gated by Q1 quality (3DGS 25.14 dB P) | yes: CPU reference of the 3DGS default schedule then a budget mode, unit-tested on small scenes | PLAN-single-chip T8b, no task yet |
| 7 | Re-baseline the hero breakdown after jclamp + sh3 (+ t2sh pre/post on the hero) | measurement | fixes every number above (Mac model: -26.5% pairs) | - | Mac part done here (pair_model); the rest is device | needed before levers 2-4 are A/B'd |

Path to the target, in order of size: final N (lever 6) and pair count (2-4, jclamp) shrink every
stage; T1's per-pair cost is the next wall (51.95 ms for 4.48M pairs is ~11.6 ns/pair; 1-2 ms at
~1.5M pairs needs ~1 ns/pair, about 10x). That 10x is not in sight from any measured lever; it needs
#780/#787 results first.

## 4. Not queued (by the spec) and not re-proposed

Held back until below ~5 ms/view: T7e (SSIM per-tile reads), t2sh-pre fold into T2. Held until
#792/#793 land: B1 full route mode, LPT + microblock split k <= 2 (#785 split: 50.88 vs 51.95,
shelved). Still shelved: sort warm start (#688, <= 0.04 ms), batch-union pre-cull (parked), any
stale or approximate cross-view order or cull in training (msg #384), and everything in
PLAN-single-chip §9.

## 5. Follow-ups: code-only prep, each with a later device A/B

| prep (no device) | device A/B after | depends on |
|---|---|---|
| F1 T2 k-grouped row order | F1b hero A/B, bit-equal outputs | F1, #792, #793 |
| F2 T1 per-tile stop index | F2b hero A/B | F2, #787, #796 |
| F3 exact alpha test for training pair lists | F3b hero A/B (pairs, T1, T2, forward) | F3, #797 |
| F4 host plan builders for a dump-free loop | F4b first measured dump-free train step | F4, #793, #796 |
| F5 densification CPU reference (default, then budget) | later, after Q1 setup | - |
| - | F6 re-baseline hero breakdown with jclamp + sh3 + t2sh | #797, #798, #792, #796 |

Source data: t773/PLAN.md, docs/t784/RESULTS.md, runs/2188 (t1_dead), d1-scaling/RESULTS.md,
hand-offs #772, #776, #778, #781, #784, #785, #753, #756, #795, #666, #688.
