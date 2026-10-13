# Structural plan 2: what is left after B3 + B4a (task #578)

2026-10-12 PDT. Plan only: no device runs. Sources are existing profiles, code and CPU models.
Base: best-iter-226, **8.615 ms/view b2b** (bh-33, p100a, 110 cores, worker dispatch, tip
9cb7385b; t574 stack fix 8.595 vs 8.619, noise level). GPU reference: 10.75 ms/view (INRIA 3DGS,
RTX A6000, bicycle 1080p, **published, not measured**); today 1.25x.

Excluded (shelved or landed; not re-sized): B1 #534/#289, B2 #538, sort reuse #529 (also the
viewer-only variant), B4' sort-in-blend #573, #517, metal trace replay #526, B3 opacity prune
(iter 225), B4a K2 fold (iter 226), chunk frustum cull (#433/#470/#493, shelved on p150; on the
p100a its 0.525 ms pfwc saving was eaten by Morton-gid blend/pfwc slowdowns, #470), VIS_FAST
(#494), r16 records, FPU quadratic form (#111/#511). The ETH-dispatch fold hang is its own task.

## Where the 8.6 ms goes (t224 profile, scaled by B3/B4a)

| stage | t224 ms | model | after B3+B4a (approx) |
|---|---|---|---|
| pfwc (all N = 6.13 M) | 1.654 | 1.195 + 0.274*M | ~1.62 |
| k2 (tile assign) | 0.647 | per pair | folded into blend BRISC |
| sort | 1.459 | 0.269 + 0.455*P | ~1.43 |
| mat+blend | 5.330 | 1.355 + 1.521*P | ~5.5 (incl. K2(N+1)) |
| gaps + end spread | ~0.04 gaps; end spread 0.105/0.092/0.146 per program | | ~0.38 |

M = 1.76 M visible, P = 2.72 M pairs. Pair cost ~2.2 ms per million pairs. FPU ~0 %; DRAM
~1 % of roofline except pfwc (368 MB/view). The blend is SFPU-bound (~60 ops per live
record body, exp_21f ~10 of them). One chip's remaining in-program slack is about 0.4 ms.

## Lever table

| # | lever | bound ms/view b2b | evidence | cost | risk | hardware / ask |
|---|---|---|---|---|---|---|
| L1 | **Headline on a free non-viewer p150, ETH dispatch 12x10, fold on** | 0 to -0.95 (sign not proven) | viewer t556: p150 ETH fold-off 8.00 device ms vs p100 fold-off 8.99 b2b (iter 225); 120 vs 110 cores; different modes, so unproven | 0.5 day (measure only) | box availability; fold hang fix must land first | free p150 among bh-48/49/50 (renew.sh measure); the charter already names p150 |
| L2 | **Alternate-frame rendering (AFR) on 2 chips**: chip A renders even views, chip B odd | -4.0 (period ~4.4-4.6 = 8.615/2 + host serial ~0.1-0.3) | b2b is a throughput metric; per-view device work is independent (B4a fold carries only N->N+1 on the same chip, so each chip folds its own next view N+2) | 4-6 days (two DeviceStates, one host thread per device, view router) | host serial work and PCIe contention; viewer link (12 MB/s at 118 FPS now) may cap page FPS; viewer latency unchanged (8.6 ms), one extra frame in flight | **ask**: 2 boards on one host (bh-30 has 4 but is the viewer box; bh-48/50 board counts unknown) or a p300 card. Must be labeled "2 chips" against the 1-GPU published number (user decision) |
| L3 | **Screen split across 2 chips** (tile rows interleaved per chip, host merges) | -2.3 to -3.0 (per chip ~5.6-6.3: pfwc 1.44-1.62 not halved + sort 0.89 + mat+blend 3.42, +5-10 % imbalance; merge 1.5 MB/chip D2H overlapped) | t224 stage models above; pfwc fixed cost 1.195 does not split | 2-3 weeks (two devices, tile partition in sort/K2, per-chip cull, merge, viewer path) | balance; merge on the critical path in latency mode; same labeling issue | same ask as L2 |
| L4 | **Persistent per-view program** (sort(N); pfwc(N+1); mat+blend(N) as one launch with device semaphores; pfwc chunks claimed dynamically) | -0.38 ceiling, -0.15 to -0.25 realistic | end spread 0.105+0.092+0.146 + gaps 0.04 (t224); B4' showed a shared-core merge loses (+2.84), so only the boundaries are on the table | 7-10 days | high: kcfg already overflowed at 117 KB vs 105 KB (SIB); BRISC stack margin 368 B; ETH hang interplay | none |
| L5 | **Precision/work per pixel under the bar** (exp degree-1, T_EPS 1/128, coarser T period, alpha floor 2/255) | ~-0.04 (exp) to ~-0.1 (all), most fail the bar | bar 42.4 vs 42.51 dB allows only +2.6 % MSE (~0.3 LSB RMS); t260 shows the floor-dropped haze dominates the error; exp is ~10 of ~60 ops/record | 1 day CPU model | almost certain no-go | none |
| L6 | **Newton step on pfwc 1/z and conic** (quality headroom, not speed) | +0.0 speed; +~0.2 dB headroom (t260: residual ~47 dB term is device projection numerics) | t260 compare_proj: 1.5e-3 rel 1/z error, means up to 0.84 px | 1-2 days | small pfwc cost (pfwc is TRISC-bound, t484) | none; only useful if it unlocks L5 |

Ideas checked and dropped: FPU colour-accumulate matmul (dense padding vs ~13 % dispatch density,
#148); bf16 SFPU math (no packed lanes, no throughput); scene-level pre-filter of never-visible
gaussians (60 % of N over the 30 views): bench-only, would be gaming; splitting pfwc and blend
across disjoint core groups (moves no core-ms, both SFPU-bound).

## Reading

- One chip has about 0.4 ms of slack left between programs (L4), plus whatever the p150 gives
  (L1). The per-pair SFPU work is the floor of everything else and the quality bar is too tight
  to trade precision for it (L5).
- The only lever that changes the order of magnitude is a second chip (L2/L3). AFR is the cheap
  way to buy throughput; the screen split also cuts latency but costs 3x more and saves less.
  Both need hardware the project does not hold and a labeling decision against the 1-GPU number.

## Ranked follow-ups

Shared acceptance (each device follow-up): keep only at >= 0.1 ms/view b2b faster than
best-iter-226 (or the current best), median of >= 3 alternating A/B passes of
`render/run.py --back-to-back` over the 30 bicycle views; md5 906e0435 or hero PSNR >= 42.4 dB vs
`benchmarks/reference_v2/hero.png`; real device hero.png + diff image + PSNR, looked at by eye
for tile seams; a kept iteration adds iters.jsonl -> REPORT.html, tag best-iter-<N>, and
`python3 tt-project/harness/watchers/verify_main.py <N> --ff` ('main fast-forwarded' only on its
OK line), then a viewer deploy + page-FPS task.

1. **L1 p150 headline A/B** (after the ETH fold fix lands). Same tip, bh-33 p100 vs a free
   non-viewer p150 with ETH dispatch, fold on. Gate: report both; switch the headline box only if
   the p150 is >= 0.1 ms/view faster and its box is reliably obtainable.
2. **L2 AFR feasibility, CPU/host only** then device: design note + host code path that opens
   two devices behind `GSPLAT_TT_AFR=1` (default off), compiled and unit-tested on CPU. Device
   step waits for the 2-board ask. Gate on device: b2b period <= 5.0 ms/view, md5/PSNR per chip
   equal to single-chip.
3. **L4 step 0: persistent-program size feasibility** (no device): from the JIT cache ELFs of
   best-iter-226, sum the code + rodata of the three programs' kernels per RISC and the BRISC
   stack chain (`opt/brisc_stack_check.py`). Go only if kcfg fits with >= 4 KB margin and BRISC
   stack margin >= 256 B. Then L4 step 1 (device): build behind a knob; gate as shared.
4. **L5 CPU model** (no device): extend docs/floor-ab-t260/model.py with exp degree-1, T_EPS
   1/128, alpha floor 2/255, alone and with the L6 Newton step, on device-dumped inputs. Go to a
   device A/B only for a variant at >= 42.4 dB modelled with >= 0.1 ms modelled saving
   (ops x records). Expected: no-go; this closes the precision question with data.
5. **L3 screen split**: only after L2 runs on 2 chips and the user accepts multi-chip numbers.

## Proposed microbenchmarks (not run)

- MB1 (L2): two-device open + alternating enqueue of an empty program, measure host serial ms
  per view and PCIe D2H contention for two 3 MB images.
- MB2 (L4): device semaphore hand-off between two kernels on all 110 cores, measure cost per
  boundary (target < 10 us) vs the current program-launch gap.
- MB3 (L1): b2b with `--dump-views` off on a non-viewer p150, fold off vs on, after the fix.

## Restrictions every follow-up spec restates

Own git worktree of ~/dev/gstt2, never edit ~/dev/gstt2 itself. Every device sync+build+run
inside `ttp lock p100 -- ...`. Never reserve or release the measurement machine (only the
designated task does); never use the viewer box bh-30 for measurement unless it is the only p150
and the viewer exception applies (tell the user). Git: push only to smarton/tt-project-opt, main
(fast-forward or merge commit) or new ttp/* branches; no PRs, no force push, no rebase/amend of
pushed commits, no branch/tag delete or move; kept iterations tagged best-iter-<N>. No host
scanning, no exploratory logins, never disable SSH host-key checking; IRD via the jump host
only. GPU numbers are published, never measured. Never search / or ~. Every iteration carries a
real device screenshot (hero.png, diff, PSNR vs benchmarks/reference_v2/hero.png), checked by eye.
