# Structural levers: measured bounds and build plan (task #530)

Written 2026-10-12 (PDT) from best-iter-224 (ad647d8e). Plan only: no device runs.
Every number below is from a measured doc named in the Source column. "Upper bound"
means the time the lever could remove if it worked perfectly. "Realistic" is my
estimate after the known costs; it is not a measurement.

## Where we are and the target

| | ms/view b2b | FPS | Kind |
|---|---|---|---|
| Project start (CPU-hosted pipeline) | 173.30 | 5.8 | measured |
| best-iter-224, bh-33 (p100a, 110 cores) | **9.131** | 109.5 | measured (#528) |
| G1: INRIA 3DGS, RTX A6000, bicycle 1080p | 10.75 | 93 | **published, not measured**; already beaten |
| G2: G1 pixel-normalized to 1024² | **5.44** | 184 | **published-derived** |

G2 is the "far below" target: it needs **-3.69 ms/view** from today.

Where the 9.131 ms goes (#528, docs/profile-t224/PROFILE.md @377b0ab7, steady state):

| Stage | ms | Model (M = visible gaussians, P = tile pairs, millions; medians M 1.76, P 2.72) |
|---|---|---|
| pfwc (project, frustum, cov, write) | 1.654 | 1.195 + 0.274·M (72 % fixed; reads 368 MB at ~300 GB/s) |
| k2 (tile pair rows) | 0.647 | -0.021 + 0.255·P |
| sort (count, prefix, emit scatter) | 1.459 | 0.269 + 0.455·P |
| mat + blend | 5.330 | 1.355 + 1.521·P (blend SFPU 3.667 mean / 3.938 max; mat TRISC1 1.420 / 2.039) |
| host gaps | 0.040 | |

Other measured facts that shape the plan:
- Each pair costs **2.23 ms per million pairs** across k2 + sort + mat+blend. Any cut in P pays in three stages.
- FPU is ~0 % busy. Nothing is DRAM-bound except pfwc's fixed term (its 368 MB floor is ~1.19 ms).
- End-of-stage idle (core spread) is 0.495 ms; host gaps 0.04 ms. The host is off the critical path.
- During mat+blend, BRISC is idle ~3.9 of 5.2 ms; during sort the TRISCs are idle (1.37 ms mean).
- Sort emit is at two ceilings at once: TRISC pack ~179 cycles/record (t215) and the chip's
  scattered 32 B NoC write rate (1.8-2.3 G/s, docs/hw-ceilings.md p6). 0.455 ms/M pairs = 2.2 G pairs/s.
- Blend: 32 microblocks of 4x8 px per tile; only 11.5-14 % of microblock slots are dispatched,
  and only 32.5 % of lanes in a dispatched microblock are live at alpha >= 1/255 (t148). ~366 cycles
  per live record (t205).
- Cross-view: visible-set Jaccard 0.935 but pair Jaccard only 0.15 (#528, #529).

## Ranked levers

Ranked by realistic gain per day of work, with image risk as tie-breaker.

| # | Lever | Upper bound (ms/view b2b) | Realistic | Cost | Image risk | Depends on | Source of the bound |
|---|---|---|---|---|---|---|---|
| 1 | **Coalesced bin emit**: stage each core's records per tile in L1, write each (core, tile) run as one burst to the same bucket slots as today; fold the count pass into pfwc | **1.2** (sort 1.459 down to a ~0.2-0.3 bandwidth floor: 87 MB of records at 300-420 GB/s) | 0.6-1.0 | ~6-9 days. sort_bin_onelaunch.cpp, sort_onelaunch_algo.h, sort_ol_town*, sort_device.cpp, writer_pfwc*.cpp, pfwc_device.cpp; tests/unit/test_sort_onelaunch.cpp | **None by design**: bucket layout byte-identical (each core's records of tile t already land contiguous at base[c][t]) | CPU model T-B (L1 budget, burst length) | #528 sort split; hw-ceilings p6 scatter ceiling; t215 pack cost; t98 coalesce bound <= 1.18 on the old pipeline |
| 2 | **Exact ellipse-tile pair test** in k2/pfwc tile enumeration (drop a pair only when no pixel of the tile can reach alpha >= 1/255) | 2.23 ms × pairs removed (M). 10 % fewer pairs = 0.61; 25 % = 1.52. **Unknown until counted** | 0.3-1.0 if the model shows >= 10 % | ~3-5 days. k2_trisc.h / k2_trisc_compute.cpp, tile_assign_bbox.cpp, sort count + emit tile walk (same test, must agree), vis_tile.h | **None if conservative**: blend already skips alpha < 1/255, so a pair it removes adds nothing | CPU model T-A (counts the pairs) | #528 slopes; PRECULL=2 already uses the opacity-aware pixel-centre rect (t157), so only the rect-vs-ellipse gap is left; 18.5 % of remaining dead blend records are ellipse corners (t148) |
| 3 | **Cross-frame overlap in one persistent program**: run k2 + sort of view N+1 on the idle BRISC/TRISC slots during mat+blend of view N | 1.46 (sort window) + 0.5 idle; TRISC-busy floor 6.85 ms gives 2.3 theoretical | 0.5-1.0 alone; **0.3-0.6 after lever 1** (less sort left to hide) | ~7-10 days. New fused program around matblend_brisc.cpp / matblend_ncrisc.cpp + sort kernels; device_state double-buffering; render.cpp b2b loop | None (same kernels, same order) | Lever 1 first (they share the sort window); kernel config size headroom (see profiling item P1) | #528 per-RISC busy; t379 xview-overlap; t198 two-cq |
| 4 | **pfwc byte cut**: hierarchical chunk frustum cull that keeps ply order (no Morton reorder), plus static opacity prune (alpha < 1/255 everywhere) with the original id carried for ties | 0.525 (chunk cull pfwc saving measured on bh-30) + 0.11 (prune, 572k of 6.13M gaussians) | 0.2-0.4 | ~4-6 days. chunk_cull.h, pfwc_device.cpp, reader_pfwc.cpp, writer_pfwc*.cpp, loader | Chunk cull: none if order kept. Prune: none if original id is the tie key; else PSNR gate | Nothing; can start now | t470 (saving 0.525, but Morton reorder cost 0.834 made it net +0.309); t493 (p150: no saving, emit burst stall +0.127, host xview +0.19); #528 opacity count |
| 5 | **Blend lane packing**: choose microblock set per record from the ellipse, not the rect; optionally a 2x16 / 8x4 shape per record | 0.43 (dead records + wasted dispatches, t148, older pipeline) | 0.1-0.3 | ~4-6 days. alpha_blend_compute_mb.cpp, mat_cull_compute.cpp, mb_cov_preconic.hpp | None if conservative (same rule as lever 2) | Shares the ellipse math of lever 2: build after it | t148, t205, t172 |
| 6 | mat TRISC1 idle fill | 0.29 | 0.1-0.2 | ~2-3 days | None | None | #528 (0.33 idle within mat TRISC1) |
| 7 | Pipelined stages on disjoint core groups | 0.53 (idle + gaps only) | ~0.2 | ~7 days | None | Superseded by lever 3 | #528: pfwc/mat/blend are per-core compute-bound, so splitting cores moves no core·ms |
| 8 | Sort reuse across views (presort / incremental) | **0 on the 30-view bench**; viewer only: 0.2-0.4 (previous-frame order + insertion repair in mat) | bench 0 | ~3-4 days | None (bit-exact repair) | None | #529 docs/sort-coherence/STUDY.md: 4 % of pairs keep order |
| 9 | Cut host round-trips (K2 totals readback, host worklist to device) | 0.04 | ~0 | | | | #528 gaps 0.040: host is already off the b2b critical path. **Drop.** |
| 10 | Different blend formulation on the FPU (quadratic form; dense GEMM per tile) | QF: 0.456 max, **net loss** (59-118 cycles added vs 32 saved per record). Dense GEMM: evaluates all 32 microblocks per record, 8-11x the lanes of today's culled dispatch | negative | | | | t111, t511 measured; t148 dispatch ratio. **Kill.** |
| — | Not a code lever: headline box on a p150 with 120 cores (ETH dispatch) | ~9 % on per-core-bound stages | | | | | t493 p150 ETH rebaseline 9.574 b2b was slower than p100a, so not a free win |

### What "different work splits across cores" buys

Tile-major vs gaussian-major is already mixed the right way: pfwc, k2 and sort are gaussian-major
(contiguous ply slices per core, DRAM-streaming), mat+blend is tile-major (LPT tile lists). The
measured core spread is only 0.495 ms, so re-splitting the same work cannot beat ~0.5 ms
(lever 7). The splits that do pay are those that put idle RISCs to work on the next view (lever 3)
or remove the global scatter between the two splits (lever 1).

### What "avoid re-sorting between views" buys

Measured: nothing on this bench. The 30 views are far apart (pair Jaccard 0.15). Only the live
viewer, with nearby consecutive poses, gains 0.2-0.4 ms (lever 8). Kept as a viewer task, not a
headline lever.

### What "fuse whole stages" buys

- pfwc + gather + sort: lever 1 is the useful part of this fusion: the records already exist in L1
  at pfwc time; writing them once as bursts instead of re-reading and scattering removes most of
  the sort window. A full single-kernel fusion still needs a global prefix barrier between counting
  and emitting (tile buckets need global offsets), so "one kernel" saves only launch gaps (0.04).
- mat + blend: already fused (matblend program, t147/t273). What is left is TRISC1 idle (lever 6)
  and overlapping the next view (lever 3).

## Can the bounds reach the target?

| Scenario | Levers | ms/view b2b | vs G2 5.44 |
|---|---|---|---|
| Today | | 9.13 | +3.69 |
| Realistic, low | 1 (0.6) + 2 (0.3) + 4 (0.2) + 3 after 1 (0.3) + 5,6 (0.2) | ~7.5 | +2.1 |
| Realistic, high | 1 (1.0) + 2 (1.0) + 4 (0.4) + 3 after 1 (0.6) + 5,6 (0.5) | ~5.6 | +0.2 |
| Sum of upper bounds | 1.2 + 2 (25 % pairs: 1.5) + 0.6 + 0.6 + 0.4 + 0.3 | ~4.5 | -0.9 |

Plainly: **G1 is beaten already. G2 (5.44) is reachable only if most levers land near their upper
bounds, and only if the ellipse test removes a large share of pairs (>= 20 %).** The likely
landing zone is 6.5-7.5 ms/view (135-155 FPS). Blend SFPU time (3.67 ms at 32.5 % lane use) is the
wall after that; only fewer pairs (lever 2) and better lane packing (lever 5) attack it, because
the FPU formulations measured as net losses.

How far each lever gets on its own (from 9.131, realistic midpoint): lever 1 → 8.3; lever 2 → 8.5
(unknown until T-A); lever 3 → 8.4 (8.7 after lever 1); lever 4 → 8.8; levers 5+6 → 8.8.

## Profiling follow-ups from #528, ranked by whether they gate a lever

- P1, **MATCULL_PROF program size overflow** (108384 > 103424 B kernel config buffer): gates
  lever 3, which puts more kernels into one program and will hit the same limit. Do it as step 0
  of lever 3, not alone.
- P2, default-off **NoC/DRAM byte counter**: useful to confirm lever 1's emit is at the NoC write
  ceiling before and bandwidth-bound after. Optional inside lever 1.
- P3, K2_PROF + OL_EMIT_PROF hang: gates nothing. Skip.

## Build tasks (in order)

Shared acceptance for every device task:
- b2b A/B on bh-33 against best-iter-224 (ad647d8e): `render/run.py --back-to-back` over the 30
  bicycle views, median of >= 3 alternating passes (A B A B A B). Latency mode is a secondary column
  only and never decides keep/shelve.
- Image: md5 906e0435 on 30/30 views, or hero PSNR >= 42.4 dB against benchmarks/reference_v2/hero.png.
- No seams: look at the hero image and the diff image by eye.
- For a kept iteration: device hero screenshot (hero.png rendered on device at that commit and
  config, diff image and PSNR against the same named reference benchmarks/reference_v2/hero.png,
  checked by eye), iters.jsonl entry, REPORT.html rebuilt, best-iter-<N> tag pushed, main moved
  forward only and verified with `python3 tt-project/harness/watchers/verify_main.py <N> --ff`.

| Task | Device? | Parallel now? |
|---|---|---|
| T-A: CPU model of exact ellipse-tile pairs | no | yes |
| T-B: CPU model of L1 budget and burst lengths for coalesced emit | no | yes |
| B1: coalesced bin emit (lever 1) | yes (A/B only) | code + bit-exact host unit test now; device after T-B |
| B2: exact ellipse pair test (lever 2) | yes (A/B only) | after T-A shows >= 0.3 ms |
| B3: pfwc byte cut, order-keeping chunk cull + prune (lever 4) | yes | yes; code now, device A/B queued under `ttp lock p100` |
| B4: cross-frame persistent overlap (lever 3) | yes | after B1 lands or is shelved; re-bound first |

### T-A: ellipse pair model (CPU, ~0.5 day)
For the 30 bicycle views, count with the current PRECULL=2 rect rule and with an exact
conservative test (max alpha of the gaussian over the tile's pixel centres >= 1/255, i.e. the
2D conic's minimum over the tile rect) the pairs P and the blend records. Report pairs removed
(median, min, max over views) and the bound 2.23 ms × removed M pairs, plus the extra per-pair
test cost in cycles. Use the existing CPU pipeline (gsplat/rasterization.py or backends/cpu_cpp),
scenes/bicycle.ply (the copy at the project root /Users/smarton/dev/gsplat_tt/scenes/bicycle.ply), benchmarks/cameras_v2.json (the cameras render/run.py uses). Go/no-go for B2: >= 0.3 ms bound.

### T-B: coalesced emit model (CPU, ~0.5-1 day)
From the same views and the current core split (sort_onelaunch_layout.h), compute per core: kept
pairs, records per tile, L1 bytes needed to stage them (32 B each) next to pfwc's CBs, the burst
length distribution if staged in chunks of K records (K from free L1), and the write time from
hw-ceilings (burst vs 32 B scatter rates). Output: the predicted emit ms and the chunk size. Go/no-go
for B1's device stage: predicted saving >= 0.4 ms.

### B1: coalesced bin emit
Stage each mover's records per tile in L1, then write each (core, tile) run to its existing bucket
slots [base[c][t] + cursor, ...) with one NoC write per run (chunked if L1 is short). Fold the per-tile
count into pfwc's writer so the count pass and its record re-read go away. The bucket layout must
stay byte-identical; prove it on the host first with tests/unit/test_sort_onelaunch.cpp extended to
the burst path, before any device run. Keep the old path behind an env kill switch.

### B2: exact ellipse pair test
Same conservative test in every place that walks a gaussian's tiles (k2 rows, sort count, sort
emit); they must agree or pairs go missing (seams). Default on only if md5 906e0435 holds.

### B3: pfwc byte cut
Chunk frustum cull over fixed ply-order chunks (no Morton reorder: t470 showed the reorder cost
0.834 ms), so emit order and ties stay the same. Avoid t493's emit burst stall and host xview cost.
Add the static opacity prune with the original gaussian id kept as the tie key. A/B on bh-33 (p100a)
where the t470 saving was seen.

### B4: cross-frame persistent overlap
Step 0: measure kernel config headroom (P1). Then run k2 + sort of view N+1 on idle movers/TRISCs
inside the mat+blend program of view N, with double-buffered device_state. Re-bound from the
post-B1 profile before building.

## Restrictions every task restates
Own git worktree of ~/dev/gstt2 (never edit ~/dev/gstt2); every device sync+build+run inside
`ttp lock p100 -- ...`; never reserve or release the machine (use the existing reservation);
code only on new ttp/* branches, smarton/tt-project-opt or main (forward only); no force push,
no rebase/amend of pushed commits, no deleting or moving branches or tags, no PRs; no host scanning
or exploratory logins, never disable ssh host-key checking; never search / or the home folder;
GPU numbers are published, never measured.
