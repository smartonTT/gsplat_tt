# Task #634: absolute-means emit gate (bh-33 p100, b2b, 30 bicycle views)

Tree: best-iter-227 + `OL_ABS_MEANS_ABL` gate (commit 486a7397). Driver: `drive.sh`,
4 rounds with rotated arm order, 20 b2b passes each, run 2026-10-13 12:39-12:45 PDT.

| arm | what | b2b ms/view r1..r4 | mean | raw md5 |
|---|---|---|---|---|
| base (0) | best-iter-227 | 8.455 8.449 8.457 8.463 | 8.456 | 1d372b94 (dump: 30/30 golden) |
| ctl (2) | subtracts run, absolute means written | 7.565 7.576 7.583 7.561 | 7.571 | a7bdd047 |
| abl (1) | subtracts skipped, absolute means written | 7.435 7.442 7.349 7.442 | 7.417 | a7bdd047 |

- abl and ctl write identical records (same md5), so ctl - abl is the clean emit saving:
  per round 0.130, 0.134, 0.234, 0.119 ms/view. Median 0.132, mean 0.154. The 0.234 round
  is one abl run that drifted down mid-run (passes 7.43 -> 7.31); the other three agree at 0.12-0.13.
- base - abl (~1.0 ms) is not the emit cost: wrong (absolute) means move gaussians out of
  their tiles, so cull and blend do less work. That is why the control arm exists.
- Not measured by this gate: the cull side (`microblock_cull_compute.cpp:420`, two soft-float
  ops per record via `add_sub_roundtrip` would become one sub) and the blend's added per-tile
  ramp offset (small: one SFPU add over two DEST tiles per tile).

Verdict: shelved. Robust emit saving 0.13 ms/view b2b (about 1.5 %), under the 0.15 ms bar
the spec set for step 2, and well under the 0.2-0.3 ms the #632 model bound predicted. A real
build touches four producer sites (sort_ol_town_compute, sort_bin_onelaunch pack_rec and
ring emit, sort_subchunk_materialize overflow gather) plus the cull and blend consumers.
Reopen only if a cull-side gate shows the combined saving clears 0.15 ms.
