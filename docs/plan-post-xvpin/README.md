# Plan: next levers after xvpin (task #406, no device)

Date: 2026-10-07. Input: existing measurements only. No device was used.

Starting point:

- xvpin (cross-view overlap + pinned output) becomes the default as iteration 211 (#404/#405).
- Measured ms/view: 8.934 on the p100a (#388) and 9.176 on the bh-30 p150 (#394).
- GPU reference: 10.75 ms/view, **published, not measured**.
- Stage means on bh-30 with xvpin (#394, `docs/xview-overlap-t379/p150/README.md`), in ms:

  | project | sort | blend | d2h | xview |
  |---|---|---|---|---|
  | 1.164 | 1.043 | 6.436 | 0.292 | 0.131 |

- The stop gate is ~1% = 0.09 ms/view. Levers below it are listed as rejected.

## 1. A fresh Tracy capture of xvpin is needed first

The last device trace is iteration 210 on bh-30 (#382, `docs/p150-tracy-210`), taken before xvpin.
Its numbers no longer add up to the xvpin wall time.

The iter-210 traced device chain, in ms from pfwc start (#382):

| stage | window (ms) |
|---|---|
| pfwc | 0 – 1.835 |
| k2_pairs | 1.905 – 2.719 |
| k2_rows | 2.354 – 2.725 |
| sort prefix / fill | 2.740 – 2.866 |
| emit | 2.838 – 4.126 |
| mat_cull_mask | 4.789 – 7.611 |
| blend | 6.599 – 10.836 |
| last writer | 10.913 |

- Untraced, the device span is about 10.25–10.30 ms. Device idle was ~0.70 ms/view.
- xvpin can only hide device time behind *host* time. It does not shorten the device chain.
- Yet the xvpin wall (9.156 ms avg_frame, r2 log) is ~1.1 ms **below** that device span.

Two explanations fit the numbers:

1. **(a) The pfwc stage was mostly launch cost.** pfwc on the device is much shorter than its traced
   1.835 ms window: dispatch, binary load or the first-core launch were inside that window, and now
   overlap blend N's tail.
2. **(b) Programs now overlap on the device.** For example, pfwc N+1 runs while the last
   blend-N cores drain. The tail spread is mat_cull_mask max 2.821 vs mean 2.286, and blend max
   4.093 vs mean 3.682.

Under (a), the device critical path is now **K2 → sort → host bridge → mat+blend**. Under (b), the
blend tail is partly hidden already, and tail-trimming levers are worth less.

The current critical path decides which of the levers below are real. That makes a chunked xvpin
Tracy capture **task 1**. It is cheap: the tooling exists in `opt/profiler/capture_tracy_chunked.sh`,
`stitch_device_csv.py`, `matblend_cores.py`, `program_gaps.py`, `overlap_concurrency.py` and
`docs/p150-tracy-210/ana210.py`.

Rules for the capture:

- Use **10-view chunks and no `--dump-device-data-mid-run`**. A mid-run dump inserts ~46 ms between
  views (#382), which would break the xview overlap: no XVIEW hit, so you would be tracing the base
  path.
- Check that XVIEW_HITS is 9/10 per chunk.

## 2. Where the time is (from existing data)

### Fused mat+blend program (~6.05 ms window on bh-30 iter 210, 110 cores)

Source: #382 `cores-bh30-210.txt`, t319 `docs/fill-zones-t319.md`.

- **Mat phase per core.** Mean 2.286 ms, max 2.821 ms.
  - The movers (NCRISC/BRISC) per-tile depth radix sort takes 1.25 / 1.34 ms per core.
  - Permute takes 0.370 ms, and done-wait 0.355 ms.
- **Mat-phase TRISC usage.**
  - TRISC1 band cull is busy 1.187 ms and **idle 1.104 ms**, waiting on the movers.
  - TRISC0 fill is 52% busy.
- **Blend phase per core.** Mean 3.682 ms, max 4.093 ms.
- **TRISC1 total work.** 1.187 + 3.682 = 4.87 ms, against a 5.97–6.05 ms window. That leaves
  **~1.1 ms of TRISC1 idle per core, all of it inside the mat phase**.

### Blend per record

Source: t205 `docs/blend-loop-model-t205`, t231 `docs/blend-decode-ahead-t231`, #382.

- 1 cycle per live record costs 0.0158 ms/view.
- 3.682 ms / 0.0158 ≈ **233 cycles per live record**. The SFPU-only bound is ~210: a 60-op body,
  MAD stalls already at 0 under BLEND_SCHED=2, plus the walk and scan.
- So the overhead left above the SFPU bound is ~23 cycles/record ≈ 0.36 ms per core.
- Most of that overhead is the decode/staging that U2 (shelved) and emit-on-idle-TRISCs (shelved)
  went after.
- **Cutting per-record SFPU ops without changing bits is exhausted.** The exp and the
  quadratic-form arithmetic are fixed by md5 906e0435. FPU QF and LOADMACRO B-full are not
  md5-safe (conclusion.md, t205).

### Blend waste

Source: t148 `docs/blend-waste-t148`.

- 27.8% of records are dead (0.244 ms). The record-level drop is shelved (mask-0 record drop).
- 5.4% of dispatches go into saturated microblocks (0.145 ms).
- **Only 32.5% of the lanes in dispatched microblocks are live.** That is SIMD granularity: a 4×8
  microblock is dispatched if any of its 32 pixels overlaps the Gaussian.

### Host bridge

Source: #394 r2 log, #382.

- Per view: bin_layout 0.082 ms and publish_host ~0.32 ms, plus the K2-rows read and blend setup.
- Untraced, mat started only ~0.27 ms after Enqueue#3. So the bridge was **nearly critical** at
  iter 210.
- With the device chain changed by xvpin, it may now be critical. That is unknown until task 1.

## 3. Ranked levers

Ranked by modelled gain. "Bound" is the most the mechanism can remove. "Realistic" applies the
#306 measured/modelled conversion factor of 0.40 (fill model, `docs/conclusion.md`) unless stated
otherwise.

### L1. Give the mat-phase idle on TRISC1 to blend work, or remove it (bound 1.10 ms, realistic ~0.45 ms)

There are two mechanisms for the same ~1.1 ms/core TRISC1 idle. They are **not additive**. Task 3
models both on the task-1 trace and builds the better one.

**L1a. Tile-boundary mat/blend interleave on all three TRISCs.**

- Ready flags (t273, iter 206) already mark per-tile mat completion. Today a core still runs its
  whole mat job list, then its whole blend list.
- With the interleave, when TRISC1 would wait on a mover (no cull job ready) and some tile T of
  this core is ready, the three TRISCs switch to blending T, then return to the mat jobs.
- Switches happen only at tile boundaries. The blend state (R, G, B, T, ramps in DEST) is born and
  dies within a tile, so **there is no DEST save/restore**. That is the difference from shelved
  Lever A (t311): a TRISC0-only lane that yields per fill job, with a 2–5 µs DEST spill per yield.
- It is also not the t147 fused-pipelining model. That model put the cull on the blend critical
  path and needed a second mover slab. L1a keeps one slab and uses only already-materialized tiles.
- Ordering inside a tile is unchanged, so every pixel sees the same records in the same order.
  The **md5 906e0435 risk is low**: only scheduling changes.
- **Risk: deadlock and protocol complexity.** TRISC0 fill and TRISC2 patch must agree on the mode
  switch through CB state. That is the KI-2 class of bug.
- **Model.** gain_per_core = min(Σ TRISC1 idle gaps during mat that are ≥ one tile-blend duration,
  blend work of tiles already ready at that point). The frame gain is the change in the
  **max-core** end time, not the mean.
- **Arithmetic.** The upper bound is 1.104 ms (t319 idle). Tiles become ready mostly late in mat:
  the big-tile chains on row 2 run to 2.3–2.8 ms. So only part of the idle has ready work. 0.40 ×
  1.10 = **~0.44 ms realistic**. Even 0.1 of the bound clears the gate.

**L1b. Remove the per-tile depth sort: global depth presort, then stable bin.**

- If the visible Gaussians were sorted by depth once, before the stable one-launch bin/emit, each
  tile's record list would already be in depth order.
- The mat movers would then skip the per-tile radix sort (1.25–1.34 ms/core) and most of the
  permute (0.37 ms). Mat would just read and cull.
- **Cost.** One global radix sort of the visible keys, ~1–2M (P = 2.05–3.19M records/view), on the
  device before emit. It would be a new sort_ol-style pass, estimated at 0.3–0.5 ms on 110 cores.
  This is a guess to be checked against the sort_ol prefix/fill timings in task 3.
- **md5 risk: medium.** The per-tile sort is a *stable* LSD radix on (key − kmin)
  (`render/kernels/dataflow/sort_radix_tile_algo.h`). Ties keep emit order. The presort must
  reproduce exactly the same tie order: stable by depth key, with ties by the current emit order
  (gaussian index). It must also use the *same* key bits. Any tie difference flips md5.
- **Arithmetic.** The saving is in mover time, ≤1.6 ms/core (sort + permute). The frame gain is
  capped by the TRISC1 idle it frees (1.10 ms), minus the presort cost on the serial chain
  (0.3–0.5 ms). That gives 0.6–0.8 ms bound and **~0.25–0.3 ms realistic** (0.40 × 0.6–0.8).
- It ranks below L1a on realistic gain, with higher risk. It is listed because if the trace shows
  the movers (not TRISC1 scheduling) are the real limit, L1a cannot help and L1b can.

### L2. Device-side bin_layout / publish (conditional on task 1; bound ~0.40 ms)

- **Mechanism.** Move the host bridge's bin_layout (0.082 ms) and publish_host (~0.32 ms) to a
  small device pass right after emit, so Enqueue#3 does not wait for a host round trip.
- This is not on the shelved list. "Sort emit into owner L1" and "K2 rows early on CQ1" were
  different mechanisms.
- **Gain** = the device idle between emit end and mat start, as measured in task 1, bounded by
  0.082 + 0.32 ≈ 0.40 ms minus the device pass cost.
- At iter 210 the slack was ~0.27 ms in favour of the host, so the gain was ~0 then. Under xvpin's
  shorter front chain (explanation (a)) the bridge may now be exposed.
- **Do it only if task 1 shows ≥0.15 ms of device idle in that gap.**
- md5 risk: low, as long as the layout and publish arithmetic is reproduced exactly. The task can
  validate it against the host result buffer.

### L3. Microblock shape vs live-lane fraction (CPU model first; bound unknown, gate 0.09 ms)

- **Mechanism.** The SFPU vector is one 32-pixel microblock, currently 4×8. The host permutation
  `mb_perm_img_of_dev` makes the microblock-to-vector mapping the identity, so the shape (8×4,
  2×16, 16×2) can change without touching the per-pixel math.
- Bicycle splats are elongated and mostly small. A different shape changes the number of
  (record, microblock) dispatches, and blend time is linear in dispatches: t147 fit,
  ~0.197 µs per live dispatch.
- Today only 32.5% of dispatched lanes are live.
- **Arithmetic.** Blend is 3.682 ms/core. A 5% cut in dispatches is 0.18 ms, and 2.5% is 0.09 ms
  (the gate). The bicycle footprint distribution is not in any doc, so the model must count
  dispatches on the CPU from the real scene before any device work.
- **md5 risk: medium.** Per-pixel record order is unchanged. But T-live early stop is per
  microblock, so a pixel's last accumulated record can change unless the per-pixel contribution
  floor already zeroes everything after T < eps. The CPU model must check bit-exactness
  per pixel against the reference_v2 path.
- The band_batch cull (TRISC1) must use the same shape.

### Rejected (below the 0.09 ms gate or shelved)

- **T-readback cadence / saturated-microblock skip.** Bound 0.145 ms (t148). A cadence change
  recovers maybe half, ~0.07 ms, which is under the gate. Readbacks also cost a pack each.
- **Per-record SFPU op cuts.** Exhausted: ~23 cycles/record above the SFPU bound, and the remainder
  is shelved (U2, emit-pack on idle TRISCs) or not md5-safe (FPU QF, LOADMACRO B-full, GEMM
  formulation from tt-splat / Slack).
- **Per-pixel early termination beyond T-live.** Already in: the microblock stops once max T <
  eps, with a per-pixel contribution floor. Per-lane termination does not save SFPU time, because
  a lane is a fixed pixel and the vector op runs for all 32 lanes (t205 option E).
- **Deeper cross-view overlap (K2 + sort of N+1 in the hook).** It does not shorten the serial
  device chain. The host would only wait on sort instead of pfwc. Reconsider only if task 1 shows
  device idle during the host tail.
- **ETH dispatch on 120 cores.** In flight as #397, not re-proposed. For context: 656.5 core-ms /
  120 = 5.47 ms vs a 6.05 ms window on 110 cores.
- All levers on the task's shelved list are left out.

## 4. Proposed tasks

| # | task | depends on | device | modelled gain (ms/view) |
|---|---|---|---|---|
| 1 | xvpin chunked Tracy capture + reconciliation | — | yes | enables 3 and 4 |
| 2 | Microblock-shape dispatch model (CPU, bicycle) | — | no | decides L3 (gate 0.09) |
| 3 | L1 model on the task-1 trace, then build the winner (L1a or L1b) | 1 | yes | ~0.44 (L1a) / ~0.25–0.3 (L1b) |
| 4 | Device-side bin_layout/publish, only if task 1 shows ≥0.15 ms of bridge idle | 1 | yes | ≤0.40 |

Every device task:

- runs its sync, build and run under `ttp lock p100 -- ...`;
- uses the existing reservation only;
- keeps host-key checking on;
- includes the hero screenshot step: the bicycle hero rendered **on the device** at the task's
  commit and config, saved as hero.png, with a diff image and PSNR vs
  `benchmarks/reference_v2/hero.png`, and an eye check for tile artifacts.

## 5. Gaps

- The Slack deck referenced in `docs/slack-notes.md` (F0C1PMDG12M) is not indexed. Its content
  could not be checked.
- No per-Gaussian footprint statistics for bicycle exist in the repo. Task 2 produces them.
