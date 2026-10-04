# Tracy re-profile of the lever A+B tip, and the iter-180 tip (task #150, 2026-10-04)

Board for every number: **yyzo-bh-07, Blackhole p100a, 110 cores** (not a p150). Bicycle,
1024x1024, 30 views, `render/run.py --no-ref`. Every untraced run below was md5-identical to
`md5-r82new.txt` over all 30 views, hero_vs_ref 100.00 dB. GPU reference 10.75 ms/view
(published, not measured).

Two code points were captured:

- **b7a04a4** (iter-179, lever A + B), the spec's target. Remote tree b5e547d = b7a04a4 + the
  profiling fix below + these scripts.
- **3372664** (iter-180, #146 blend diet), which landed while this task ran. Remote tree
  = 3372664 + the same commits. It is the current tip, so the lever ranking uses it.

## Profiling fix: Tracy did not run at the lever B tip

With `TT_METAL_DEVICE_PROFILER=1` the fused pfwc program is 71,216 B. The Tensix kernel config
buffer is 70,656 B (69 KB), so `EnqueueMeshWorkload` threw `TT_FATAL: Program size (71216) too
large for kernel config buffer (70656)` in the warmup view. The untraced build is not affected.
Round 1's capture hid it because `remote_tracy.sh` filtered the log; it now keeps
`opt/profiler/<tag>/capture.log`.

Fix: `GSPLAT_TT_KCFG_EXTRA_KB=N` (`render/host/device_state.cpp`) opens the device with a worker
L1 allocator N KB smaller, so the ring buffer
(`l1_unreserved_base - KERNEL_CONFIG base`) is N KB larger. Unset or 0 means the default open.
`opt/profiler/capture_tracy.sh` sets 4. The default-env run after the change (round 3) was
21.20 ms/view and md5-identical.

## Frame time (untraced, ms/view, yyzo-bh-07 p100a)

| code | arm | ms/view | FPS | md5 30/30 |
|---|---|---:|---:|---|
| b7a04a4 | default (r1) | 21.4 | 46.7 | identical |
| b7a04a4 | host profile (r1) | 21.2 | 47.2 | identical |
| b7a04a4 + KCFG fix | default (r3) | 21.20 | 47.2 | identical |
| b7a04a4 | `GSPLAT_TT_PFWC_FUSE=1 GSPLAT_TT_PUBOC_PRE=0` (r2) | 25.06 | 39.9 | identical |
| 3372664 + KCFG fix | default (r4) | 19.7 | 50.8 | identical |

**Device-pack arm (review #149 add-on):** with packs built on the device the fused path is
still byte-identical over 30 views after bank batching. It costs +3.86 ms/view, all in
project (`gather_wait` 8.26 vs 4.41 ms). The host-prebuilt default stays.

## Device timeline (Tracy, 30 views, ms per view, yyzo-bh-07 p100a)

Window = first start to last end over all cores. Busiest RISC = the RISC whose busiest core is
longest (its mean core / busiest core). t121 was a 10-view capture.

| program | t115 a03bd1e | t121 lever A | t150 b7a04a4 | t150 3372664 | busiest RISC at 3372664 (mean / max core) |
|---|---:|---:|---:|---:|---|
| pfwc (fused writer since lever B) | 2.46 | 2.46 | 2.93 | 2.93 | BRISC 2.70 / 2.93; all 5 RISCs 0.95-1.00 of window |
| proj scan + scatter | 3.76 | 3.71 | gone | gone | |
| idle before segment K2 / TA | 0.42 | 0.79 | 0.01 | 0.01 | |
| segment K2 (TA scatter) | 1.42 | 1.40 | 1.51 | 1.51 | NCRISC 1.44 / 1.51 |
| idle before sort (host bridge) | 0.06 | 0.09 | 0.55 | 0.53 | |
| sort (t115: hist + bridge + emit + radix + publish; then one-launch) | 10.51 | 4.67 | 4.70 | 4.71 | BRISC 4.54 / 4.71 (emit 3.70, count 0.93, barrier 0.45) |
| idle before materialize (host) | 0.08 | 0.33 | 0.31 | 0.31 | |
| materialize + cull | 3.17 | 3.77 | 3.75 | 3.75 | NCRISC 3.07 / 3.75; TRISC 3.14 / 3.75 |
| idle before blend | 0.01 | 0.01 | 0.01 | 0.01 | |
| blend | 7.52 | 7.53 | 7.59 | 5.96 | TRISC 5.42 / 5.89 (SFPU); NCRISC 5.17 / 5.48 |
| **all-core idle** | 2.00 | 1.27 | 0.88 | 0.85 | |
| **device span** | 29.45 | 24.82 | 21.37 | 19.72 | |

At 3372664 the busiest single RISC (NCRISC, busiest core) is busy 17.66 ms of the 19.72 ms span.
The other 2.06 ms is 0.85 ms of host bridges plus per-program ramp and core imbalance.

Load imbalance (busiest core minus mean core, same RISC) at 3372664:

| program | RISC | imbalance ms |
|---|---|---:|
| materialize | NCRISC | 0.68 |
| materialize (cull) | TRISC | 0.61 |
| blend | TRISC | 0.46 |
| sort_ol_count | BRISC | 0.31 |
| pfwc | TRISC | 0.22 |
| sort_ol_emit | BRISC | 0.16 |

## Next levers, ranked by measured upper bound (tip 3372664, 19.72 ms device span)

| # | lever | upper bound ms/view | basis | status |
|---|---|---:|---|---|
| 1 | **#147 fuse materialize into blend per core** | **~2.9** (3.75 window + 0.31 idle - ~1.2 cull TRISC work that moves into blend) | this capture's windows; cull cost 1.2 ms from #134 / docs/blend-analysis | queued, deep. Largest lever; also removes the 0.61-0.68 ms materialize imbalance |
| 2 | Host bridges to device: build the sort layout and the materialize worklist on the device | 0.84 (0.53 + 0.31 idle) | this capture | fold the 0.31 part into #147 (device worklist, #134 add-on); the 0.53 before sort is new |
| 3 | Blend tile LPT calibration | 0.46 (TRISC imbalance) | this capture | small; after #147, since fusing changes the per-core cost model |
| 4 | Lever C (#142) dead-record pre-cull | 0.42 measured on the 24.5 ms tip | #142, old tip | running; shrinks K2, sort and materialize, so it compounds with 1 |
| 5 | sort_ol_emit body (largest window no task targets) | <= 3.70, realistic unknown | this capture | needs an emit zone split before a task |
| 6 | #148 blend waste counters | 0 by itself | measurement | queued; worth running only if #147 leaves blend > ~5 ms |

pfwc (2.93 ms) runs all five RISCs at 0.95-1.00 of its window, so no single-RISC fix helps it;
it can only shrink by doing less work, e.g. chunk frustum cull (t115 lever G, <= 1.5-2 ms
estimated, not measured).

## Files

- `drive.sh` .. `drive4.sh` (Mac drivers, one `ttp lock p100` per device step),
  `remote_time.sh`, `remote_tracy.sh`.
- `out/`: b7a04a4 rounds 1-3 (`run-r1-*.log`, `hp-r1.txt`, `run-r2-nopre.log`,
  `run-r3-base.log`, `gaps.txt`, `roofline.txt`, `zones.txt`, `zone_occupancy.txt`, `capture.log`).
- `out/iter180/`: the same for 3372664 (round 4).
- Captures (`render.tracy`, `dev30.csv`) stay on the box:
  `yyzo-bh-07:/localdev/smarton/gstt2-t150/opt/profiler/{t150-tip,t150-i180}/`.
