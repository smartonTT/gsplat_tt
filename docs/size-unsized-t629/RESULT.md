# t629: sizing the four unsized ideas from t614

Base: best-iter-227 code + default-off probes (51111ee6), FOLD=1, bh-33 p100, back-to-back
30 bicycle views, 20 passes. Run 2026-10-13 11:49-11:58 PDT under `ttp lock p100`
(`drive.sh`, `remote.sh`), 3 rounds. Raw numbers: `out/summary.txt`; HTL split: `htl_split.py`.
Output unchanged in every arm (b2b raw_md5 1d372b94 in all 14 runs). No screenshot, no
dashboard iteration (sizing only).

Device order per view (FOLD=1): mat+blend(N) with K2(N+1) -> device idle at the view boundary
-> sort onelaunch(N+1) ~1.44 ms -> layout/pub_host H2D -> pfwc(N+2) ~1.45 ms -> mat+blend(N+1).

## Measurements

b2b median ms/view:

| round | base | htl | ts50 | ts100 | ts200 | ts400 |
|---|---|---|---|---|---|---|
| r0 | 8.450 | 8.456 | | | | |
| r1 | 8.452 | 8.461 | 8.472 | 8.564 | 8.563 | 8.742 |
| r2 | 8.433 | 8.454 | 8.465 | 8.503 | 8.586 | 8.761 |

tsN = pfwc MATH TRISC busy-waits N us per core per launch. Cost vs htl: ts50 +0.01,
ts100 +0.05..+0.10, ts200 +0.10..+0.13, ts400 +0.28..+0.31 (slope ~0.9 past the knee).
**Knee ~20-50 us**: the TRISC has only that much slack under the movers.

HTL split of the view boundary (`project_gather_wait`, 0.233-0.241 ms median, 3 rounds),
offsets from view start:

| span | median | when |
|---|---|---|
| TA/K2 host bookkeeping before the sort args (tile_assign_fused_k2) | ~90-95 us | 5 -> 95-100 us |
| `sort_early_rtargs` (SetRuntimeArgs, 2 kernels x all cores) | 18-19 us | |
| `sort_early_enqueue` | 26-29 us | sort(N+1) queued at ~140-148 us |
| `cq1_proj_m` (CQ1 waits for K2 event, reads proj_M) | 40-43 us | off the device path |
| rest of the wait | ~60 us | sort already running |

So the device sits idle from blend(N) done until ~0.14 ms into the next view, plus python
(G 0.055 ms, t614). All of it is host CPU work, no gather program (FOLD=1 folds it).

## Sized bounds

| idea | bound (b2b ms/view) | basis |
|---|---|---|
| (a) pfwc mover transfer pattern | **<= ~0.05**: not a lever alone | knee 20-50 us: an infinitely fast mover saves only the TRISC slack. Matches t598 (~0.06 measured). Both NoCs are already used (t232), bursts batched (t598). |
| (b) sort device time on the path | 1:1, but **capped at ~0.13-0.23** by the host's own path to the pfwc enqueue (t614 ps150/ps400 slack); **~0-0.05 after L-A'** below | device path blend-done -> pfwc start = idle 0.2 + sort 1.44 = 1.64 ms; host path = G + 1.42 ms. Cuts to the sort only pay until the host path is critical. |
| (c) fold gather into pfwc | **0**: already folded under FOLD=1 | no gather program runs; `project_gather_wait` is TA bookkeeping + early sort enqueue + cq1 proj_M read (split above) |
| (d) overlap TA/sort host prep with the 6.97 ms blend wait | **~0.15-0.20** | the 0.14 ms of host prep before sort(N+1) is enqueued, plus G 0.055, leaves the device idle; t614 g3/g7/g15 slope 1.00 says every us of that counts. Capped by the host path to the pfwc enqueue (~1.37-1.42 ms vs sort 1.44). |

The draft said (d) was ~0; that was wrong. The HTL split shows the sort enqueue itself waits
behind ~0.14 ms of host work at the view boundary.

## Proposed lever

**L-A' (replaces t614 L-A, whose target, gather(N+2), does not exist under FOLD=1): enqueue
sort(N+2) in the fold hook, right behind mat+blend(N+1)+K2(N+2), during the blend wait.**
- Same in-order CQ0, no persistent kernel, no new program: only moves `sort_onelaunch_enqueue_early`
  (and the K2/TA bookkeeping it needs) from the start of view N+2 into the hook of view N+1.
  Its inputs are static per scene (acc speed sums, noc_xy, num_tiles, tiles_x) or come from
  K2(N+2) on the device (P read on-device, task #198). The sort must take the b4a buffers
  K2(N+2) writes to.
- The clamp path (proj_M[2] != 0: early sort took clamped pairs) stays as is: drain, regrow,
  rerun K2 and sort at the usual place. Buffer regrow must Finish first (task #552 rule).
- Size: 0.15-0.20 ms/view b2b. Kill: < 0.10 ms/view over 2 interleaved rounds, or sweep md5
  != 077caaba, or any hang.
- Shelved-list check: not B4 (#591, persistent cross-frame overlap), not L4 (#579, persistent
  program), not "early pfwc enqueue" (t614, killed: sort, not pfwc, is the late one). It only
  reorders enqueues, like xview overlap (#379) did for pfwc.
- Cheaper fallback if the reorder is blocked: cache the static per-view host prep
  (mover_speeds / noc_xy recomputed every view, sort runtime args rebuilt for every core).
  Bound <= ~0.11 ms (the 90 + 18 us), likely below the gate alone.

Follow-on: after L-A' the device and host paths to the pfwc enqueue are balanced (~1.44 vs
~1.4 ms), so (b) sort cuts only pay together with moving layout/pub_host/pfwc-rtargs host work
(t614 phase B, 1.14 ms) into the blend wait too. Profile that after L-A' lands.

(a) note: TRISC and movers are within ~50 us of each other in pfwc, so neither a TRISC-only nor
a mover-only cut moves b2b by 0.1. Only a bundle that cuts both counts; this supports plan-4
rank 1 (RECFAST+VIS_FAST) plus a mover cut after it, not a mover lever alone.

Caveat: bh-33 load average was 9-25 during the run (other host activity); base and htl arms
agree within 0.03 ms across 3 rounds.
