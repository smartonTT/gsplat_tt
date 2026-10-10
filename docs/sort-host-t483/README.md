# t483: what the rest of the host sort stage spends (b2b)

Commit 5edea09e (branch ttp/t483-time-the-untimed-part-of-host-sort-stage) adds host timers for
the parts of the `sort` stage no leaf timer covered before (#474):

| key | what it times |
|---|---|
| sort_pre | P read -> emit: buffer grow checks + device_state registers |
| sort_log | the per-frame [SORT] stderr line |
| sort_cont_prep | blend host prep before its runtime args: buffer lookups, out-ring slot acquire (pinned image slot), parked-materialize fallback |
| sort_cont_rtargs | blend per-core runtime-arg build + SetRuntimeArgs |
| sort_cont_other | rest of the continuation wall |

They are plain `steady_clock` reads in the existing stage-timer struct (same cost as the other
leaves) and print only with `GSPLAT_B2B_ALL_STAGES=1`. Output unchanged: dump md5 906e0435 =
11x10 golden, 30/30 views (`out/md5-r9-dump.txt`).

## Run

yyzo-bh-04 (p100a, the measurement reservation, 11x10 grid). No free non-viewer p150 was held and
bh-30 is the viewer box, so this is p100a, not p150. `drive.sh` under `ttp lock p100`: sync+build,
one discarded warm-up run, 3 runs of `render/run.py --back-to-back` (1 check pass + 1 warm-up +
20 measured passes, 30 views each), `GSPLAT_B2B_ALL_STAGES=1`. Logs: `out/run-r*-b2b.log`.

b2b ms/view: 9.407 (warm), 9.406, 9.409, 9.407.

## Breakdown, ms/view (median of passes 1-21; "slow" = pass 0/1, the worst)

| key | r1 quiet | r2 quiet | r3 quiet | slow (pass 0/1) |
|---|---|---|---|---|
| sort (total) | 0.580 | 0.748 | 0.579 | 1.05-1.24 |
| sort_bin_emit | 0.177 | 0.233 | 0.175 | 0.35-0.41 |
| sort_publish_host | 0.158 | 0.229 | 0.160 | 0.15-0.24 |
| **sort_cont_prep** | **0.126** | **0.122** | **0.123** | **0.50-0.55** |
| sort_cont_rtargs | 0.030 | 0.049 | 0.031 | 0.03-0.05 |
| sort_bin_layout | 0.042 | 0.052 | 0.043 | 0.04-0.05 |
| sort_mat | 0.039 | 0.053 | 0.039 | 0.04 |
| sort_log | 0.004 | 0.005 | 0.004 | 0.005 |
| sort_pre, sort_cont_other, sort_pread | 0.001 each | | | |
| residual (sort - all leaves) | 0.001 | 0.002 | 0.001 | 0.001 |
| sort + blend | 6.89 | 6.88 | 6.89 | 6.88-6.96 |
| period | 9.408 | 9.403 | 9.407 | 9.44-9.53 |

The sort stage is now fully accounted for (residual ~0.001 ms). The new piece that moves is
`sort_cont_prep`: ~0.12 ms quiet, 0.50-0.55 ms in the slow first passes, single spikes to 0.26.
Run r2 sat at a higher host level throughout (emit, publish, rtargs all +30-60%), with the same
period.

## Why cont_prep costs what it does: the bench re-pins an output slot per frame

Each run logs 578 `OUT_RING new pinned slot` lines (`slots=4 grows=4 replaced=573`): 26 of 30
views in every pass. `render/run.py --back-to-back` keeps all 30 frames of a pass (the
byte-for-byte compare), so the zero-copy out ring (`GSPLAT_TT_OUT_ZEROCOPY_SLOTS`, default 4)
finds no free slot and replaces one: `aligned_alloc` + `PinnedMemory::Create` (pin + NoC map) +
an `fprintf` per frame. The viewer, and `--b2b-drop`, hold one frame and cycle two slots, so they
never pay this. It is a bench-only host cost.

## Is it on the critical path? Not on p100a

`sort + blend` stays at 6.88-6.96 ms whatever sort costs: when host sort takes longer, the blend
stage (host waiting for the device) gets shorter by the same amount, and the period stays at
9.40 ms. On p100a the b2b period is device-bound and all host sort time, including the pinning,
is hidden behind the device. Moving sort host work to the device or overlapping it more would save
~0 ms b2b here.

## Recommendation

1. No b2b lever in host sort on p100a: expected saving ~0 ms (device-bound).
2. Bench hygiene, cheap: stop the per-frame re-pin in b2b keep mode, e.g. run with
   `GSPLAT_TT_OUT_ZEROCOPY_SLOTS` >= views+2 (32), or have run.py copy each kept frame and drop
   its lease. Expected: `sort_cont_prep` 0.12 -> ~0.02 ms quiet and 0.5 -> ~0.02 ms on slow passes,
   period unchanged on p100a. On bh-30, where #474 saw 0.25-1 ms slow sort passes next to the
   live viewer, a per-frame pin + kernel mapping is a likely source of that jitter; removing it
   should make the bench match what the viewer pays. Needs one A/B (keep vs SLOTS=32) on a p150.
