# Task #423: remaining levers after iteration 214, and a stop line

CPU-only plan. No device runs. Inputs: #406 plan, #407/#412 xvpin Tracy (`ttp/t407-xvpin-tracy`
e1ec634a, `ana407.txt`), #413 mat job zones, #397/#409/#410 eth docs, #417 mat ramp, #419
blend-interleave model, #418 mover sort (`ttp/t418-...` e9558e4e), t169/t226/t232 docs and the
shelved list in project memory.

## Where the time goes

Iteration 214 (ETH 12x10, p150 default): **8.412 ms/view** in the #409 A/B (8.494 as logged for the
iteration) on bh-30. GPU reference 10.75 ms/view is **published, not measured**.

Traced 11x10 worker view (#412, period 10.387 ms traced vs 9.164 untraced, ~13% profiler cost):

| program | traced window | max - mean core | share (untraced, scaled to 214) |
|---|---|---|---|
| pfwc (project+fill+count) | 1.891 | 0.125 | ~1.5 ms (18%) |
| k2 (pairs/rows) | 0.829 | 0.055 | ~0.7 ms (8%) |
| sort (emit 1.29 inside) | 1.391 | 0.067 | ~1.1 ms (13%) |
| mat + blend | 6.243 | 0.144 | ~5.0 ms (60%) |

All-core idle 0.041 ms/view; host waits 6.53 ms per view on the blend event, so host work is hidden.
The device is the whole critical path, and mat+blend is 60% of it.

New since the spec: **#418 packed mover sort -0.456 ms/view on p100a** (8.273 vs 8.726 frame, 30/30
md5 906e0435), queued to land as iteration 215. Its Tracy (30 views): TRISC1 idle outside jobs
1.084 -> 0.766 ms mean, but the worst core is unchanged (1.651 -> 1.617), now held by NCRISC's
big-tile sort (`fz_mv_big`) and the 8-word perm. The perm and big-tile sort cuts are already queued.

## Levers asked about (gate 0.15 ms/view per lever)

| lever | expected ms/view | md5 risk | verdict |
|---|---|---|---|
| L1b global depth presort | net <= 0 (see below) | medium-low | **reject** |
| L2 device bin_layout/publish | <= 0.008 (measured bridge idle) | low | **reject** |
| pfwc X fusion (t226) | ~0.11 untraced (0.13 traced) | low | under gate |
| pfwc vis/precull cut (t226) | ~0.09-0.13 | medium | under gate, risky |
| pfwc FPU transform | n/a | high (TF32 not exact) | reject |
| pfwc / k2 / sort imbalance | ~0.11 / 0.05 / 0.06 | low | under gate |
| new ETH critical-path stage | unknown, hints ~0.1-0.2 | none (tooling) | **measure first** (proposal 1) |

**L1b global depth presort.** The per-tile sort (`sort_radix_tile_algo.h`) is a stable LSD radix on
`depth_bits` with ties kept in emit order, so a global stable presort by (depth_bits, gid) would be
md5-exact in principle. The cost kills it: emit would have to read each visible gaussian's record
(>= 13 words, ~52 B, `gather_vis_scatter.cpp`) in depth order. That is ~1.66M random reads per view;
hw-ceilings p5 puts random 64 B reads at ~0.54 G/s chip-wide, so >= 3 ms, or >= 1 ms of DRAM traffic
plus barriers if the payloads ride 3-4 global radix passes instead. This sits on the serial chain.
The saving it buys is bounded by the mover sort time, <= 1.1 ms traced before #418, and #418 has
already taken ~0.45 ms of that. Net <= 0. Not additive with #418 or the queued big-tile sort cut.

**L2 device bin_layout/publish.** The spec gated this on emit->mat bridge idle >= 0.15 ms. #412
section (b) measured mean gap 0.009 ms and all-core idle 0.008 ms/view (worst view 0.029). Reject.
Caveat: that trace is 11x10 worker. At 12x10, t410 shows host mat 0.083 -> 0.124 and publish_host
0.315 -> 0.341, still hidden by xvpin unless the bridge opens up; proposal 1 checks it.

**pfwc / project on 120 cores.** pfwc is TRISC-bound (~1.72-1.78 ms TRISC per core traced, t232;
writers ~1.25). ETH moved project only -0.051 against an ideal ~-0.095. The t226 candidates left are
all under the gate, and the cov2d SFPU and read split are already default. No new pfwc lever.

**New stage after ETH.** From #409: blend scaled ideally (6.448 -> 5.895 vs ideal 5.911), but the frame
fell only 0.455 while blend fell 0.553, so ~0.1 ms appeared elsewhere (sort 1.060 -> 1.173, project
less than ideal). There is no ETH Tracy: the profiler-instrumented idle-ERISC `cq_prefetch` is 0x4668
bytes > the 0x4490 limit (t397). So this is the only open question with data missing.

## Shelved lever whose premise changed (coordinator decision, not proposed)

#169 chunk frustum cull was shelved at +0.012 ms/view pre-xvpin, because the host cull (+0.11) and
Morton-order bin_emit (+0.11) ate the device gain (gather_wait -0.22), and it was not md5-exact (tie
order, max 19 LSB, 72 dB). Under xvpin the host cull is hidden, and the device-side bound is ~0.5 ms
of today's pfwc (39% of chunks skipped). An md5 fix may exist: break equal-key ties by original gid
after the per-tile sort. Listed here only so the coordinator can decide whether to unshelve.

## Proposals

1. **ETH 12x10 Tracy enablement and scaling check** (device, p150). Make a Tensix-only profiler run
   work on the eth overlay (dispatch-core profiling off, or a profiler-only overlay with fewer
   dispatch zones), capture 3 x 10 views on the iteration-215 stack, run `ana407.py`. Open a device
   bin_layout/publish task if bridge idle >= 0.15 ms/view; open a scaling-loss task if any program's
   12x10 window misses (11x10 window x 110/120) by >= 0.15 ms; otherwise close.
2. **Stop check after the queued mat cuts** (CPU only). Once the perm and big-tile sort cuts land,
   rerun `ana407.py` (c) on their Tracy and the #419 model. If TRISC1 idle outside jobs is < 0.3 ms
   mean and the worst-core gap is < 0.15 ms above mean, mat is TRISC1-bound (band_batch ~1.19 ms
   busy) and the plan recommends stopping. Otherwise name the single remaining lever >= 0.15.

## Recommendation

The project is near its stop line. Iteration 214 is 1.28x the published GPU figure; #418 adds
~0.45 ms (iteration 215). After the queued perm/big-tile cuts and proposal 1, stop unless one of
them shows a lever >= 0.15 ms/view. Do not reopen L1b, L2, pfwc tweaks or anything on the shelved
list. The one exception worth a user/coordinator look is the #169 premise change above.
