# B4 re-bound (task #539): k2 + sort of view N+1 inside mat+blend of view N

CPU-only analysis, 2026-10-12. No device runs. Inputs:

- the t528 device-profiler chunks of best-iter-224 (bh-33 p100a, 110 cores, 1350 MHz),
  `docs/profile-t224/out/dev-P*.csv.gz` on `ttp/t528-profile-t224`;
- the steady b2b split of best-iter-224: **9.131 ms/view** = pfwc 1.654 + k2 0.647 + sort 1.459 +
  mat+blend 5.330 + gaps 0.040 (measured);
- earlier results: t200/t202 (emit cost per record), t230/t231 (blend thread roles), t512/t517
  (pfwc overlap), t508 (mat split), #529 (sort coherence), T-A #533 and T-B #534.

Model: `model.py` (this folder). It reads each core's program windows and zone busy times for
24 steady views (the last 5 of each chunk; the first kept view of each chunk is dropped because
it holds a 25 ms host stall). Output: `out/model_out.txt` (per view) and `out/model_summary.txt`.
Profiled periods run about 2% above the unprofiled b2b numbers (median 9.33 vs 9.131 ms).

Model numbers are **estimates, not measurements**.

## Answer

| case | upper bound | realistic | bit-exact |
|---|---|---|---|
| Bench b2b, B4 as written in PLAN (movers only) | 2.11 ms/view | **net loss** (-0.6 to -3.3 ms) | yes |
| Bench b2b, B4' (BRISC + TRISC2 take k2 + sort N+1) | 2.11 ms/view | **0.6 to 1.2 ms/view** | yes |
| Bench b2b, B4a (k2 only, staged first step) | 0.65 ms/view | 0.4 to 0.55 ms/view | yes |
| Live viewer, small motion: overlap | same as bench | same as bench | yes |
| Live viewer, small motion: extra from frame-to-frame reuse | 0.85 ms | 0.2 to 0.4 ms (mat depth-order fix-up only) | yes, if the fix-up runs to a full order |

**Verdict: GO for B4', in stages, gated on a step-0 feasibility check (L1 and kernel size).**
The realistic bench gain clears the 0.1 ms gate by a wide margin. B4 as PLAN wrote it (the
movers do all the work) is NO-GO: it does not fit.

Ideal end state: period = pfwc + mat+blend ≈ 1.65 + 5.33 ≈ 7.0 ms/view (143 FPS). The realistic
end state is 7.9-8.5 ms/view (118-127 FPS) on today's code.

## Why the PLAN version does not fit

The schedule is `pfwc(N+1) → [mat+blend(N) ∥ k2+sort(N+1)]`. pfwc cannot run beside blend
because both need the TRISC SFPU, and k2(N+1) needs pfwc(N+1)'s output. The period becomes
pfwc + max(mat+blend, hidden work) + 2 barriers.

During mat+blend, TRISC1 is about 97% busy (mat 1.42 + blend 3.67 of 5.22 ms). All k2 + sort
work (N+1) therefore has to run on RISCs that are free during blend. Per core, per view (model
medians):

| work of k2 + sort (N+1) | RISC-ms per core |
|---|---|
| mover zones in k2 (k2_pairs, k2_rows) | 0.99 |
| mover zones in sort (fill, fillb, prefix, barrier, emit) | 2.74 |
| TRISC zones in k2 (k2_trisc, scalar) | 0.83 |
| TRISC zones in sort (sort_ol_town, the t202 tile-owned emit) | 3.43 |
| **total at today's rates** | **7.99** |

What is free inside mat+blend(N), per core:

| RISC | free time | source |
|---|---|---|
| BRISC | 3.91 ms (after its mat body ends at 1.38 ms) | sort_subchunk_mat end → program end |
| TRISC2 (pack) | 3.81 ms (after mat_cull_mask ends at 1.44 ms, minus 0.05 for syncs) | t230/t231: in decode-ahead blend it only joins the T readbacks (every 512 records) and the tile-end packs |
| NCRISC (blend loader) | 3.72 ms zone, of which only 0.06 ms reading | mostly waits for CB space, but it feeds blend, so time-slicing it is risky |

With the movers only, the work does not fit. The TRISC parts re-costed on a mover (×1.2,
216 vs 179 cycles per record, t200/t202) need 8.8 mover-ms per core. The t196 mover-only emit
needs 6.8. Free mover time is 3.9 ms with BRISC alone, or 5.7 ms if half of the NCRISC loader
wait can be used. Model result:

- BRISC only: **-0.9 to -3.3 ms/view** (a loss).
- With 50% of NCRISC: -1.5 to +0.6 ms.
- It breaks even only with about 80% of the NCRISC wait, which would starve blend.

That matches the PLAN's warning: its 0.3-0.6 assumed B1 would shrink the emit, and B1 is shelved
(#534).

## B4': use TRISC2 and BRISC

TRISC2 is the piece the PLAN missed. It is a scalar core that already runs the TRISC emit code
(sort_ol_town), and it is idle for about 3.8 ms of every blend.

- **Work assignment:** BRISC takes the mover parts, TRISC2 the TRISC parts. Emit ranges can move
  between them, because both emit paths exist and are byte-identical (t202).
- **Fit:** work 7.99 RISC-ms against 7.72 free (median).
- **Model, per core, worst core sets the period:**
  - no work-claiming: saves 1.45 ms/view (1.16-1.76 over 24 views);
  - with work-claiming across cores: 1.90 (1.72-2.08);
  - adding half of the NCRISC wait reaches the 2.11 cap.

Deductions from the 1.45 model figure, to get the realistic 0.6-1.2:

- The roles are not perfectly poolable. TRISC parts 4.26 vs TRISC2 free 3.81 leaves 0.45 to
  shift to BRISC, which has only 0.2 spare. About 0.2-0.3 ms overflows.
- Two or three device-wide barriers inside the hidden work (k2 counts → prefix → emit), each
  waiting for the slowest core's blend progress: 0.03-0.1 ms.
- Blend slowdown from L1 port contention (TRISC2 and BRISC scalar record traffic beside TRISC0/1
  ring traffic) and readback latency when TRISC2 is mid-batch: 0.05-0.2 ms. TRISC2 must check
  for a pending readback between emit batches of about 4 records (about 0.5 µs; about 60 syncs
  per core per view).
- The sort's L1 working set has to shrink (see risks). Smaller staging means more NoC round trips
  for the emit: 0-0.3 ms.

### Bit-exact

Yes. k2 and sort run the same code on the same data. Records land at prefix-computed offsets, so
which RISC emits a record and when does not change the bytes; the town and mover emit split is
already byte-identical (t202). mat+blend(N) reads sort(N)'s buffers, untouched by N+1, because
the DRAM bucket and pfwc buffers are double-buffered (DRAM is ample).

### Risks, in step-0 order

1. **L1.** Today the sort uses about 2 × 567 KB per core (T-B). mat+blend is already 21 KB over
   the static CB budget, with four blend CBs aliasing mat CBs. After mat ends on a core, an
   estimated 500-800 KB is free with a manual layout (non-aliased BRISC mat CBs plus unused parts
   of aliased regions), not 1.1 MB. The sort staging must be cut to fit, or the CBs allocated in
   phases inside one persistent program. This is the deciding question. k2 alone (B4a) needs far
   less.
2. **Kernel size.** BRISC and TRISC2 must carry the k2 and sort code next to mat+blend. The
   kernel config is already tight: MATCULL_PROF overflows (108384 > 103424 B), and the fused
   program fits only up to +34 KB of profiler config (t302).
3. **TRISC2 readback duty.** It must poll between emit batches. A missed readback stalls
   TRISC1; a deadlock is possible if the emit waits on a ring that the blend drains.
4. **mat mover critical path.** fz_mv_big reaches 1.91 ms on some cores. Hidden work on BRISC
   must start only after BRISC's mat body ends (as modelled) and must not take NoC slots mat
   needs.

## Live viewer

The viewer is pipelined like the bench (memory, 2026-10-10), so B4' gains the same 0.6-1.2
ms/frame on device time. The cost is about one mat+blend (about 5 ms) of extra pose-to-photon
latency, because N+1's pose has to be fixed before mat+blend(N) starts. Today's pipelined viewer
already fixes it at pfwc(N+1).

Reuse between frames adds little (#529):

- k2 and sort must rerun every frame (binning changes, 0% identical tiles even at small motion),
  and after B4' they are hidden anyway.
- The only reuse lever left is mat's in-tile depth sort (about 0.85 mover-ms, on the critical
  path). At 0.06-0.5° per frame, 77-97% of pairs keep their tile. An insertion fix-up from the
  last order breaks even at about 8 inversions per pair, so it gains 0.2-0.4 ms at slow rotation
  and nothing on the bench (12.6 inversions per pair).
- It stays bit-exact only if the fix-up runs to a complete order with the same tie-break.

## Other structural levers the data suggests (ranked)

Bounds are model or earlier-task estimates, not measurements.

| # | lever | bound ms/view | realistic | bit-exact | notes |
|---|---|---|---|---|---|
| 1 | **B4a: k2(N+1) only, inside blend(N) on BRISC + TRISC2** | 0.65 | 0.4-0.55 | yes | Step 1 of B4'. The work (1.8 RISC-ms) fits 2× over in the free time, k2's L1 need is small, and it proves the persistent-program plumbing (double buffers, a barrier inside blend, TRISC2 readback polling). |
| 2 | **B4' full: + sort(N+1)** | 2.11 | 0.6-1.2 total (with #1) | yes | Gated on the L1 step-0. |
| 3 | Land the #536 opacity prune | 0.133 measured | 0.133 | md5 906e0435 (new golden) | Measured; only the tag and landing are left. |
| 4 | Work-claiming pfwc on the blend tail (t512) | 0.32 | 0.1-0.25 | yes | After B4', pfwc is 23% of the period and the 0.5 ms mean end-of-blend idle per core is the main slack left. The static deal failed (t517); dynamic claiming was never tried. |
| 5 | mat fill jobs + big split (t508) | 0.18-0.26 | 0.1-0.2 | yes | On the critical path in both today's and B4' schedules. |
| 6 | B2 exact ellipse-tile test (T-A) | 0.41 | 0.1-0.2 | yes | Pair cut helps mat+blend (critical) more once k2 + sort are hidden: about 1.5 of the 2.23 ms/M pairs cost stays on the critical path. |
| 7 | Viewer-only mat depth-order fix-up | 0.85 | 0.2-0.4 (viewer, slow motion) | yes, if run to full order | Nothing on the bench. |
| 8 | Use TRISC0/TRISC2 for more blend issue work | unknown | unknown | must be | Blend is TRISC1 issue-bound (t230). The FPU is about 0% busy and FPU blend forms measured as losses (t111, t511). Needs a probe of how busy TRISC0 is under decode-ahead before any bound. |

**Recommended order:** #3 (free), then #1 (low risk, about 0.5 ms), then the step-0 L1 study
for #2, then #4 and #5 on the new critical path (pfwc + mat+blend).
