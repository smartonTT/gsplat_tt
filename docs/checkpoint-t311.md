# Stop-or-reframe checkpoint after lever B, the lever-A model and TRISC fill (task #311, no device)

No device run. Every number names its source. GPU numbers are published, not measured:
G1 10.75 ms (INRIA 3DGS, RTX A6000, bicycle 1080p), G2 5.44 ms (G1 pixel-normalized to 1024²).

## Where we stand

| item | ms/view untraced | source |
|---|---:|---|
| iter 206 (fused mat+blend, default) | 11.129 | best-iter-206, #293 (yyzo-bh-07 p100a) |
| #306 MATCULL_TRISC_FILL A/B (off / on) | 11.081 / 10.923 (−0.158) | `docs/matcull-trisc-t306/README.md` @ 3e1e475b |
| iter 207 (fill on by default, #315 pending) | ~10.92 | #306 |
| GPU G1 (published, not measured) | 10.75 | REPORT.html |

After iter 207 we are ~0.17 ms (1.6 %) behind G1.

## 1. What #306 actually bought, from its own host timers

`TTW_TIMING stage_blend` is the host wait on the fused mat+blend program
(`docs/matcull-trisc-t306/out/run-*.log`):

| round | off | on | on − off |
|---|---:|---:|---:|
| r1 | 6.923 | 6.708 | −0.215 |
| r2 | 6.962 | 6.746 | −0.216 |

So the device gain is a steady **−0.216 ms** in mat+blend. The frame number (−0.158) is lower only
because r1-on had a noisy d2h (0.289 vs ~0.21). The t304 model removed 0.68 ms of fill/patch from
each mover and predicted −0.56 (range 0.30-0.72). About 0.34 ms of the modeled gain is missing.

Two explanations fit the data, and nothing measured separates them (the fill path has no zones):

- **H1, TRISC0 is now the mat bound.** TRISC0 fills for both movers of a core
  (`pick_job` in `mat_cull_compute.cpp`, one TRISC pipeline per core). With the model's TRISC/mover
  cost ratio of 0.83 it needs ~1.13 ms, well under the movers' ~1.87 ms. It becomes the bound if
  the fill costs ≥ ~1.7× the modeled ratio per record (volatile L1 loads after a cache
  invalidate on TRISC0, 6 loads + 6 stores per record). Then mat ≈ TRISC0 busy ≈ 2.3 ms, matching
  the measured gain.
- **H2, the movers wait for structural reasons.** Whole-slab jobs (up to 8192 records), a shared
  job queue for two movers (head-of-line blocking), and the last item per mover not hidden. TRISC0
  then still has ~1.3 ms of idle in mat.

A third factor widens the range: #176 found untraced mat-side gains at ~0.5-1× the traced ones,
so the traced gain could be 0.22-0.43 ms, closer to the model.

## 2. Does t297's 2.415 ms TRISC idle still hold? No.

t297 measured 2.415 ms/core of TRISC idle in mat with fill **off**. With fill on (iter 207):

| case | TRISC0 busy in mat | TRISC idle left for a lane | mat window |
|---|---:|---:|---:|
| t297 (fill off) | ~0.1 | 2.415 | 2.513 |
| H2 (model ratio 0.83) | ~1.13 + cull | ~1.1-1.3 | ~2.3 |
| H1 (TRISC0-bound) | ~2.1-2.3 | ~0-0.2 | ~2.3 |

The lane needs TRISC0 too (blend unpack and decode-ahead), not only TRISC1, so the fill and the
lane compete for the same thread.

## 3. Lever A on top of iter 207

Starting point: #303 realistic small-first 0.543 ms traced (lane work 0.953 ms/core, no view
regresses), realistic today's order 0.267, pessimistic 0.192 (`docs/lane-model-t303.md` @ 97d88778).

- **H2:** the lane's 0.95 ms barely fits into ~1.1-1.3 ms of TRISC0 idle, the mat window is
  ~0.22 ms shorter, and every fill job is another point where the lane must yield and spill DST.
  Estimate ~0.3-0.4 ms traced, **0.15-0.4 ms untraced**.
- **H1:** every lane cycle on TRISC0 is taken from the fill, which is the mat bound, so the mat
  phase grows by about what the lane saves. **~0 ms.**
- Risk is unchanged and high: DST save/restore per yield (2-5 µs), three TRISC threads
  interleaving two programs, a 64 KB ring + ~41 KB carved out of a full CB4.

**Verdict: do not build A now.** Its expected value on iter 207 is under the 0.3 gate in H1 and
only at the gate in H2. It comes back only if the measurement in section 6 shows ≥ 1.0 ms of idle
on each TRISC during mat with fill on.

## 4. Small-first mat order (stage 0) on its own: not worth a device run

The #303 model puts small-first alone at **+0.011 ms (slightly worse)**. Reordering items inside a
mover slot does not change the slot's total work, so the mat makespan cannot shrink. It only
moves small tiles' ready flags earlier, which helps nobody without a lane, and it pushes the big
tiles (the ones the blend claims first, in descending order) later. With fill on, the order is
still irrelevant to the makespan. It only makes sense as stage 0 of A.

## 5. Remaining levers, ranked (untraced ms/view)

| rank | lever | expected | basis | verdict |
|---:|---|---:|---|---|
| 1 | Recover the fill shortfall: rebalance fill/patch across TRISC0/TRISC2/mover (H1) or finer jobs, per-stream queues (H2) | 0.1-0.34 | the 0.34 gap between t304's model (0.56) and #306's measured 0.22 | **gated on the measurement below** |
| 2 | A, blend lane during mat, on iter 207 | 0-0.4 | section 3 | shelve unless the measurement shows ≥ 1.0 ms idle per TRISC |
| 3 | Radix sort on TRISC1 (t304 candidate 2; `sort_record_ids` is 52 % of mover time, ~1.27 ms/mover) | unknown | needs a measured TRISC/mover cycle ratio; TRISC1 is idle in mat | model only if the measurement gives a ratio ≤ 1.0 and TRISC1 idle ≥ 2 ms |
| – | C, dynamic chunk claim | < 0.3 | tails pfwc 0.300 (WRITER_SPLIT=0), emit 0.084, town 0.124 (t297) | below the gate alone; recheck the pfwc tail at defaults in the same capture |
| – | Stage 0 small-first alone | +0.011 | #303 | do not run |
| – | B, fold K2 into sort_ol | +0.19 (slower) | #298 | shelved |
| – | Host serial path (d2h 0.21-0.29, publish_host 0.15-0.17, pfwc dispatch 0.07) | each < 0.3 | t297 host bridge, #171 | shelved |
| – | FPU quadratic-form blend | ≤ ~0.7 ceiling, not md5-safe | #111 (2.0 ms of a then 10.6 ms blend; blend is 3.66 ms now) | not reopened |

Nothing on this list approaches G2 (5.44 ms, published). The blend (3.66 ms traced) and pfwc
(~2.3 ms) dominate, and every blend lever measured so far is shelved.

## 6. The call: one gating measurement, then stop unless it shows room

The only lever with a plausible ≥ 0.3 ms is recovering the fill shortfall, possibly combined with
A. Both depend on one unknown: where TRISC0 and the movers spend the mat phase with fill on.
That is cheap to measure (one Tracy run at defaults, which fit since #309) and it closes 1.6 % to
G1, which is a compounding reason to look despite the ~1 % rule. So:

1. **Measure** (one device task, after #315 lands iter 207): Tracy zones on the fill path. Go/no-go
   table is in the follow-up spec.
2. If the measurement shows ≥ 0.3 ms of recoverable room, **build the matching fix** (rebalance
   or finer jobs), behind a knob, A/B 30 views.
3. Otherwise **stop**: write the conclusion (iter 207 ≈ 10.92 ms/view ≈ 91.6 FPS on p100a vs
   G1 10.75 published), refresh REPORT.html, and notify the user.

Lever A is not on the path unless step 1 explicitly reopens it.
