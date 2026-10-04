# Task #183: blend tail and claim order (no device run)

Question (from `docs/rerank-17ms.md` section 4A): how much of the blend's
5.87 ms/view is end-of-program imbalance, and would a better claim order or
32x16 splits of the biggest tiles remove it?

Data: the t170 default-chain capture `t170-on` (30 views, yyzo-bh-07 p100a,
2034568 + #170), remote path
`/localdev/smarton/gstt2-t170/opt/profiler/t170-on/dev30.csv`. It is not in
git. For a cross-check on exact per-tile costs, I also used the t147 dprint
`docs/fuse-matblend-t147/out/t147-tc.dprint` (older PRECULL=1 tip, 3 launches).

## Step 1: the tail is real (gate passed)

Per view, using blend TRISC end times per core (`out/step1.txt`, `out/step1b.txt`):

| measure (30 views) | mean | median | min | max |
|---|---:|---:|---:|---:|
| core end, max - mean | **0.439 ms** | 0.417 | 0.259 | 0.734 |
| core end, max - p90 | 0.260 | 0.239 | 0.094 | 0.534 |
| latest core: end - its last claim | 1.157 | 1.034 | 0.643 | 2.475 |
| latest core: last 2 subchunks | 0.494 | 0.456 | 0.267 | 0.841 |
| latest core end - global last claim | 0.800 | 0.779 | 0.619 | 1.024 |

The mean core end is 5.358 ms, matching the spec. The tail is 0.44 ms/view,
which is above the 0.3 ms gate.

## Why the tail exists: early claim

In `render/kernels/dataflow/reader_alpha_blend_mb_devcull.cpp`, the claim loop
(around line 255) takes the next tile from the shared counter as soon as it has
pushed the previous subchunk. Only after that does it wait for a ring slot
(`cb_reserve_back(CB_BUCKET_BULK, ...)`, around line 387). The bulk ring has 2
slots, so each core has 3 tiles committed at once: one in compute, one in the
ring, and one claimed but waiting for a slot.

In the capture, claims 0 to about 330 (the biggest tiles) are all taken in the
first ~150 us, about 3 per core. The host LPT balance is lost in this race, and
some cores end up with three large tiles (for example 1.7 + 1.7 + 1.3 ms). The
last tile a late core claimed is already committed ~1.2 ms before that core
finishes.

## Step 2: replay

Model (`sim.py`): an event replay of the 2-slot ring.
- The claim costs 3.5 us (the measured gap is 3 to 4 us) and the subchunk read
  20 us.
- `early` claims as the kernel does today. `late` reserves the ring slot first,
  then claims.
- Per-subchunk costs come from the capture (`run_b.py`): subchunk j finishes when
  slot j+2 is reserved. A core's last two subchunks share one measured span,
  which is split 50/50.

How good the model is:
- Replaying each core's own measured sequence reproduces core ends within 30 us.
- The dynamic `early` replay is pessimistic: max_end is 5.988 ms against 5.796
  measured, about 0.21 ms per view. The start race lands differently in every
  run, and the measured run sits near the model's good end.
- I therefore quote savings both inside the model and against the measurement.

Grid, 30 views, 6 jitter draws (start +-10 us, cost +-3%), from `out/grid_b.txt`:

| variant | max_end ms | tail ms | vs model early | vs measured |
|---|---:|---:|---:|---:|
| measured | 5.796 | 0.439 | | |
| early (today) | 5.988 | 0.661 | 0 | |
| **late** | **5.523** | 0.195 | **-0.465** | **-0.274** |
| early + oracle cost order | 5.879 | 0.552 | -0.109 | |
| late + oracle cost order | 5.362 | 0.034 | -0.626 | -0.435 |
| split top-16, early, today's order | 5.922 | 0.583 | -0.066 | |
| split top-32, early, oracle order | 5.596 | 0.243 | -0.393 | |
| split top-32, late, today's order | 5.535 | 0.182 | -0.453 | |
| split top-32 (h=0.6), late | 5.645 | 0.185 | -0.343 | |

Sensitivity:
- Late-claim tail: 0.19 to 0.28 ms with last_split 0.3 to 0.7, and ~0.196 ms
  with reads of 5 to 40 us.
- With group_gap 0, early 0.77 and late 0.20; the late number is stable.

Cross-check on exact per-tile costs (t147 dprint, `t147_order.py`,
`out/t147_order.txt`), max_end in ms:

| launch | early, LPT (today) | late, LPT | late, record count desc | late, live fit desc | late, oracle |
|---|---:|---:|---:|---:|---:|
| 0 | 6.115 | 5.933 | 5.909 | 5.909 | 5.858 |
| 1 | 6.052 | 6.052 | 5.882 | 5.921 | 5.858 |
| 2 | 4.778 | 4.527 | 4.491 | 4.472 | 4.451 |

- Late claim plus plain record-count-descending order saves 0.17 to 0.29 ms
  against today in all three launches.
- The live-count order (what 4A proposed: mat writes live counts) does no better
  than the record count the host already has.

## Answers

- **(a) Live-cost claim order:**
  - On its own, with today's early claim, it saves only ~0.1 ms.
  - With late claim, ordering by record count (no live counts needed) gets most
    of the oracle gain.
- **(b) Splitting the top-k tiles into 32x16 halves:**
  - Helps only while claims stay early.
  - Under late claim it gains nothing: -0.453 vs -0.465 for late alone, and
    worse with any per-half overhead (h=0.6).
  - It also needs mat, writer and output changes. **Not worth building.**
- **(c) Both:** no better than late claim plus a cost-descending order.
- **Best lever: late claim + record-count-descending claim order.**
  - Modeled saving: 0.27 ms against the measurement (conservative) to 0.47 ms
    inside the model. The exact-cost cross-check gives 0.17 to 0.29 ms.
  - Central estimate ~0.3 ms/view traced.
  - Blend is device-bound and on the critical path of every view, so the
    untraced saving should be about the same. This is not measured.

## Build spec (md5-identical to md5-r82new.txt)

1. **Late claim.** In `reader_alpha_blend_mb_devcull.cpp`, at the top of the
   claim loop and before `noc_fast_atomic_increment`, add
   `cb_reserve_back(CB_BUCKET_BULK, BULK_REC_SLOT);`.
   - `cb_reserve_back` only waits for space and does not move the write
     pointer, so the existing reserve inside the subchunk loop returns at once
     for the first subchunk.
   - Gate it with a compile-time define from `GSPLAT_TT_BLEND_LATE_CLAIM`
     (default 1; 0 restores today's behavior).
   - Empty tiles (`L_sub == 0`) are unaffected.
2. **Claim order = record count descending.**
   - In the host code that builds the per-core blend lists (`build_lpt`,
     `render/host/sort_device.cpp:1309`, consumed through `lpt_meta`), add a
     mode that deals tiles round-robin in record-count-descending order: core c
     gets ranks c, c+n, c+2n, and so on.
   - The kernel's rank interleave then claims exactly in global descending
     order, with no kernel change.
   - Same env switch, or a separate `GSPLAT_TT_BLEND_CLAIM_DESC`, so the two
     parts can be A/B tested separately.
3. **Why md5 stays identical:** each tile's output depends only on its own
   records. Since #60, which core blends which tile already changes from run to
   run, and the md5 has stayed stable.
4. **Gate:**
   - Paired untraced A/B on yyzo-bh-07 p100a, 3 rounds x 30 views, arms
     off / late / late+desc. All arms must be md5-identical to md5-r82new.txt.
   - Keep if >= 0.3 ms/view. If late alone lands at 0.2 to 0.3, keep both parts
     together only if the pair passes.
   - One Tracy capture to confirm that the blend max - mean core end drops from
     0.44 to ~0.2 ms.

## Reproduce

```
scp <p150 host>:/localdev/smarton/gstt2-t170/opt/profiler/t170-on/dev30.csv .
python3 parse.py dev30.csv blend.pkl
python3 step1.py blend.pkl; python3 step1b.py blend.pkl
python3 run_b.py blend.pkl --jitter 6 --split_k 0,8,16,32
python3 t147_order.py ../fuse-matblend-t147/out/t147-tc.dprint
```
