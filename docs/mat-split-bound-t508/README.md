# t508: mat fill jobs + big-tile split, per-view bound (not built)

Task #508, 2026-10-11 (PDT). Goal: build the t335 lever (big-tile split over both movers,
512-record fill jobs, rank-ordered big gather) behind an env flag, and A/B it in b2b mode.
Gate: >= 0.15 ms/view b2b.

## Verdict: split alone not built; the combined lever is the only form that can clear the gate

Before building, I bounded each piece per view on t506's MATCULL capture (p150 bh-30,
best-iter-223, chunks M0/M10/M20, 33 views, `bound508.py`, `out.txt`). t506 and t335 used
the per-core mean over views. This bound uses the max over cores in each view, which is
the makespan the frame actually waits for.

| bound (per view, mat makespan) | model ms | gain | frame ms/view x0.28 / x0.40 | clears 0.15? |
|---|---|---|---|---|
| base | 2.112 | - | - | - |
| ideal split (big cores end at their BRISC end) | 2.054 | 0.059 | 0.016 / 0.023 | no |
| realistic split (2 parts, repeated O(N) work) | 2.094 | 0.019 | 0.005 / 0.007 | no |
| fill jobs alone (every mover done-wait removed, big items as today) | 1.788 | 0.324 | 0.091 / 0.130 | no |
| combined (no done-waits and big items free) | 1.460 | 0.652 | 0.182 / 0.261 | yes, as a ceiling |

What this shows:

- **The split alone is worth at most ~0.02 ms/view.** Per view the big cores (1-2..6-2,
  ~9.7 per view) do lead, but other cores end only 0.05-0.12 ms behind them: 14-2, and the
  x-3 cores whose NCRISC carries ~14k records plus ~0.54 ms of done-waits
  (`detail-M10.txt`). Freeing the big items moves the makespan to those cores.
  t506's 0.12-0.13 split-alone figure came from per-core means. That averaging hides the
  fact that the second-slowest core in each view is close behind.
- A split part also repeats most of the item's work. Each part reads all N keys (~0.26 ms)
  and runs the 4-level select over N plus a stream of the whole tile in the gather, so
  about 70% of an item's time does not shrink with the part size.
- **Fill jobs alone are bounded at 0.09-0.13 ms/view**, also under the gate, and this
  bound assumes every done-wait disappears.
- Only both together reach 0.18-0.26 ms/view, and only as a ceiling. Measured levers have
  so far realized well under their ceilings (t335 assumed 15% overhead; the split above
  loses ~70% of its ideal).

## Why the combined build is a larger job than the spec assumed

1. **Fusion protocol.** `OL_MAT_SELECT` already splits big items into parts, but it turns
   off mat+blend fusion (`sort_device.cpp:877`, `!ol_mat_select()`). The blend reader polls
   one ready flag per (tile, subchunk) (`reader_alpha_blend_mb_devcull.cpp` ~408). Parts
   need per-part flag planes: more host flag pages, a part count per subchunk in the
   worklist, and a reader that polls every part's flag before it starts the subchunk.
2. **Big-item work per part.** The cut has to land on the key reads and the select
   (a shared pre-pass per tile, or ranks handed from one part to the next), not only on
   the gather. Otherwise the repeated O(N) work eats the gain, as the realistic row above
   shows.
3. **Fill jobs.** Streamed 512-record jobs on the permute path, plus a rank-ordered big
   gather so jobs can start before the whole big tile is gathered. The t335 model gives
   ~0 for finer jobs without that gather.

Those are three coupled kernel changes: materialize mover, blend reader, and host
worklist/flags. Each needs device iterations for md5 parity, which does not fit this
task's remaining budget. No device arms were run and nothing changed in code. Default
behaviour is unchanged.

## Cheaper lever seen on the way

The x-3 cores' NCRISC holds ~14k records and waits ~0.54 ms on done-waits, while their
BRISC finishes early. BRISC buffers are capped at `kMatMover0Cap` = 6144 records, and
`build_mat_worklist` pins items above that to NCRISC. Rebalancing that cap, or splitting
those mid-size items between movers (no fusion change needed, since they are not big
items), would shorten the second-slowest cores. On its own that gain is capped by the
big cores (~0.06 model ms per view), so it pays only together with the combined lever.

## Files

- `bound508.py`: per-view bounds from t506's MATCULL CSVs (imports `../b2b-timeline-p150-223/ana506.py`).
  Run: `python3 bound508.py <out2>/dev-M0.csv.gz <out2>/dev-M10.csv.gz <out2>/dev-M20.csv.gz`,
  where `out2` is `tt-project/state/runs/1393/wt/docs/b2b-timeline-p150-223/out2` (not committed, 57 MB).
- `out.txt`: the bounds above. `detail-M10.txt`: top 6 cores per view, chunk M10.
