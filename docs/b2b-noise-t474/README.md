# Task #474: bh-30 back-to-back pass noise

Question: on bh-30 (p150, the user's viewer box) the b2b passes of
`render/run.py --back-to-back` spread 9.26-10.07 ms/view, against 0.04 ms on
the p100. Why, and how to A/B with a 0.1 ms keep gate there.

## Runs (bh-30, viewer stopped only for the bench, best-iter-216 tree)

- Run a (`out-a/`): 5 runs x 10 measured passes: 3 plain, 1 `--b2b-drop`,
  1 pinned with `taskset -c 12-23`.
- Run b (`out-b/`): 6 runs x 20 measured passes: 3 plain, 3 at nice -10 plus
  autogroup nice -10.
- `GSPLAT_B2B_ALL_STAGES=1` prints every stage timer per pass (B2B_STAGES
  lines); `sampler.log` samples load, aiclk and top CPU users every 0.5 s.

## Cause

- Not the device or clocks: aiclk stays at 1350 MHz, the governor is
  `performance`, and the device-bound stages (project gather wait ~2.62 ms,
  blend ~5.2-5.3 ms) barely move between quiet and slow passes.
- The extra time is host time in the `sort` stage, outside its timed leaves
  (emit/publish/layout/mat): quiet passes have sort ~1.09-1.25 ms, slow ones
  1.40-2.0 ms. It comes in episodes: a run is either quiet (passes within
  ~0.05 ms) or has a stretch where every other pass is ~0.2 ms slow, plus an
  occasional single ~10.3 ms pass. One run can switch between states.
- Warm-up: the check pass is always slow (10.3-10.6, emit 0.84 vs 0.43 ms),
  and the first pass after it is often slow too (9.47-10.32 in 6 of 10 runs).
  With the old default (3 measured passes) that pass weighs on the median.
- The host is shared: the container has 24 cores and the user's
  `mutagen-agent` runs constantly at ~50% CPU with bursts to 2-5 cores while it
  rescans /localdev/smarton. Bursts do not line up one to one with the slow
  passes, so it is host jitter in general (scheduling, cache), not one process.
- Priority does not fix it: nice -10 + autogroup -10 gave 2 quiet runs and 1
  run with the every-other-pass pattern, same as normal priority. Pinning to
  cores 12-23 made it worse (24 pool threads on 12 cores).

## Fix (render/run.py)

- `--b2b-warmup` (default 1): untimed passes after the check pass. They are
  still rendered and compared to the check pass.
- `--b2b-passes` default 3 -> 20 (each pass is ~0.28 s, so +5 s per run).
- Output line gains `warmup=N`.

Individual passes on bh-30 still jitter by 0.2 ms or more; the run median
does not.

## Spread before / after (10 unpinned runs on bh-30)

Recomputed from the same pass data:

| procedure | run results | range | stdev |
|---|---|---|---|
| old: median of 3 measured passes | 9.289-9.740 | 0.451 ms | 0.132 ms |
| new: 1 warm-up, median of the next passes (10 or 19) | 9.286-9.350 | 0.064 ms | 0.022 ms |

## A/B procedure for the 0.1 ms keep gate

1. One build per tree. Run `render/run.py --back-to-back` with its defaults
   (1 check + 1 warm-up + 20 measured passes), one run per tree per round.
2. Alternate A B B A A B (3 runs per tree) in one session on one board.
3. Per tree, ms_view_b2b = median of its 3 run medians (`ms_frame_median`).
   With a run-median stdev of ~0.022 ms the A-B difference has a stdev of
   about 0.025 ms, so the 0.1 ms gate is about 4 sigma.
4. No nice and no CPU pinning. Stop the viewer only for the bench.
