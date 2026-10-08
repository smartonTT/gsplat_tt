# Task #413: where TRISC1 idles inside mat (per job, per core)

#412 found TRISC1 idle about 1.08 ms mean / 1.64 ms max per core per view in the mat
phase, waiting on the BRISC/NCRISC movers. This task adds per-job Tracy zones to the
mat TRISCs and splits that idle into start, between-job and tail gaps.

## What changed

- `render/kernels/compute/mat_cull_compute.cpp`: `MJ_ZONE("mj_wait")` around the
  pick_job/mailbox wait and `MJ_ZONE("mj_job")` over the rest of each job (fill, band,
  pack/patch, done word), on all three TRISCs. The zones compile only with `MATCULL_PROF=1`
  and `PROFILE_KERNEL`.
- `render/host/blend_device.cpp`, `render/host/sort_device.cpp`: `GSPLAT_TT_MATCULL_PROF=1`
  sets `MATCULL_PROF=1` for the fused mat compute and the mover (cull) kernels. The movers
  already had per-subchunk `MAT_PZ` zones (mat_ol_rd/sort/perm/wr/keys/gather, mat_meta,
  mat_cull_wait) behind the same define.
- Default build: unchanged. The untraced run below is md5 906e0435 on 30/30 views.
- `docs/xvpin-tracy/ana407.py`: new section (c) (`job_gaps`), plus `--mhz` and `--job-csv`.
  Unit test: `tests/test_ana407_job_gaps.py`.

## Run (yyzo-bh-04, p100a, 110 cores, 1350 MHz; commit 0cc15151; defaults = xvpin + zero-copy)

- `docs/mat-job-zones-t413/drive.sh` (runs under `ttp lock p100`): ssh-preflight, sync, one
  untraced 30-view round, then three Tracy chunks (views 0:10, 10:20, 20:30) with
  `GSPLAT_TT_MATCULL_PROF=1 GSPLAT_TT_PROFILE_READ_EVERY=11` (`remote_tracy.sh`), then the
  hero screenshot.
- Untraced: avg_frame_ms 8.748, ALL_VIEWS_IDENTICAL 30/30 (md5 906e0435), XVIEW_HITS 29/30.
- Tracy: XVIEW_HITS 9/10 per chunk; 30 frames stitched, 50,504 rows per frame; mj_job
  count matches fz_u_nj on every core.
- Hero (device, p100a): psnr 42.51 dB vs `benchmarks/reference_v2/hero.png`, golden
  match, max_lsb 0. I looked at `out/hero.png` and `out/hero_diff10.png`: no tile seams or
  blocky areas; the diff is only fine edge detail (spokes, foliage).
- Analysis: stitch with `opt/profiler/stitch_device_csv.py` into `tmp/t413/stitched.csv`, then
  `python3 docs/xvpin-tracy/ana407.py tmp/t413/stitched.csv --out docs/mat-job-zones-t413/out
  --zone-csv .../ana-zones-per-view.csv --job-csv .../job-gaps-per-core.csv`; the report is
  `out/ana413.txt` (section (c)).

## Result: TRISC1 idle per core per view (ms; mean over views of mean / max over cores)

| | mean | max |
|---|---|---|
| mat length (T1 mat_cull_mask) | 2.299 | 2.839 |
| jobs | 10.02 | 13.43 |
| **start gap** (mat start -> 1st job) | **0.611** | 0.805 |
| **between-job gaps** (sum) | **0.469** | 0.904 |
| between-job gap, largest | 0.218 | 0.575 |
| between-job gap, mean | 0.057 | 0.156 |
| tail gap (last job -> mat end) | 0.005 | 0.017 |
| in-job idle (job - band - copy) | 0.012 | 0.013 |
| **total idle** | **1.096** | 1.661 |
| band_batch (busy) | 1.191 | 1.342 |

About 56% of the idle is the start gap, 43% is between jobs, and the tail and in-job idle
are negligible. Once a job reaches TRISC1, it runs without stalls.

What the movers do while TRISC1 waits (share of gap time, each mover counted separately):

- start: NCRISC mat_ol_sort 79%, BRISC mat_ol_sort 71%, BRISC perm 22%, NCRISC rd 16%.
  Both movers are sorting the first subchunk.
- between: NCRISC sort 52% / perm 42%, BRISC sort 35% / perm 34%. They are sorting and
  permuting the next subchunk.

So mat is mover-bound: TRISC1 waits on the movers' sort and perm.

## How much blend work could fill the gaps

"crit" is the change in the slowest core's mat+blend length (traced, 6.15 ms) if every
core filled that much of its gaps with its own blend work.

| gaps counted | fillable, mean core (ms) | crit (ms/view) |
|---|---|---|
| >= 5 us | 1.082 | -0.833 |
| >= 20 us | 1.050 | -0.803 |
| >= 50 us | 1.012 | -0.779 |
| >= 100 us | 0.924 | -0.721 |
| whole blend jobs (0.372 ms each) | 0.442 | -0.235 |

Caveats:

- The start gap (0.61 ms) cannot hold this view's blend: no mat output exists yet. Only
  work that does not depend on this view's mat could go there, such as the previous view's
  blend tail or the next view's pfwc. The realistic fill is then the between gaps only
  (0.47 ms mean, 0.90 ms max), and only for tiles whose mat is complete (needs ready flags).
- At whole-blend-job granularity only 0.44 ms fits (-0.235 ms crit). Blend would have to
  be split finer than one job for the bigger numbers.
- These are traced numbers. Tracy adds overhead (traced device period 10.6 ms vs 8.75 ms
  untraced), so the absolute gain would be smaller untraced.

## Levers this points to

1. Shorten the start gap: a smaller first subchunk, so the first job reaches the TRISCs
   sooner (0.61 ms mean idle at stake per core).
2. Speed up the movers' sort and perm, or move part of the sort off the movers; this cuts
   both the start and the between gaps.
3. Interleave blend into the between gaps at sub-job granularity, gated on per-tile mat
   ready flags (up to about 0.47 ms mean per core).

## Files

- `drive.sh`, `remote_tracy.sh`: capture drivers.
- `out/ana413.txt`: full analysis; `out/job-gaps-per-core.csv`: per view and core gaps;
  `out/ana-zones-per-view.csv`: zone windows per view.
- `out/dev-c*.csv.gz`, `out/tracy-u-c*.csv.gz`, `out/T*.log`, `out/tracy-c*.out`: raw captures.
- `out/round1-xvdef.out`, `out/run-r1-xvdef.log`, `out/md5-r1-xvdef.txt`: untraced round.
- `out/hero.png`, `out/hero_diff10.png`: device hero and 10x diff vs reference_v2.
