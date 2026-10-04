# Task #170: pair stage diet (K2 read-ahead, batched writes, count fold)

Board: yyzo-bh-07 (Blackhole p100a). Workload: bicycle, 30 views, 1024x1024.
Base: smarton/tt-project-opt at 2034568 (task #174 tip, PRECULL=2 default).

## What changed

- **K2 diet** (`pfwc_fuse::emit_pairs_diet`, used by
  `render/kernels/dataflow/tile_assign_scatter_seg.cpp`). The K2 reads the
  lofs/box pages ahead through a ring buffer, writes pair pages in batches with
  no flush per page, and drops the divide it did for each pair.
  Kill switch: `GSPLAT_TT_K2_DIET=0`.
- **Count fold.**
  - The K2 counts pairs per tile in a stack array (RISC local memory, not L1).
    It then writes one count row per (core, mover), row `2*k+mover`, with
    `ceil(screen_tiles/16)` pages.
  - The one-launch sort (`sort_bin_onelaunch.cpp`, arg 27) takes these rows.
    It skips its own count pass and its first barrier, and reads no keep pages.
  - The K2 uses the same page ranges as the sort: the speed-proportional
    ranges from #174 via `pfwc_fuse::k2_range_speed`, or `k2_range` when the
    speed split is off. The K2 publishes its ranges with the rows.
  - The sort folds only when cores, row_pages, tiles, P and every core's range
    match. Otherwise it logs `[SORT] ONELAUNCH k2_fold=0 why=<code>` and runs
    its own count pass.
  - Kill switch: `GSPLAT_TT_K2_FOLD=0`.

## Gate: paired A/B, untraced

All arms of r5 to r7 are md5-identical to `md5-r82new.txt` (46a725ab).
r1 to r4 ran on the pre-rebase tip (PRECULL=1, ref ee35138a) and are kept for
the record only.

view_total, ms/view:

| round (run order)        | off (no diet) | nf (diet only) | base (diet+fold) | base - off |
|--------------------------|--------------:|---------------:|-----------------:|-----------:|
| r6 (base, nf, off)       | 18.238        | 17.519         | 17.326           | -0.912     |
| r7 (off, nf, base, u, unf) | 18.303      | 17.614         | 17.438           | -0.865     |
| mean                     | 18.270        | 17.566         | 17.382           | **-0.889** |

- The diet alone saves -0.704 ms/view, and the fold another -0.184.
- r7 also ran both arms with the speed split off (uniform ranges):
  - `u` (uniform, with fold): 18.023
  - `unf` (uniform, no fold): 18.211
- The fold still pays with uniform ranges. The speed split is worth about
  0.6 ms with the fold on.

Stage means over r6 and r7, ms/view:

| arm  | project | sort  | blend | gather_wait | bin_emit |
|------|--------:|------:|------:|------------:|---------:|
| off  | 4.614   | 4.519 | 8.880 | 4.438       | 4.076    |
| nf   | 3.939   | 4.530 | 8.883 | 3.767       | 4.082    |
| base | 4.114   | 4.072 | 8.882 | 3.950       | 3.669    |

## Tracy (30 views, yyzo-bh-07)

Program busy time, ms/view (`out/tracy2-*-gaps.txt`):

| arm  | pfwc  | K2    | gap   | sort  | gap   | mat   | blend | idle  | traced end |
|------|------:|------:|------:|------:|------:|------:|------:|------:|-----------:|
| off  | 2.981 | 1.436 | 0.528 | 4.012 | 0.306 | 3.007 | 5.887 | 0.847 | 18.171     |
| nf   | 2.981 | 0.814 | 0.435 | 3.963 | 0.314 | 3.008 | 5.857 | 0.762 | 17.384     |
| base | 2.981 | 0.985 | 0.422 | 3.547 | 0.305 | 3.008 | 5.868 | 0.740 | 17.129     |

Zone makespans, ms/view (`out/tracy2-*-zones.txt`):

| arm  | k2_pairs | k2_rows | sort_ol_count | sort_ol_barrier | sort_ol_prefix | sort_ol_emit |
|------|---------:|--------:|--------------:|----------------:|---------------:|-------------:|
| off  | (K2 1.436) | -     | 0.794         | 0.360           | 0.037          | 3.071        |
| nf   | 0.764    | -       | 0.794         | 0.362           | 0.037          | 3.067        |
| base | 0.922    | 0.006   | -             | 0.407           | 0.373          | 2.931        |

K2 + count + barrier, ms/view:

| arm  | K2 + count + barrier | with prefix |
|------|---------------------:|------------:|
| off  | 2.590                | 2.627       |
| nf   | 1.920                | 1.957       |
| base | 1.335                | 1.708       |

## Verdict

KEEP. The change gives -0.889 ms/view (-4.9%) and is md5-identical. Both
kill switches restore the old paths.

## What is left

- **Counting costs the K2 about 0.16 ms** (k2_pairs 0.764 -> 0.922). This is
  the `cnt[t]++` done for each pair. The same cost shows with uniform ranges,
  so it is not range imbalance. A difference array updated once per
  rectangle row, plus one prefix pass over the screen tiles, would remove
  most of it.
- **sort_ol_prefix grows from 0.037 to 0.373 ms. This is now explained.**
  - Per-core prefix time under the fold is 180 us median and 372 us max (nf:
    28 and 37 us). Launch skew is only 0.2 us. Source:
    `t170-on`/`t170-nf` `dev30.csv`, script `out/pfx.py`.
  - Under the fold, NCRISC fills the sort's 1024-page gid/tid window as soon
    as the kernel starts. So do the BRISCs that own no prefix page, in
    barrier 2. That is about 26 MB of 64 B reads in total, and each core's
    220 row-page reads for the prefix wait behind them.
  - The window is still worth having. Probe of `GSPLAT_TT_OL_WIN_PAGES`, all
    arms md5-identical (view_total, ms/view):

    | round | base | 1536 | 1280 | 512 | 256 | 128 | 64 |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | r8 | 17.282 | | | 17.437 | 17.780 | 17.923 | 18.218 |
    | r9 | 17.360 | 17.287 | 17.344 | | | | |

  - Smaller windows lose 0.16 to 0.94 ms. Larger ones gain at most 0.07,
    which is noise. So the time until the emit starts is set by the window
    fill, not by the prefix. A real fix needs fewer or larger reads in that
    fill. Measure the fill first, with a zone around it.

## Files

- `drive.sh`, `remote_time.sh`, `remote_tracy.sh`: drivers for sync, A/B with
  the md5 gate, and Tracy.
- `out/run-rN-<arm>.log`, `out/md5-rN-<arm>.txt`: the timing runs.
- `out/tracy2-<arm>-{zones,gaps}.txt`: Tracy captures on the rebased tip.
- `out/tracy-*`: Tracy captures before the rebase.
- `out/run-r8-*`, `out/run-r9-*`: the window-size probe.
