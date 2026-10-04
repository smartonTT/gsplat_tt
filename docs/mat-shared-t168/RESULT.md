# t168: shared big-tile sort in one-launch materialize — result (not landed)

yyzo-bh-07 p100a, bicycle, 30 views, 3 interleaved untraced rounds (view_total ms/view, STAGES line).

| round | on (shared) | off (GSPLAT_TT_OL_MAT_SHARED=0 = tip) | on - off |
|---|---|---|---|
| r1 | 18.491 | 18.510 | -0.019 |
| r2 | 18.494 | 18.531 | -0.037 |
| r3 | 18.576 | 18.633 | -0.057 |
| mean | 18.520 | 18.558 | **-0.038** |

Gate 0.3 ms: missed. md5: all 6 runs identical to md5-r82new.txt (opt/md5_compare.sh PASS, 30 views).

Tracy (MATCULL_PROF=1, per-view makespan over 30 views):

| zone | off | on |
|---|---|---|
| sort_subchunk_mat window | 3.104 | 3.024 |
| mat_ol_sort (busiest core) | 1.722 | 1.561 |
| mat_ol_keys | 0.332 | 0.070 |
| mat_ol_gather | 0.398 | 0.153 |
| mat_ol_wait (new) | - | 0.439 |

Why it is small: the big-tile re-sorts are gone (keys 634 -> 211 items), but the gather items now
wait on the single sort item (up to 0.44 ms/view on one core), so the window only drops ~0.08 ms
traced and ~0.04 ms untraced. The window is set by the serial chain sort -> gathers of the biggest
tile, not by the duplicated sort work. Further gain needs the big-tile sort itself split
(option 2: split the key sort across the core's two movers and merge) so the chain gets shorter.
