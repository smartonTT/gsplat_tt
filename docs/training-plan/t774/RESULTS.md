# T2 heavy-row split: reference, core plan and plane balance (#774, Mac, no device)

L1 of `docs/training-plan/t773/PLAN.md`. Code: `gsplat/t2_split.py`; tests:
`tests/unit/test_t2_split.py`; hero plan: `hero_plan.json` (per-core planes included).

## Design

- **Light path.** Rows with k <= 16 slots keep the rank planes of their own 64-row group. The row
  layout and the param/m/v pages are unchanged. A heavy lane reads as count 0, so a group runs
  max(light kmax, 1) <= 16 planes.
- **Heavy path.** Rows with k > 16 form a flat list, sorted by k (descending, then row id) and
  packed 64 per heavy group. Each lane walks its own contiguous slot run P[r] .. P[r]+k-1, so a
  heavy group runs its first lane's k planes. These sums replace the heavy lanes' +0.0 light sums
  before the projection backward.
- **Bit-equal.** Every row still sums its slots from +0.0, j ascending, one fp32 add per slot. The
  zero padding adds +0.0, which is exact because the accumulator is never -0.0. So the result is
  bit-equal to `t2_param_backward.segment_reduce` (the current T2 reference), and so is everything
  downstream (grads, Adam, pfwc).
- **Core plan.** Heavy groups go first, largest first, onto the least-loaded core (LPT). Light
  groups then get one contiguous range per core (cores in order), using the smallest capacity
  that fits them all (binary search). A core's light work is then just a (first, last) group range
  in its runtime args, and its DRAM reads stay contiguous. Cost unit: planes. An optional fixed
  cost per active group (`group_overhead`) can be added once device time per group is measured.
- **Reuse across views (msg #384).** Nothing is kept: each view's counts produce a new plan, and
  all sums are recomputed.

## Hero view (counts from the t750 hero T2 dump, 1,809,729 rows, 4,478,225 slots)

| | today (t2g, blocks round-robin) | split, kcap 16 |
|---|---:|---:|
| rank planes, total | 963,766 | 201,472 (light 181,694 + heavy 19,778) |
| busiest core (110 cores) | 30,772 | **1,835** |
| mean core | 8,761.5 | 1,831.6 (mean/busiest 0.998, min 1,815) |

Heavy rows: 2,975 in 47 heavy groups. Many are full-screen Gaussians (k = 1014, every tile),
so several heavy groups run 1014 planes each. Each such group sits alone on one core, and light
groups fill the rest. Per-core planes, percentiles 0/10/50/90/100: 1815 / 1828 / 1832 / 1835 / 1835.

Sensitivity to kcap (same counts):

| kcap | heavy rows | heavy groups | light planes | heavy planes | total | busiest core |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 10,193 | 160 | 156,050 | 21,034 | 177,084 | 1,613 |
| 16 | 2,975 | 47 | 181,694 | 19,778 | 201,472 | 1,835 |
| 32 | 1,705 | 27 | 192,440 | 19,334 | 211,774 | 1,930 |
| 64 | 1,388 | 22 | 199,584 | 19,115 | 218,699 | 1,993 |

The planes are modelled (E), not device time. At #773's 2.9 us/plane estimate, busiest-core work
drops from 30,772 to 1,835 planes, about 16.8x. The heavy path reads long contiguous runs, so its
per-plane cost may differ from the gather's. A device run should measure both.

## Tests (Mac, 2026-10-14)

`python3 -m pytest -q tests/unit/test_t2_split.py`: 11 passed, 1 skipped (16 s).

The skipped test is the full hero dump. It runs when `T2_HERO_REF_IN` points to a ref_in.npz
(494 MB). It was run here against the t750 hero dump: 1 passed (68 s), and
`python3 -m gsplat.t2_split model <ref_in.npz> --verify` printed `bit_equal: true`.

What the tests cover:

- Bit-equal sums on synthetic heavy-tail counts (k up to 1014; wide magnitudes, cancellation and
  ±0.0 slots), over kcap 0/1/4/16 and 1/3/7/110/200 cores.
- Bit-equal sums on the hero's own counts with random slots: every group that holds a heavy row,
  plus every 16th other group.
- A negative check: summing heavy rows in a different order is caught as not bit-equal.
- The T0 oracle (#674) gate (`t2_param_backward.check`) passes on `t0_sh0_64x64_n50.npz` with
  the split reduce at kcap 1, so most rows take the heavy path, and its sums are bit-equal.
- Plan invariants: every group and heavy group runs exactly once, ranges are contiguous, the
  per-core totals add up, and ordering is deterministic.
- Hero balance: the busiest core is within 16 planes of the lower bound, and mean/busiest is at
  least 0.98.
