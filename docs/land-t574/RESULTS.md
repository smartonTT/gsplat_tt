# t574: BRISC stack-margin fix (muldiv_lo32) on the default path

Box: bh-33 (p150, measurement reservation), 2026-10-12 20:41-20:52 PDT, under `ttp lock p100`.
Arms: base = best-iter-226 (90268457), new = 2c895ad2 (a645f27f fix + drive). Fresh kernel caches per arm.

## Stack margins (opt/brisc_stack_check.py --scan, default config, B4A fold)
| kernel | base | new |
|---|---|---|
| matblend_brisc (pfwc diet_start) | 112 B | 368 B |
| tile_assign_scatter_seg | 176 B | 432 B |
| others (cq_*, sort_bin_onelaunch, writer_pfwc_split) | >= 3632 B | >= 3728 B |

No `__udivdi3` in any new brisc.elf. Full lists: out/stack-base.jsonl, out/stack-new.jsonl.

## Correctness
- 30/30 views md5 identical between arms (out/md5-r0-dump-*.txt), sweep md5 906e0435.
- Hero on device: opt/metal-screenshots/ttw-t574/hero.png, diff hero_diff10.png,
  PSNR 42.51 dB vs benchmarks/reference_v2/hero.png, golden match (max 0 LSB).
  Looked at both images: no tile seams or block artifacts.

## Back-to-back (render/run.py --back-to-back, 30 views, 20 passes), median ms/view
| round | base | new |
|---|---|---|
| warm (r0) | 8.595 | 8.597 |
| r1 | 8.595 | 8.616 |
| r2 | 8.596 | 8.619 |
| r3 | 8.589 | 8.621 |
| median r1-r3 | 8.595 | 8.619 |

New is +0.024 ms/view (+0.28%) in all three alternating rounds: small but consistent.
Below the charter's ~1% threshold; landed as a safety fix (matblend margin 112 -> 368 B).
Not a new best (no >= 0.1 ms/view gain), so no best-iter tag.
