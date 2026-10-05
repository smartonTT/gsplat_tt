# t174: PRECULL=2 + speed-proportional emit page ranges (GSPLAT_TT_OL_MOVER_SPEED)

Board: yyzo-bh-07 (Blackhole p100a). Bicycle, 30 views, 1024x1024, untraced, view_total ms/view.
Code 8295e08 (pre-rebase), driver `drive_gate.sh`, log `gate.log`.

## Root cause (t166-p2f Tracy, EMIT_PROF=1, PRECULL=2 even split)
Records per mover are equal (~12k/view). Under PRECULL=2 the one-launch emit is set by a few slow movers:
BRISC (NOC0) on rows y=2,3 stall issuing blendrec reads (ep_brec 1.0-1.37 ms vs ~0.46 elsewhere,
emit 3.0-3.54 ms) and NCRISC (NOC1) on columns x=14,15 (emit 2.5-2.8, brec 0.6-0.83).
Emit window 3.576 ms (PRECULL=1: 2.795). The fast loop itself is not slower per record; the
t166 fixed row split moved pages onto the already slow x=14,15 NCRISCs.

## Fix
`render/host/sort_mover_speed.h`: per physical core BRISC/NCRISC relative speed (records/cycle,
mean 1000) from t166-p2f. The host gives each mover a page range in proportion to its speed
(`speed_bounds`, contiguous and in order, so output is unchanged). Default on only with PRECULL=2;
the row split (`GSPLAT_TT_OL_SPLIT_ROWS`) is now default only with PRECULL=2 and is ignored when
mover speed is on. PRECULL now defaults to 2. Kill switches: GSPLAT_TT_PRECULL=1, GSPLAT_TT_OL_MOVER_SPEED=0.

## Gate (tip = PRECULL=1, base = PRECULL=2 + mover speed, rows = PRECULL=2 + t166 row split)
| round | tip | base | rows | base-tip |
|---|---|---|---|---|
| r1 | 18.595 | 18.202 | 18.288 | -0.393 |
| r2 | 18.428 | 18.174 | 18.313 | -0.254 |
| r3 | 18.453 | 18.157 | 18.378 | -0.296 |
| mean | 18.492 | 18.178 | 18.326 | **-0.314** |

Stages (mean): sort tip 4.350 / base 4.497 / rows 4.62; blend 9.278 / 8.883; project 4.656 / 4.590.

Correctness: tip md5 identical to md5-r82new.txt (all 30 views, 3 rounds). Base md5 set
46a725ab... in all rounds, equal to the rows arm and to the t166 PRECULL=2 image. Base vs tip:
every differing pixel is 1 LSB, PSNR 74.24-77.81 dB (hero 75.64 dB).

## Tracy of the new default (t174-p2w, EMIT_PROF=1)
Emit window 3.011 ms (p2f 3.576). Still uneven: BRISC y=2 max 2.91 ms, NCRISC x=14/15 up to 2.84;
the measured costs predict 2.49 ms if re-balanced (`out/emit_cores-t174-p2w.txt`). Table refresh
from this capture is a follow-up.
Tracy: yyzo-bh-07:/localdev/smarton/gstt2-t166/opt/profiler/t174-p2w/render.tracy

## Goldens
Hero golden refreshed from t162r1-base hero_clean.png (old one in
tests/fixtures/hero/archive/hero_golden_8bit_pre-t174-precull2.png). 30-view md5 set:
out/md5-t174r1-base.txt (old: out/md5-r82new.pre-t174.txt).
