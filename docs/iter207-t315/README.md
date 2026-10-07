# t315: iter 207, MATCULL_TRISC_FILL on by default

`GSPLAT_TT_MATCULL_TRISC_FILL` (t306, `docs/matcull-trisc-t306/README.md`) is now on by
default: unset = on, `=0` = off (`render/host/matcull_trisc_fill.h`, checked by
`tests/unit/test_matcull_trisc_fill.cpp`). t306 measured -0.16 ms/view bit-identical; the
coordinator ruled ~1.4% clears the charter's ~1% bar. Keep rule for this task: mean gain
>= 0.1 ms over at least 3 alternating rounds, consistent in every round, md5 906e0435 30/30
in every run.

Driver: `docs/iter207-t315/drive.sh 207 HEAD`, one `ttp lock p100` around sync + build,
tt-build stamp, 3 alternating rounds (base,off / off,base / base,off) and the device hero
shot (`opt/ttw/screenshot.sh`, same tree). Box: yyzo-bh-04 (p100a); both arms run in the
same session because absolute ms/view are box-specific (bh-04 iter-208 baseline 11.093).

## Untraced A/B

Pending.
