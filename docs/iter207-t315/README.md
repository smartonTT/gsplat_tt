# t315: iter 207, MATCULL_TRISC_FILL on by default

`GSPLAT_TT_MATCULL_TRISC_FILL` (t306, `docs/matcull-trisc-t306/README.md`) is now on by
default: unset = on, `=0` = off (`render/host/matcull_trisc_fill.h`, checked by
`tests/unit/test_matcull_trisc_fill.cpp`). t306 measured -0.16 ms/view bit-identical; the
coordinator ruled ~1.4% clears the charter's ~1% bar. Keep rule for this task: mean gain
>= 0.1 ms over at least 3 alternating rounds, consistent in every round, md5 906e0435 30/30
in every run.

Driver: `docs/iter207-t315/run_all.sh` (3 rounds via `drive.sh`, then
`opt/ttw/screenshot.sh 207 HEAD`), each step under `ttp lock p100`.

## Untraced A/B

Pending (device host unreachable at the first attempt).
