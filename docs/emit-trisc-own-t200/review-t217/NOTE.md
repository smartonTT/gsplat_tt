# t217: TOWN fallback on device at tiles_x=30 (960 px)

Tip of smarton/tt-project-opt, yyzo-bh-07 p100a, bicycle views 0-3 at 960 px (tiles_x=30, so town=false on device).
Runs: on1, off1, on2, off2 (GSPLAT_TT_OL_EMIT_TOWN=1/0).

MD5 (identical in all 4 runs):
- view00_hero cf788a2c5d4ad2cb4fcadd9ed0f07fd6
- view01_orb_00 a4f4b34f5233354d4ad5cbf39acdb239
- view02_orb_01 edad52c1e3855cba2c9df943335ad1e9
- view03_orb_02 ca43f28171db96bf34b9fbd8d33625f9

ms/view, orb_00/01/02: on1 14.1/14.4/14.8, on2 14.1/14.5/14.9, off1 14.0/14.3/14.7, off2 14.2/14.5/15.0.
Time difference from the unused TRISC launch and +46 KB L1: none measurable (within ~0.1 ms run noise).

Result: PASS. No hang. All runs exit with a host segfault after rendering, in run.py's golden mse
compare (960 px vs 1024 px golden). Same in both modes; not a device issue.
