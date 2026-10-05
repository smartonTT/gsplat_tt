# t302: profiler kcfg that fits the fused mat+blend program

Problem: under the device profiler, `kcfg_extra_kb` (render/host/kcfg_size.h) added 8 KB on
top of the split pfwc's +32 KB, so +40 KB. The default fused mat+blend program (iter 206)
then overflowed L1 in its static CBs by 5888 B (tracy-t293-fuse-capture.log). The old
workaround was `GSPLAT_TT_PFWC_WRITER_SPLIT=0`, which profiles a different pfwc.

Fix: with the split pfwc, the profiler size is +32 KB (no extra bump). Window from the t293
logs: the traced split pfwc with the BRISC reader is 97536 B and fits from +27 KB; the fused
program fits up to +34 KB. Without the split the old `+8` stays. Untraced sizes are
unchanged (tests/unit/test_kcfg_size.cpp checks both).

## Evidence (yyzo-bh-07 p100a)

- Tracy at full defaults (`tracy.sh t302-def`, no WRITER_SPLIT=0, no KCFG_EXTRA_KB), commit
  184b829: rc 0, 4 programs per view on 110 cores: pfwc 2.005, K2 0.985, sort_ol 1.441,
  fused mat+blend 6.346 ms (traced). mat->blend wait 0.000, blend_end_max 6.252 ms.
  Files: docs/matblend-ready-t273/t289/out/tracy-t302-def-*.
- Untraced paired round (drive2.sh, base 51ac7cf vs head, alternating builds): RESULTS_TBD

Note: drive.sh's untraced step used `--timeout 900`, which devrun now refuses (600 s
ceiling); drive2.sh uses 400 s (one arm takes ~12 s).
