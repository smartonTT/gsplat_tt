# t606: RECFAST + VIS_FAST bundle device A/B (structural plan 4, spec 1)

bh-33 p100, 2026-10-13 02:30-02:44 PDT, one build at ab9b4077 (bf425d81 + this driver), separate
TT_METAL_CACHE per arm off /home. Arms: A = defaults, B = `GSPLAT_TT_PFWC_RECFAST=1
GSPLAT_TT_PFWC_VIS_FAST=1`, C = `GSPLAT_TT_PFWC_VIS_FAST=1` (attribution, r5 only).
Driver `drive.sh` -> `remote_ab.sh`, outputs in `out/` (run logs gzipped).
`render/run.py --back-to-back`, 30 bicycle views, 20 timed passes after 1 warm-up, ms/view:

| round | order | A | B | C | B gain |
|---|---|---|---|---|---|
| r0 (warm-up round) | A B | 8.442 | 8.370 | | 0.072 |
| r1 | A B | 8.434 | 8.368 | | 0.066 |
| r2 | B A | 8.435 | 8.366 | | 0.069 |
| r3 | A B | 8.447 | 8.370 | | 0.077 |
| r4 | B A | 8.445 | 8.376 | | 0.069 |
| r5 (attribution) | C B A | 8.444 | 8.367 | 8.422 | 0.077 (C: 0.022) |
| **mean r1-r4** | | **8.440** | **8.370** | | **0.070 (0.83 %)** |

B wins every round, but the bundle gives the same 0.07 ms/view as RECFAST alone (t599: 0.071):
VIS_FAST alone is worth 0.022 and adds nothing measurable on top of RECFAST.

md5: A 077caaba = golden 30/30. B c19db0d9, **off golden**: views 27 and 29 differ
(`out/md5-r0-dump-*.txt`); b2b raw_md5 B and C both b0ca00be vs A 1d372b94, so VIS_FAST is the
cause (RECFAST alone was bit-exact in t599). Hero (view 0) is bit-exact: B device hero
(`opt/metal-screenshots/ttw-t606-bundle/`) 42.85 dB vs benchmarks/reference_v2/hero.png, golden
match, max 0 LSB; hero and diff x10 checked by eye: no tile seams, blocks or stripes, diff only on
spoke/frame/leaf edges as in best-iter-227.

Verdict: **shelved** on both kill rules (gain 0.070 < 0.1 ms/view gate; md5 off golden). The VIS_FAST
predicate was not made exact: an exact predicate can only keep or lose VIS_FAST's 0.0-0.02 ms, so the
bundle cannot reach the gate either way. Both flags stay default 0; no golden refresh proposed.
