# Task #146: blend TRISC1 instruction diet (blend lever 2)

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 30 views, 1024x1024, untraced.
md5 checked per view against `md5-r82new.txt`.

## What changed (`render/kernels/compute/alpha_blend_compute_mb.cpp`)

| knob (0 = old form) | change |
|---|---|
| `GSPLAT_TT_BLEND_CONST_HOIST` (h), default 1 | the exp and clamp constants of the pair bodies are staged once per subchunk (L12-L14 and DEST slot 6 indices 11-12) instead of ~22 SFPLOADIs per pair body; the dead 255 clamp is dropped |
| `GSPLAT_TT_BLEND_RAW_STAGE` (r), default 0 | coefficient staging into DEST slot 6 issues raw SFPLOADI words kept in registers; 1/65535 is loaded once per record (~17 fewer RISC instructions per record) |
| `GSPLAT_TT_BLEND_FAST_TRED` (t), default 1 | the T-saturation max reduce uses a range compare with a per-row-pair early exit (same period, 512) |

Unit test: `tests/unit/test_blend_t_live.cpp` (320000 checks, 0 mismatches vs the old reduce).

Judgment calls:
- (b) uses the "lite" raw staging. Full pre-encoded words from materialize would need 13+ words
  per record, doubling the 32 B record and halving slab capacity.
- (c) keeps the period at 512. A shorter period (256) was tried as an arm: 22.83 ms/view but
  **not bit-identical** (30/30 views differ, hero 76.3 dB), so it was dropped.

## Interleaved A/B (on the lever A base, 3 rounds, `out/t146-ab-rounds.txt`)

Blend stage ms (host timer), mean of 3 rounds; every arm md5-identical over 30 views:

| arm | r1 | r2 | r3 | mean | vs off |
|---|---|---|---|---|---|
| off | 11.275 | 11.249 | 11.275 | 11.266 | |
| h | 10.638 | 10.653 | 10.639 | 10.643 | -0.62 |
| r | 11.099 | 11.061 | 11.035 | 11.065 | -0.20 |
| t | 10.313 | 10.314 | 10.319 | 10.315 | -0.95 |
| all | 9.444 | 9.431 | 9.448 | 9.441 | -1.83 |

ms/view: off 24.60, all 22.69 (-1.91).

Tracy (BLEND_PROF, 10 views, `out/t146-{off,all}-*.txt`): blend program 7.69 -> 5.76 ms/view,
`tile_blend_sfpu` makespan 8.03 -> 5.89 ms.

## Tip confirmation (rebased onto lever B, b7a04a4)

3 interleaved rounds (`out/t146-tip-rounds.txt`), all arms md5-identical over 30 views:

| arm | ms/view per round | mean ms/view | blend ms |
|---|---|---|---|
| off (h=0 r=0 t=0) | 21.34 / 21.20 / 21.16 | 21.23 | 11.27 |
| ht (h=1 r=0 t=1, the new default) | 19.72 / 19.64 / 19.60 | 19.65 | 9.68 |
| all (h=1 r=1 t=1) | 19.43 / 19.34 / 19.42 | 19.39 | 9.45 |

Decision: keep h and t (default on): 21.23 -> 19.65 ms/view (-1.58, -7.4%), 50.9 FPS.
r adds only -0.23 ms blend (-0.26 ms/view; paired 0.29 / 0.30 / 0.18) on top of h+t, under the
0.3 ms bar, so it stays default off. The code is kept behind the knob: it is bit-identical and
may count for more once materialize is fused into blend (lever 1) and staging is on the
critical path again.
