# Task #68 — blend TRISC1 instruction mix; per-gaussian coefficients staged in DEST

**Board: yyzo-bh-07 (Blackhole p100a), not a p150.** Bicycle, 30 views.

## Static mix (objdump of the JIT TRISC1 ELF, before)

`alpha_blend_compute_mb` TRISC1 from `ttmc-gstt2-t65` (tip 27348a1 kernel), classified
by `insn_mix.py` over the 48 unrolled dispatch bodies (16 pair + 32 single):

| body | RISC insns | runtime-scalar build | Tensix insns pushed | runtime SFPLOADI | const SFPLOADI |
|---|---:|---:|---:|---:|---:|
| pair (2 microblocks) | 209 | 88 (42%) | 140 | 22 | 22 |
| single (1 microblock) | 144-147 | 80 (54-56%) | 82 | 20 | 11 |

Each runtime fp32 scalar reaches the SFPU as two SFPLOADIs (lo/hi 16 bits) whose
instruction words TRISC1 builds in GPRs: `zext.h`/`srli` + `lui` + `add` + `sw` per
half = 8 RISC insns per scalar. The dispatch re-built all ~10 per-gaussian scalars
(A, B, C, mx, my, op, r, g, b, pixel floor) for every microblock dispatch. The single
body issues 144+ RISC insns for 82 Tensix insns, so TRISC1 instruction issue, not
the SFPU, bounds it.

## Change

`BLEND_COEF_DEST` (host env `GSPLAT_TT_BLEND_COEF_DEST`, default 1): per live record
the 9 scalars are built once and SFPSTOREd as lane-broadcast vectors into DEST slot 6
(free: 0-3 R/G/B/T, 4-5 ramps); the pixel floor once per subchunk call. The dispatch
bodies read them with SFPLOAD (1 native insn, immediate address). fp32 DEST
round-trips exactly (same path as the accumulators), so output should be
byte-identical. `=0` compiles the old form, for an in-session A/B from one tree.

## Scripts

`run_all.sh <rev>` (detached, repo root): `opt/sync_remote.sh` -> `remote_verify.sh`
(30-view md5, k0 vs k1) -> `remote_ab.sh` (3 interleaved rounds) -> `remote_tracy.sh`
k0/k1 -> objdump of the new TRISC1. Remote tree `/localdev/smarton/gstt2-t68`.
