# Task #80 — blend per-record TRISC1 floor: attribution and cut

**Board: yyzo-bh-07 (Blackhole p100a), not a p150.** Bicycle, 30 views, 1024x1024.
Base: 4780373 (= 966c173 code, smarton/tt-project-opt). Result: **48.90 -> 44.39 ms/view
(-4.51, -9.2%), 22.5 FPS**. All 30 views md5-identical to base, hero_vs_ref 100 dB.

## Step 1: what sets the ~6 ms floor

New ablation levels of `GSPLAT_TT_BLEND_ABL` (timing only, output wrong, compiled out by
default): `3` = `1` + skip the UNORM decode and coefficient staging; `4` = skip the whole
per-record loop; `5` = `1` but keep the 16-step mask walk (a nop per taken branch).
One round on the base code (blend stage ms, 30 views):

| level | what still runs | blend ms | step |
|---|---|---:|---:|
| a0 | everything | 16.97 | |
| a5 | walk + decode + staging + loop, no SFPU bodies | 13.44 | bodies 3.5 |
| a1 | decode + staging + loop | 10.84 | **walk 2.6** |
| a3 | loop, mask read, T readbacks | 6.61 | **decode + staging 4.2** |
| a4 | no record loop (reader, CB handshakes, tile emit) | 5.61 | loop 1.0 |

So the floor was TRISC1, not the NCRISC reader: 7.8 ms of per-record TRISC1 work
(walk 2.6, decode + staging 4.2, loop 1.0) over a 5.6 ms reader/handshake floor.
The objdump showed why: `unorm16_to_f` was an out-of-line call with branches and stack
spills (4 per record), and each staged scalar took 9 RISC insns (2 runtime SFPLOADI words
+ SFPSTORE).

## Step 2: changes (all bit-identical, each behind a host env knob, default on)

1. `dm_fp32::unorm16_to_f` branchless and always-inline (normalize with clz, round the low
   8 bits nearest-even, let a round-up carry into the exponent). Exhaustive host test
   `tests/unit/test_trisc_fp32.cpp` passes (all 65536 inputs; run on yyzo-bh-07, the Mac
   SDK cannot link). Also used by the cull kernel.
2. Jump-table mask walk (`GSPLAT_TT_BLEND_JUMP_WALK`): ctz finds the lowest set pair, the
   2-bit pair value indexes a table of out-of-line bodies (they read their coefficients
   from DEST slot 6, so they take no arguments). Only set pairs are visited, same ascending
   order. The table is filled at runtime: a const initializer needs dynamic relocations,
   which the kernel loader rejects ("unexpected dynamic relocations").
3. UNORM16 op/colour decoded on the SFPU (`GSPLAT_TT_BLEND_SFPU_UNORM`): one SFPLOADI
   (USHORT) + int->fp32 cast + one fp32 multiply by fl(1/65535), staged once per subchunk.
   This is the reference expression fl((float)q * fl(1/65535)), so it is exact when SFPMAD
   rounds nearest-even. `=2` is a check mode: stage the RISC decode as before and poison
   microblock 0's R if the SFPU decode differs in any bit; its 30 views are md5-identical
   to base, so no staged value differed.

## Measured (3 interleaved rounds, final code 0626a2c, 30 views each)

| | base 4780373 | new | new, RISC decode (knob 0) |
|---|---:|---:|---:|
| avg ms/view | 48.859 / 48.842 / 48.994 | 44.321 / 44.457 / 44.389 | 45.729 / 45.666 / 45.673 |
| mean | **48.90** | **44.39** | 45.69 |
| blend stage ms (mean) | 16.91 | 12.36 | 13.72 |
| Tracy tile_blend_sfpu makespan ms/view (10 views) | 12.39 (t78 a0) | 7.88 | - |
| Tracy tile_blend_load (NCRISC) makespan | 11.82 (t78 a0) | 7.42 | - |

Split by change (blend ms, from the rounds above and round 2): branchless decode -1.4,
jump walk -1.7, SFPU decode -1.4.

Floor after the change (ablations on the new tree, one round): a1 8.37, a3 6.61, a4 5.62.
Remaining per-record TRISC1 work: staging 1.8 ms (5 raw fp32 scalars as 2 SFPLOADI words
each, 4 USHORT loads), loop 1.0 ms. Both are under the 3 ms gate.

## Next lever

The a4 floor: with no per-record TRISC work at all the blend stage still takes 5.6 ms
(reader NoC reads of the slab, CB handshakes, T readbacks, tile emit). It is now 45% of
the blend stage. Profile tile_blend_load / rd_l1_bulk at a4 before picking a fix.

## Files

`remote_*.sh` (device scripts, copied to `/localdev/smarton/t80_scripts`; trees
`/localdev/smarton/gstt2-t80` = base + ablations, `gstt2-t80b` = change),
`ab-base.log`, `ba.log`, `zones-new.txt`.
