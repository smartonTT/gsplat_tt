# Task #78 — blend SFPU mix: const-load hoist / cheaper exp (gate: NO-GO)

**Board: yyzo-bh-07 (Blackhole p100a), not a p150.** Bicycle, 30 views, 1024x1024.
Base: 3fd2f1c (smarton/tt-project-opt, iter-172). Measured at 5b9e3ef = base + the
ablation knob below (default path unchanged: 30 views md5-identical, hero_vs_ref 100 dB).

## Static mix (pair body, 130 TRISC1 insns, from run 226 objdump)

| part | insns | of which const SFPLOADI |
|---|---:|---:|
| inputs: SFPLOAD x, y (4) and mx, my, A, B, C, op from DEST slot 6 (6) | 10 | 0 |
| conic (4 add for dx/dy, 8 mul + 4 mad) | 16 | 0 |
| min(power, 0) (swap + nop, per chain) | 4 | 0 |
| exp_21f, 2 chains (24 each) | 48 | 18 |
| alpha: op mul, min(., 0.99), pixel floor (load, mad, setcc, mov, encc) | 20 | 4 |
| T/R/G/B update incl. DEST loads/stores | 31 | 0 |
| ret | 1 | 0 |

Removable without changing output: the 22 const SFPLOADIs (3 constants could live in the
programmable const regs L12-L14, the rest as 1-insn SFPLOADs from DEST slot 6), and
exp's upper clamp to 255 (loadi + swap + nop per chain; unreachable because the input
is min(power, 0) <= 0, so z <= 127). About 24 of 130 insns per pair, 12 of 72 per single.

A cheaper exp (folding log2e into the conic, a 1-term polynomial, Schraudolph-style bit
tricks) changes the output bits, so it cannot meet the md5-identical acceptance. It would
need a new golden and a visual check.

## Ablation knob (timing only)

`GSPLAT_TT_BLEND_ABL` (host env -> compute define `BLEND_ABL`, default 0 = compiled out):
- `1` skips the SFPU blend bodies. The per-record loop, coefficient staging, mask walk,
  T readbacks and the reader all still run. The output is wrong (8 dB).
- `2` pads each pair body with 24 SFPNOPs and each single with 12. The output is unchanged
  (30 views md5-identical to a0). This is the cost of the issue slots a hoist would free.

## Measured (3 interleaved rounds, 30 views each; Tracy 10 views)

| | a0 (default) | a1 (no SFPU bodies) | a2 (+24 nop/pair) |
|---|---:|---:|---:|
| avg ms/view (mean of 3) | 48.91 | 42.82 | 49.80 |
| blend stage ms | 16.91 | 10.83 | 17.79 |
| Tracy tile_blend_sfpu makespan ms/view | 12.39 | 6.06 | - |
| Tracy tile_blend_load (NCRISC) makespan | 11.82 | 5.76 | - |

Rounds: a0 48.97/48.89/48.86, a1 42.84/42.89/42.75, a2 49.76/49.85/49.80.

## Read

- All SFPU blend math (every pair and single body) costs 6.1 ms/view end to end
  (6.3 ms of tile_blend_sfpu makespan). The other ~6 ms of that makespan is per-record work
  on TRISC1 (UNORM decode, 9-scalar DEST staging at ~8 RISC insns per scalar, a 16-step
  mask-pair branch walk, loop overhead), with the NCRISC reader close behind at 5.8 ms.
- Const-load hoist + dead clamp: 24 issue slots per pair cost 0.89 ms/view when added, so
  removing them saves at most ~0.9 ms/view.
- Cheaper exp: the exp's 30 non-const insns per pair (23% of the body) are worth at most
  ~1.5 ms/view even if exp were free, and any cheaper form changes the output.
- Upper bound for this task (hoist + a free exp): ~2.4 ms/view, under the 3 ms gate.
  **Decision: not implemented.** Only the ablation knob is landed; it is compiled out by
  default.

## Next lever (follow-up)

The per-record TRISC1 floor (~6 ms makespan at a1) is now as large as all of the SFPU math.
Candidates:
- Walk only the set pairs of the mask (computed jump over nonzero pairs) instead of 16
  compare/branch steps.
- Have the producer pre-build the SFPLOADI instruction words (or fp32 op/colour), so staging
  is lw+sw per half instead of zext/srli+lui+add+sw.

First check whether the NCRISC reader (5.8 ms at a1) sets that floor.

## Files

`remote_*.sh` (device scripts; tree `/localdev/smarton/gstt2-t78`), `verify.log`, `ab.log`,
`zones-a0.txt`, `zones-a1.txt`, `insn_mix.py` (copy of the t68 classifier).
