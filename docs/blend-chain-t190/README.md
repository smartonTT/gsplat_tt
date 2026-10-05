# Tail-chained blend mask walk (task #190, build only, no device)

**Result: built behind `GSPLAT_TT_BLEND_CHAIN_WALK` (default 0 = off). Both variants compile.
In objdump every body that has a successor ends in `jr` (a sibling call, no `jalr`), loads the
next table entry as its first load, 60-106 instructions before that `jr`, and takes 1 jump.
Every body path issues exactly the same SFPU ops as knob 0, so the image should stay
md5-identical. Not measured yet: a device A/B decides (gate >= 0.3 ms/view).**

## The knob
`render/host/blend_device.cpp`: `GSPLAT_TT_BLEND_CHAIN_WALK` -> define `BLEND_CHAIN_WALK`, next to
`BLEND_JUMP_WALK`. It works only on top of the jump walk (`BLEND_JUMP_WALK=1`, the default), and,
like it, only in the MATH thread (TRISC1). UNPACK and PACK code does not change.

| value | walk |
|---|---|
| 0 (default) | the task #80 jump walk, unchanged: a loop does ctz, table `lw`, `jalr` body, clear the pair, `bnez`. Same ELF as before. |
| 1 | the t189 spec: body J computes `rest = mask >> (2J+2)`. If `rest == 0` it branches to its own copy of the math and returns. Otherwise it loads the next body, runs its math and jumps (`jr`) to the next body. |
| 2 | branch-free: the next index is computed even when `rest == 0`. Then ctz is 32, and `32 & 30 = 0` lands on slot `4(J+1)+0`, which holds an end stub that only returns. |

Table: pair p with 2-bit value pm at index `4p + pm` (the jump walk's layout); slots `4p + 0`
hold the end stub. Index: `blend_chain_index(rest, base)` in
`render/kernels/compute/blend_chain_walk.h`: `c = ctz(rest)` (32 if 0), `b = c & 30`,
`index = 4*base + 2b + ((rest >> b) & 3)`. One change from the t189 sketch: a body does not
clear its pair and rescan the mask. It shifts the mask by its own (constant) position, so the
mask is passed on unchanged and the table pointer rides along as the 2nd argument. Pair 15 has
no successor and only returns.

## How it was checked (no device)
- `cc.sh <src> <out> 0 1 2` rebuilds the blend compute kernel's TRISC1 ELF on yyzo-bh-07 the way
  the tt-metal JIT does: same flags, defines, linker script and firmware symbols, taken from the
  t188 JIT cache, in a private directory (run under `ttp lock p100`; the device was not used).
  Knob 0 rebuilds the t188 default ELF exactly: same 4627 instructions at the same addresses as
  `docs/blend-dispatch-t189/out/trisc1-default.dis`. Objdump, symbols and sizes are copied to
  `out/cw<knob>/` (knob 0: sizes and symbols only; its objdump is the t189 one).
- `chain_bodies.py ../blend-dispatch-t189/out/trisc1-default.dis out/cw1/trisc1.dis
  out/cw2/trisc1.dis` -> `out/body_counts.txt`: per-body counts, plus checks that the table `lw`
  comes before the first SFPU op, that the chain path ends in `jr`, that no body has a `jalr`,
  and that each body path issues the same SFPU / Tensix ops (operands included, in order) as the
  same knob-0 body. All checks pass; a one-operand change in a copy of the objdump makes it fail.
  No body adjusts `sp`, so the chain does not grow the stack.
- Record loop (`_start`): 908 instructions for every knob. Its Tensix ops are the same 157 in
  the same blocks; only the tile-end block sits at another address. The rest are RISC
  differences, mainly 16 more table stores at init (the end-stub slots) and no spills around the
  walk.
- Host test `tests/unit/run_cpp.sh tests/unit/test_blend_chain_walk.cpp`: both variants visit
  the same table slots in the same order as the jump walk, for all 2^16 pair-occupancy patterns
  (3 fills each), every mask with up to 3 bits, 2M random masks and edge masks; end slots land at
  `4p`. 5,191,199 checks, 0 bad. A broken index (bit 0 of b cleared) gives 1,325,206 bad.

## Objdump evidence
Knob 2, body `<0,1>` (single microblock; `out/cw2/trisc1.dis` at 0x7f64):
```
srli   a4,a0,0x2      # rest = mask >> 2
ctz    a5,a4
andi   a5,a5,30
srl    a4,a4,a5
andi   a4,a4,3
sh1add a5,a5,a4
sh2add a5,a5,a1       # a1 = table
lw     a5,16(a5)      # next body, loaded before the 60 SFPU ops
...    60 SFPU ops
jr     a5             # sibling call
```
Knob 1, same body: `srli; beqz a4,.L7; ctz; andi; srl; andi; sh1add; sh2add; lw a5,16(a5);`
60 SFPU ops; `jr a5`; `.L7:` the same 60 SFPU ops; `ret`. Pair 14 bodies are shorter, because
`rest < 4` there and the compiler drops the pair search (knob 1: `srli; beqz; sh2add; lw; jr`).
End stub (`blend_chain_end`): `ret`.

Call site (the same for knobs 1 and 2): the first lookup (`ctz, andi, srl, andi, sh1add,
sh2add, lw a2`) is scheduled among the record loads at 0x7aec-0x7b0c, 83 instructions before
`mv a1,s2; jalr a2` at 0x7c54, followed by `j` back to the record loop. Knob 0 instead runs, per
live record, `sw s1; li s10,3; mv s1,s11; sw s2; mv s11,t3`, the loop at 0x7d58
(`ctz, andi, srl, andi, sh1add, sh2add, lw a4, jalr a4, sll, andn, bnez`) and
`lw s2; lw s1; mv t3,s11; j`.

## Counts
Per body (RISC instructions / taken jumps); SFPU ops are the same for all knobs (60 single,
109 pair):

| body | knob 0 | knob 1 | knob 2 |
|---|---|---|---|
| has a successor | 12 / 3 (11 walk + `ret`; `jalr`, `ret`, `bnez`) | 10 / 1 (`jr`) | 9 / 1 (`jr`) |
| last of the record | 12 / 2 (`bnez` falls through) | 3 / 2 (`beqz`, `ret`) | 9 + stub `ret` = 10 / 2 |

Per live record with n body calls:

| | knob 0 | knob 1 | knob 2 |
|---|---:|---:|---:|
| RISC instructions | 12n + 9 | 10n + 3 | 9n + 11 |
| taken jumps | 3n | n + 3 | n + 3 |
| table load used by the next instruction | n | 0 | 0 |
| at n = 2.18 | 35.2 / 6.5 / 2.2 | 24.8 / 5.2 / 0 | 30.6 / 5.2 / 0 |

n = 2.18 is t189's 0.78 calls per microblock times 2.8 microblocks per live record. When pair 15
is the last body, knobs 1 and 2 save one more jump and 2 (knob 1) or 9 (knob 2) instructions.
Note: t189 counted "1 taken jump instead of 3" per call. Per record the drop is only 2n - 3 =
1.4 jumps, because each record still has the call into the first body, the final return and the
loop jump.

Code size (TRISC1 `.text`): knob 0 18508 B, knob 1 33812 B (+15.3 KB, every body has a second
copy of its math), knob 2 19940 B (+1.4 KB). With the other blend kernels (TRISC0 2836, TRISC2
1628, reader 4808, writer 1236 B) the program is 29.0 / 44.3 / 30.4 KB, below the 69 KB kernel
config buffer.

## Rough model (not a measurement)
Saved cycles per live record = fewer instructions + 1.36 x T (taken jump) + 2.18 x L
(load-use). t189's fit has ~18.5 stall cycles per body call besides its 13 instructions, which
fits 3T + L with T = 3-5 and L = 2-3. One cycle per live record is ~0.016 ms/view (t189: 0.035
ms/view per cycle per body call).

| | cycles / live record | ms/view |
|---|---:|---:|
| knob 1 | 10.4 + 1.36T + 2.18L = 18.8-23.7 | 0.30-0.38 |
| knob 2 | 4.6 + 1.36T + 2.18L = 13.0-17.9 | 0.21-0.29 |

Not modeled: instruction fetch. Knob 1 doubles the body code; if TRISC1 instruction fetch misses
more, the extra copy can eat the gain, so knob 2 is worth measuring too. Only knob 1 reaches the
0.3 gate in this model, and only just.

## Device gate (next task)
On the existing p100 reservation (`ttp lock p100`): one build, knobs 0 / 1 / 2 interleaved over
paired rounds, bicycle 30 views. Pass: md5 equal to `md5-r82new.txt` on all 30 views for knobs 1
and 2, and a paired gain >= 0.3 ms/view. Also record blend program time per knob (finer than
ms/view). Below the gate: keep the default 0, record the numbers, and stop the
quad/vertical-body idea too (t189: it lives on the same per-call cost).
