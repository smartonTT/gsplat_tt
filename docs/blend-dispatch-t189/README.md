# Blend per-dispatch cost model (task #189, no device)

Question: can blend's per-dispatch overhead be cut by >= 0.3 ms/view?

**Answer: only one lever can reach 0.3 ms: a tail-chained mask walk. The model puts it at
0.14-0.52 ms/view (middle estimate 0.31), so it is a conditional go: build it behind a knob (off
by default) and keep it only if a paired A/B on the device shows >= 0.3. Nothing else clears the
gate. Packing more records into each dispatch is not possible.**

## Inputs
- Default blend TRISC1 ELF (defines: COEF_DEST, JUMP_WALK, SFPU_UNORM, CONST_HOIST, FAST_TRED,
  PIXEL_FLOOR; RAW_STAGE=0). Built in the t188 JIT cache on yyzo-bh-07 and only read there with
  objdump. Its source differs from this tree only by the t172 instrumentation, which is compiled
  out by default. Output: `out/trisc1-default.dis`.
- Per-body instruction counts: `count_bodies.py out/trisc1-default.dis` (`out/body_counts.txt`).
- Counts and fit from t172 (`docs/blend-fixed-t172/out/tc_split.txt`, view 0): 82 cycles per
  microblock, 108 per live record, 22 per record. Pair ratio from t148: 1.275 microblocks per
  body call. The blend loop has not changed since t148.
- `model.py` (`out/model.txt`): all numbers below. 1 cycle per microblock = 0.044 ms/view; 1
  cycle per body call = 0.035 ms/view.

## What one dispatch issues
| | SFPU ops | of which load / store / nop | RISC |
|---|---:|---:|---:|
| single body (1 microblock) | 60 | 18 / 4 / 5 | ret |
| pair body (2 microblocks) | 109 (104 + 5 replayed) | 26 / 8 / 8 | ret |
| walk per body call (`dispatch_blend_jump`) | - | - | 13: ctz, andi, srl, andi, sh1add, sh2add, **lw -> jalr** (load-use), ret, sll, andn, bnez; 3 taken jumps |

A pair saves 11 ops (10 shared coefficient loads, 2 nops, minus 1) against two singles. 43% of
microblocks run in pair bodies and 57% in singles; there are 0.78 calls per microblock.

## Where the 82 cycles per microblock go (3.65 ms/view)
| part | cycles | ms/view |
|---|---:|---:|
| SFPU work (loads, math, stores) | 53.1 | 2.35 |
| sfpnop (hazard pads after swap/exexp) | 4.6 | 0.20 |
| walk RISC instructions | 10.2 | 0.45 |
| stalls not explained by the instruction count | 14.5 | 0.64 |

Body work, mostly the arithmetic, is 65% of the time. The fixed per-dispatch overhead (walk +
stall + nops) is ~1.3 ms/view. Each body call costs 31.5 non-SFPU cycles for only 13 RISC
instructions, so about 18 cycles per call are stalls: 3 taken jumps plus the `lw` of the table
entry used at once by `jalr`. Over the whole record loop, 18% of its cycles (1.0 ms/core) are not
covered by issued instructions. Caveat: the fit's microblock and live-record coefficients are
correlated (about 2.8 microblocks per live record), so part of this stall may belong to staging.

## Levers (ms/view)
| lever | model | verdict |
|---|---:|---|
| fill every sfpnop | <= 0.20 | no: the compiler already scheduled them; they follow swap/exexp |
| drop the duplicate k99/floor loads in pair bodies | 0.02 | no |
| larger bodies (vertical or 2x2 quads) | 120 bodies at up to ~0.8 KB each, ~60 KB of TRISC1 code; coefficients do not fit the 8 LREGs, so a quad only saves calls | no: code size; gain bounded by the walk below |
| every single paired (ceiling) | 0.54 | impossible: arbitrary pairs need 496 bodies |
| pack several records into one dispatch (lane occupancy is 32%) | - | no: a lane is a fixed pixel and records depend on each other through T, so there is no free lane to fill. Finer masks only add dispatches. Per-lane coefficients would cost more than they save. |
| cheaper live-record staging (108 cycles, ~92 issued) | - | out of scope; RAW_STAGE (#146) measured -0.23 and was shelved |
| **tail-chained walk** | **0.14 / 0.31 / 0.52** (low / mid / high) | **conditional go** |
| ceiling: all walk instructions and stalls gone | 1.10 | not reachable |

## Build spec: tail-chained walk (if it is run)
- Knob `GSPLAT_TT_BLEND_CHAIN_WALK` -> define `BLEND_CHAIN_WALK` (default 0) in
  `render/host/blend_device.cpp`, next to `BLEND_JUMP_WALK`.
- Bodies take the mask: `void body(uint32_t mask)`. Each body computes its successor first:
  `mask &= ~(3u << B); if (!mask) { <math>; return; }`, then
  `b = ctz(mask) & ~1; next = g_blend_bodies[2*b + ((mask >> b) & 3)]`, then the math, then
  `return next(mask);`. GCC turns this into a sibling call (`jr`). Check in objdump that it is a
  `jr`, not `jalr` + `ret`, and that the `lw` is issued before the SFPU ops. The call site does
  one `g_blend_bodies[...](mask)` per live record.
- Per call this gives 1 taken jump instead of 3, 2 fewer instructions, and the table load is
  hidden under the body's SFPU issue. Same body order, so the output should be md5-identical.
- Gate: md5 equal to `md5-r82new.txt` on all 30 views, and a paired A/B (knob 0/1, same build,
  `ttp lock p100`) of >= 0.3 ms/view. Otherwise leave it off and record the number.
- If the A/B is below the gate, also stop the quad/vertical-body idea: both live on the same
  per-call cost.
