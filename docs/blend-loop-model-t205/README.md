# Blend records-loop cost model (task #205, no device)

Question: is there a lever worth >= 0.3 ms/view in the blend records loop (blend is 5.54 ms/view
busy; the loop is 98% of blend MATH time)?

**Answer: yes, two steps that build on each other. Both are md5-safe by construction.**

1. **F, stall-free bodies:** the Blackhole ISA says SFPMAD/SFPMUL/SFPADD take two cycles.
   Every compiled body has 11 back-to-back "MAD result used by the next op" pairs per
   microblock. The hardware holds the next op for one cycle at each of them. That is 31
   cycles per live record (0.49 ms/view of SFPU time), and it explains 11 of the 14.5
   cycles/microblock that t189 could not account for. Reordering the same ops (with renamed
   LREGs) removes all 11 in a single body. `schedule.py` checks that the reordered body
   writes bit-identical values. Model: **+0.40 ms/view** if the instruction FIFO is shallow,
   which is what the measured 366 cycles per live record fits. It gives nothing if the
   stall is really on the RISC side (H1/H3).
2. **A2, F plus replay and RWC tail:** each body issues its 13-op front and its 19-op tail
   from the 32-entry replay buffer. The tail addresses R/G/B/T relative to RWC D, so one
   recording serves all 32 microblocks. The RISC then runs ahead and does the walk, scan and
   staging while the SFPU works. Model: **+0.35 to +1.21 ms/view under every hypothesis and
   FIFO depth, +0.55 to +0.84 under the best-supported one (H4).**

**Verdict: BUILD F then A2 behind one knob (spec below).** Every other idea is SHELVE.

## Inputs
- t172/t189 counts for view 0, per core: 59857 microblocks, 46946 body calls, 21323 live
  records, 26325 records. 1 cycle per live record = 0.0158 ms/view.
- Measured: 366 cycles per live record = 231 dispatch (82/microblock) + 108 staging (31 Tensix
  ops + 77 RISC) + 27 scan. t189: per microblock 53.1 SFPU work + 4.6 nop + 10.2 walk + 14.5
  unexplained.
- t189 disassembly of the default TRISC1 ELF (`../blend-dispatch-t189/out/trisc1-default.dis`).
- Blackhole ISA `tt_llk_blackhole/instructions/assembly.yaml`. SFPMAD, SFPMUL, SFPADD,
  SFPADDI, SFPMULI, SFPLUT and SFPSWAP are each "a two cycle operation". The compiler pads
  SFPSWAP with sfpnop but puts MAD consumers right after the MAD. The output is correct, so
  the hardware stalls those consumers.
- Files: `stalls.py` (stall count, `out/stalls.txt`), `schedule.py` (stall-free single body,
  symbolic equivalence check, `out/schedule.txt`), `model.py` (discrete-event model of RISC
  push + FIFO + SFPU issue, `out/model.txt`).

## Where the cycles go (per live record)
| part | cycles |
|---|---:|
| body ops (1.597 singles x 60 + 0.605 pairs x 109) | 161.8 |
| MAD -> use stalls (11 per microblock, singles and pairs alike) | 30.9 |
| walk (13 RISC instructions x 2.2 calls) | 28.6 |
| staging (31 pushes + 77 RISC) | 108 |
| record scan | 27.2 |
| **sum (no overlap)** | **356.5** (measured 366) |

The parts add up with almost no overlap. So the RISC is rarely ahead of the SFPU: the
instruction FIFO acts shallow, and every RISC-only cycle is an SFPU bubble. Two more checks:
- Load-use pairs (8 per single) cannot also stall, or the stall total (19) would overshoot the
  fit (14.5).
- t191's tail-chained walk measured 0.163 ms/view, not the 0.31-0.52 that a 31-cycle "RISC
  stall" walk (H1) predicted. That fits H4 (walk ~17.5 cycles/call).

Hypotheses in the model, all fitted to 366:
- H1: the 14.5 is RISC stall in the walk (31.4 cycles/call).
- H2: the 14.5 is SFPU stall (11 MAD + 3.5 other).
- H3: the 14.5 is push cost (1.21 cycles/push).
- H4: 11 MAD stalls (counted) + 3.5 on the walk. **Best supported.**

FIFO depth d is unknown, so the model runs d = 2..32.

## Ideas (ms/view saved vs today; range over d = 2..32; gate 0.3)
| idea | cycle model | H1 | H2 | H3 | H4 | md5 | verdict |
|---|---|---:|---:|---:|---:|---|---|
| **F** stall-free hand schedule; same 60/109 ops, reordered + renamed; pair keeps shared loads, <= 4 stalls | -11 SFPU cycles/single, -18/pair | 0 | 0.00-0.43 | 0 | 0.00-0.40 (d=2: 0.40) | identical (schedule.py) | **BUILD, step 1** |
| F2: F with pair = two singles in one call | +11 ops/pair | -0.13 | -0.13-0.32 | -0.16 | -0.13-0.28 | identical | no: keep the shared pair |
| **A2** F + replay 1 (front, 13 ops, D=0) + INCRWC to D=2m + replay 2 (tail, 18 ops + SETRWC) | RISC pushes 32 fewer ops/microblock; +4 RWC ops; pair = two singles | 0.35-1.02 | 0.45-0.68 | 0.53-1.21 | 0.55-0.84 | identical (schedule.py) | **BUILD, step 2** |
| A2 + B-min (SFPLOADMACRO load + delayed store) | -4 issue slots/microblock | +0.04 over A2 | +0.10-0.12 | +0.03-0.12 | +0.07-0.12 | identical | SHELVE: < 0.3 on top of A2 |
| A: one replay early in the body (24-32 ops) | lead only if d >= 16 | -0.10-1.24 | -0.18-0.59 | 0.13-1.39 | -0.15-0.76 | identical | SHELVE: negative at shallow d; A2 dominates |
| B-min alone | -4 issue slots and -4 store stalls/microblock | 0.18 | 0.18-0.35 | 0.22 | 0.18-0.35 | identical | SHELVE: the stall half is what F removes by reordering; after F only the 4 slots remain (A2 + B-min adds 0.04-0.12) |
| B-min + colours in LREGs | -7 slots/microblock | 0.26 | 0.26-0.43 | 0.32 | 0.26-0.43 | identical | SHELVE: most of the H2/H4 gain is the store stalls F removes; the colour part is inside A2; shared pairs have no free LREGs |
| B-full: SFPLOADMACRO doing the R/G/B MAD | -11 slots/microblock | 0.44 | 0.44-0.61 | 0.53 | 0.44-0.61 | **no**: the macro puts the loaded value in as the multiplicand; the update needs it as the addend | SHELVE |
| C: several records per body call | - | - | - | - | - | - | SHELVE: records chain through T; 2 x 13 coefficients do not fit 8 LREGs |
| D: staging on an idle RISC (unpack to DEST) | -77 RISC cycles/live record | - | - | - | - | **no**: unpack goes through tf32 | SHELVE |
| E: fill idle lanes with other records | - | - | - | - | - | - | impossible (t189: a lane is a fixed pixel) |
| replace the 5 swap-pad nops/single with useful loads | <= 0.2 (t189) | | | | | identical | try only inside step 1: unknown if the pad is a data or structural hazard |

A2 at H4: 360 -> 314 cycles per live record at d=2 (-0.74 ms/view), 334 -> 281 at d=32. From
the current tip (14.235 ms/view, iter 194; blend sources unchanged since t199) that is about
13.4-13.7 ms/view (73-75 FPS). The SFPU floor after A2 is
211 cycles per live record. The rest is staging and scan RISC time that A2's replay lead only
partly hides.

Closed levers not re-proposed: blend work cut (#148), mask-0 drop (#185), tail-chained walk
(#191), late claim (#188), quad/vertical bodies, mover table/fold, RAW_STAGE.

## The stall-free single body (schedule.py, microblock m)
Same 60 ops as today; only the order and the LREG numbers change. Each comment says what was
moved and which stall it fills.
```
sfpload L2,256+2m        sfpload L1,320+2m                          # x, y (pushed)
# replay 1 (D=0): front, 0 stalls
sfpload L6,384  sfpload L5,386  sfpadd L2,L10,L2,L6,2  sfpadd L1,L10,L1,L5,2
sfpmul L5,L2,L2,L9  sfpload L4,388  sfpmul L4,L4,L5,L9  sfpmul L2,L2,L1,L9
sfpload L3,390  sfpmad L2,L3,L2,L4  sfpmul L1,L1,L1,L9  sfpload L0,392  sfpmad L0,L0,L1,L2
# pushed middle
sfpnop  sfpswap L0,L9,1  sfpnop  sfpmul L0,L0,L12,L9
sfpload L4,406 (C0, fills)  sfpaddi L0,17150  sfpnop  sfpswap L9,L0,1  sfpnop
sfpexexp L1,L0,0  sfpexman L0,L0,0  sfpshft L0,L1  sfpexexp L1,L0,1  sfpexman L0,L0,1  sfpcast L0,L0
sfpmad L2,L0,L13,L14  sfpload L5,394 (OP, fills)  sfpmad L0,L0,L2,L4
sfpload L6,408 (K99, fills)  sfpsetexp L1,L0  sfpmul L0,L5,L1,L9  sfpload L7,396 (FL, fills)
sfpswap L0,L6,1  sfpnop  sfpload L3,398  sfpload L4,400  sfpload L5,402   # colours
# A2 only: INCRWC to D=2m; replay 2 (tail, addresses relative to D)
sfpmad L1,L7,L11,L0  sfpload L2,192 (T, fills)  sfpsetcc L1  sfpmov L0,L9  sfpencc 3,10
sfpmul L1,L0,L2,L9  sfpadd L0,L10,L10,L0,2  sfpload L6,0  sfpmad L6,L1,L3,L6  sfpmul L0,L2,L0,L9
sfpstore L6,0  sfpload L7,64  sfpmad L7,L1,L4,L7  sfpload L6,128  sfpstore L7,64
sfpmad L6,L1,L5,L6  sfpstore L0,192  sfpstore L6,128     # + SETRWC D=0 (A2)
```
In F, the tail immediates are absolute (192+2m, 2m, 64+2m, 128+2m) and there is no RWC op.
Nothing moves into or out of the setcc..encc region. `schedule.py` runs both bodies
symbolically and compares every DEST word written and the constant LREGs. Negative tests (an
operand swap, a load moved into the CC region) are flagged.

## Build spec (follow-up task)
Knob `GSPLAT_TT_BLEND_SCHED`, read in `render/host/blend_device.cpp` next to
`BLEND_JUMP_WALK`, sets the define `BLEND_SCHED`: 0 = today (default until measured),
1 = F, 2 = A2. It is used only with `BLEND_USE_JUMP_WALK` (bodies take no arguments).

**Level 1 (F)**, in `render/kernels/compute/alpha_blend_compute_mb.cpp`:
- Add raw-TTI body templates `sched_single<IX>()` and `sched_pair<IXA,IXB>()`
  (`TTI_SFPLOAD/SFPMAD/SFPMUL/SFPADD/SFPADDI/SFPSWAP/...` from `ckernel_ops.h`).
- Copy each op's opcode, instr_mod and immediate from today's objdump; change only the order
  and the LREGs, as in the listing above.
- Addresses per microblock: x 256+2m, y 320+2m, T 192+2m, R 2m, G 64+2m, B 128+2m;
  coefficients at today's fixed addresses.
- Pair: hand-schedule today's shared 109-op pair down to <= 4 MAD -> use stalls (22 today;
  most are fixed by swapping neighbours or interleaving the two tails). Verify it by adding it
  to `schedule.py` (`run()` and `mad_stalls()` take any body). Fallback is two F singles in
  one body (F2), which models about 0.1 ms/view worse.
- Fill `g_blend_bodies` from these templates when `BLEND_SCHED >= 1`.
- In the objdump, check: 60 ops per single, the order as listed, and no `ttreplay` in the
  bodies.

**Level 2 (A2):**
- Replay slots: replay 1 = the 13 front ops in slots 0-12. Replay 2 = the 18 tail ops with
  relative immediates (T 192, R 0, G 64, B 128) plus
  `TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D)` in slots 13-31.
- Body:
  1. push the x/y loads;
  2. `lltt::replay(0, 13)`;
  3. push the middle (24 ops + 3 colour loads);
  4. push `TTI_INCRWC(0, k, 0, 0)` steps summing to 2m (k <= 15 each, so up to 5; the count
     is compile-time per m);
  5. `lltt::replay(13, 19)`.
  Pairs = two A2 singles in one body.
- Record with `lltt::record(0, 13)` + the front ops and `lltt::record(13, 19)` + the tail
  ops (default NoExec). Do this at the start of each tile's records loop, after every other
  replay use on TRISC1: the compiler's auto-replay records slots 0-3 in the `_start` zero-fill
  loop and slots 0-4 in today's pair bodies.
- Check in the objdump that no other `ttreplay s,l,1,*` (record) runs between our record and
  the loop. If the compiler adds replays inside the loop, turn its replay pass off for this
  kernel. The option is `-mtt-tensix-optimize-replay`; confirm the negative form with
  `--help=target` on the build host.
- D must be 0 outside bodies: staging uses absolute addresses, ADDR_MOD_7 has zero increment,
  and replay 2 ends with SETRWC D=0.

**Gates (each level, same build, `ttp lock p100`, bicycle 30 views):**
- md5 46a725ab on all 30 views. The bodies are identical by construction, so a mismatch is a
  transcription bug: fix it, do not tune it.
- Paired A/B 0 vs 1 and 0 vs 2 (and 1 vs 2), with the tracy blend busy ms/view per level.
  Make the best level the default if it beats 0 by >= 0.3 ms/view. Otherwise leave 0, record
  the numbers and shelve.
- How to read level 1: about +0.4 confirms H2/H4 (shallow FIFO). About 0 means H1/H3; A2 still
  models >= 0.35 there, so build level 2 either way.
- Optional, same session: read TRISC1 perf counters around one view's records loop: FPU bank
  SFPU_INSTRUCTION (id 1), INSTRN_THREAD FPU_INSTRN_AVAILABLE_1 (16) and THREAD_STALLS_1 (25);
  `tt-llk/tests/helpers/include/counters.h` shows the register reads. They give the SFPU busy
  share directly.
