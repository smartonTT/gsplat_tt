# Stall-free blend bodies F and A2 (task #219, code only, no device)

Builds the #205 records-loop lever (`docs/blend-loop-model-t205/README.md`, listing and build
spec) behind one knob. Nothing here ran on the device; the gate plan is at the end.

## What changed
- `render/host/blend_device.cpp`: `GSPLAT_TT_BLEND_SCHED` -> kernel define `BLEND_SCHED`,
  next to `BLEND_JUMP_WALK`. **0 = today (default)**, 1 = F, 2 = A2. It is a JIT define, so
  the env var takes effect per process without rebuilding the .so.
- `render/kernels/compute/alpha_blend_compute_mb.cpp`, only under `BLEND_USE_JUMP_WALK`:
  - Level 1 (F): raw-TTI `sched_single<IX>()` (60 SFPU ops) and `sched_pair<IXA,IXB>()`
    (109 ops, shared loads kept). Opcode, instr_mod and immediate of every op are copied from
    `docs/blend-dispatch-t189/out/trisc1-default.dis`; only the order and the LREGs differ.
    Addresses: x 256+2m, y 320+2m, T 192+2m, R 2m, G 64+2m, B 128+2m. `blend_pair_body<J,PM>`
    calls them, so `g_blend_bodies` holds them. The pair reached 0 MAD->use stalls (spec:
    <= 4), so the two-singles fallback was not needed.
  - Level 2 (A2): `blend_sched_record()` records the 13 front ops into replay slots 0-12 and
    the 18 tail ops (T 192, R 0, G 64, B 128, relative to RWC D) plus
    `SETRWC(CLR_NONE, 0, 0, 0, 0, SET_D)` into slots 13-31, with `lltt::record<NoExec>`. It runs
    once per tile right after `TC_PART1(stage)`, before the records loop. A body is: x/y
    loads, `replay(0, 13)`, the 27-op middle, `INCRWC` steps of <= 14 summing to 2m (0-5 of
    them, fixed per m at compile time), `replay(13, 19)`. A pair is two A2 singles.
  - Levels 1 and 2 `#error` unless `BLEND_CONST_HOIST`, `BLEND_PIXEL_FLOOR` and
    `!BLEND_FPU_QF_ABL` (the build the ops were copied from). These are the defaults.
- `docs/blend-loop-model-t205/schedule.py`: the pair schedule and hazard rules (commit e9e8bfc).

## Checks (all pass)
| check | result |
|---|---|
| `schedule.py` (`../blend-loop-model-t205/out/schedule.txt`) | F single and A2: 32 bodies bit-identical to compiled, 0 MAD->use stalls (compiled 11). F pair: 16 bodies identical, 109 ops, 0 stalls (compiled 22). 0 hazard-rule violations. |
| `check_cpp.py` (`out/check_cpp.txt`): encodes the kernel source's TTI calls with the `ckernel_ops.h` field layouts | Encoder matches all 19 opcodes' t189 words. Level 1: 48/48 bodies = listing, single 60 / pair 109 ops, no ttreplay. Level 2: record fills 32 slots, issues nothing; 48/48 bodies = listing after replay. Level 0: new code compiled out. Swapping two operands or two ops in the source makes it fail (tried). |
| `build.sh` (`out/build.txt`): compile-only TRISC1 build on yyzo-bh-07, private scratch dir, under `ttp lock p100` | Levels 0/1/2: rc 0, 0 warnings. .text 18508 / 18700 / 12924 bytes. Level 0 .text is instruction-for-instruction identical to the tip (4063ed7) kernel built the same way. |
| `check_elf.py` (`out/check_elf.txt`, objdump `out/trisc1-sched{1,2}.dis`) | Level 0: 48/48 bodies = t189. Level 1: single 60 ops x32, pair 109 x16, all = listing, 0 ttreplay in bodies. Level 2: one record site (`_start` 0x76bc), slots = front / tail + SETRWC; 48/48 bodies = listing after replay; TRISC1 pushes 31-36 instructions per single (60 today) and 63-72 per pair (106). |

Replay use at level 2: our two records are the only `ttreplay` in the whole TRISC1 ELF. The
compiler's own replays (slots 0-3 in the `_start` zero fill, 0-4 in today's pair bodies) are
gone at level 2 and it added none inside the loop, so its replay pass does not need turning
off. D = 0 at body entry is the existing invariant (staging uses absolute addresses,
ADDR_MOD_7 has zero increment) and every A2 body ends with SETRWC D = 0.

Build inputs: the t188 JIT cache entry of this kernel (generated headers; its defines equal
today's host defaults) and the compile/link flags tt-metal logged for it in t60. Rerun:
`build.sh <tree> <scratch>` on the host, then
`python3 check_elf.py <t189 or level-0 dis> <level-1 dis> <level-2 dis>` (absolute paths) and
`python3 check_cpp.py [tt-metal root]`.

## Expected effect (model, not measured)
From #205: F +0.40 ms/view if the instruction FIFO is shallow, 0 if the stall is on the RISC
side. A2 +0.35 to +1.21 under every hypothesis, +0.55 to +0.84 under the best-supported one.

## Gate plan for the device task
Same tree and build for every level; each sync+build+run under `ttp lock p100`; bicycle, 30
views; `GSPLAT_TT_BLEND_SCHED=0/1/2` in the env. TRISC1 text is +192 B at level 1 and
-5.6 KB at level 2; if level 1 hits a kernel config buffer TT_FATAL, raise
`GSPLAT_TT_KCFG_EXTRA_KB` for all arms.
1. md5 46a725ab on all 30 views at levels 1 and 2. The bodies are identical by construction,
   so a mismatch is a transcription or semantics bug: fix it, do not tune it. Level 1 passing
   and level 2 failing points at D (INCRWC step, SETRWC, D not 0 at body entry) or a replay
   slot clobbered after the record.
2. Paired A/B 0 vs 1, 0 vs 2 and 1 vs 2, with tracy blend busy ms/view per level.
3. Make the best level the default only if it beats 0 by >= 0.3 ms/view. Otherwise keep 0,
   record the numbers and shelve.
4. Reading level 1: about +0.4 confirms the MAD-stall hypothesis (shallow FIFO). About 0 means
   the stall is on the RISC side; level 2 still models >= 0.35 there, so judge level 2 on its
   own numbers.
5. Optional, same session: TRISC1 perf counters around one view's records loop
   (SFPU_INSTRUCTION, FPU_INSTRN_AVAILABLE_1, THREAD_STALLS_1; see
   `tt-llk/tests/helpers/include/counters.h`) to get the SFPU busy share directly.

## Device gate (task #229, measured)
yyzo-bh-07 (Blackhole p100a), bicycle 30 views 1024x1024, one sync + build of 7f0850d (tip
356f66e + drivers), each step under `ttp lock p100`. Drivers and raw logs: `dev-t229/`
(`drive.sh`, `remote_time.sh`, `remote_tracy.sh`, `verify.sh`, `out/`).

md5: all 11 untraced arms (smoke L1/L2 + 3 rounds x L0/L1/L2) give 30/30 views identical to
`md5-r82new.txt` (46a725ab). The smoke ran at the default kernel config size (no TT_FATAL).

Untraced, 3 rounds, order rotated (0 1 2 / 1 2 0 / 2 0 1), avg_frame_ms per view:

| level | r1 | r2 | r3 | mean | STAGES blend |
|---|---|---|---|---|---|
| 0 (compiled) | 13.609 | 13.665 | 13.620 | 13.631 | 9.148 |
| 1 (F, raw-TTI) | 13.648 | 13.635 | 13.676 | 13.653 | 9.199 |
| 2 (A2, F + replay) | 12.602 | 12.609 | 12.638 | 12.616 | 8.168 |

Paired: 1 vs 0 +0.022 (+0.039 / -0.030 / +0.056), 2 vs 0 -1.015 (-1.007 / -1.056 / -0.982),
2 vs 1 -1.037 ms/view. Level 2 = 79.3 FPS.

Tracy, 30 views each at `GSPLAT_TT_KCFG_EXTRA_KB=32` (the traced runs are slower in wall time; read
the device numbers): blend program (mat+blend segment) busy ms/view 8.539 / 8.610 / 7.565 for levels
0 / 1 / 2; `tile_blend_sfpu` makespan 5.381 / 5.444 / 4.413 ms/view; `tile_blend_load` 5.291 / 5.345 /
4.343. pfwc, K2 and sort_ol are unchanged (2.33 / 0.98 / 1.44).

Reading: level 1 is about 0, so the MAD->use stall is not what limits the bodies. The stall is on
the RISC side: TRISC1 cannot push instructions fast enough. Level 2 cuts the pushed instructions
per body about in half (31-36 vs 60 single, 63-72 vs 106 pair) and saves -0.97 ms/view of blend
SFPU makespan. That is above the #205 model's best-supported range (+0.55 to +0.84) and inside
its full range (+0.35 to +1.21).

Decision: KEEP level 2. `GSPLAT_TT_BLEND_SCHED` defaults to 2 in `render/host/blend_device.cpp`
(=0 compiled bodies, =1 F).

Default verify on 29c288b (default flipped to 2; `verify.sh`, 2 rounds, order swapped): default
12.699 / 12.630, mean 12.665 ms/view (79.0 FPS); `GSPLAT_TT_BLEND_SCHED=0` 13.691 / 13.603, mean
13.647. Paired -0.982 ms/view. All 4 arms 30/30 views md5-identical (46a725ab). Tracy of level 2:
`opt/profiler/ttw-196/render.tracy` (iter 196).
