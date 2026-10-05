# t230: Tracy re-baseline after BLEND_SCHED=2, and a model of more TRISC1 push cuts

Tip 06d4fa7 (BLEND_SCHED=2 default), yyzo-bh-07 p100a, 30-view bicycle,
`GSPLAT_TT_KCFG_EXTRA_KB=32`, profiler build. Scripts: `remote_tracy.sh`, `drive.sh`.
Raw output: `out/tracy-{zones,gaps,roofline}.txt`, `out/tracy-capture.log`.
The capture matches the t229 level-2 traced run.

## 1. Busy time per program (ms/view, Tracy, all-core busy)

| program | busy ms/view |
|---|---:|
| pfwc | 2.330 |
| K2 (pairs) | 0.984 |
| sort_ol (town + emit) | 1.436 |
| mat + blend (one merged program) | 7.562 |
| all-core idle gaps | 0.023 |
| device span | ~12.31 |

mat and blend are now one program, so their busy time comes from zone makespans:

| zone | makespan ms/view |
|---|---:|
| mat_cull_mask | 2.836 |
| sort_subchunk_mat | 2.783 |
| tile_blend_sfpu | 4.417 (5.381 at SCHED=0) |
| tile_blend_load | 4.341 |
| pfwc | 2.269 |
| sort_ol_emit | 1.281 |
| sort_ol_town | 1.171 |
| k2_pairs | 0.922 |

## 2. Blend busy time per RISC (merged program 3)

| RISC | program 3 mean / busiest core (ms) | blend-phase zone (mean / max) |
|---|---:|---|
| BRISC | 6.836 / 6.961 | ~4.46 (program minus mat 2.375) |
| NCRISC | 6.807 / 7.319 | tile_blend_load 4.315 / 4.391 |
| TRISC0 | 6.90 / 7.40 | tile_blend_sfpu 4.387 / 4.468 |
| TRISC1 | 6.90 / 7.40 | tile_blend_sfpu 4.387 / 4.468 |
| TRISC2 | 6.90 / 7.40 | tile_blend_sfpu 4.387 / 4.468 |

Caveat: TRISC0 and TRISC2 walk the same record loop as TRISC1, with no-op dispatch,
so their zone spans are the same as TRISC1's. Tracy cannot show that TRISC0 and
TRISC2 are idle for most of the blend. Only TRISC1 issues SFPU work.

Kernel time per frame, summed per RISC on the busiest core: BRISC 11.50,
NCRISC 11.80, TRISC 10.83-10.85 ms. Overlap bound 11.80 vs span 12.31.

## 3. Can more replay cut TRISC1 pushes? No: the replay buffer is full

- The Blackhole replay buffer has 32 slots. The ISA says REPLAY uses only the
  log2(depth) low bits of `start_idx` and `len`, and LLK assumes 32 slots.
- An A2 single body has 59 instructions that do not depend on m:
  - the front 13, in slots 0-12
  - 27 middle instructions, 5 of them SFPNOP
  - the tail 18 plus SETRWC, in slots 13-31
- With 32 slots, front plus tail is the best cover: it saves 30 pushes per single.
  Replaying the 27 middle instructions instead saves only 26.
- A second replay section per pair (the "per-pair mid sections") would need slots
  that do not exist. Re-recording per record costs about as many pushes as it saves.
- So body replay is at its cap. What is left in the body is small (M1, M2 below).

## 4. Model (`model.py` → `out/model.txt`)

The model reuses the t205 stream model (`docs/blend-loop-model-t205/model.py`).

**Step 1: calibration.** The model was fitted to the two #229 measurements:
- level F (stall-free bodies): +0.022 ms
- A2: -1.015 ms untraced

Best fits:

| hypothesis | FIFO depth | error | A2 cycles per live record |
|---|---:|---:|---:|
| H3: push cost | 16 | 0.11 | 306 |
| H1: RISC stalls (jumps, load-use) | 32 | 0.12 | 300 |
| H4: MAD latency + RISC | 32 | 0.20 | 268 |
| H3: push cost | 8 | 0.24 | 312 |

The SFPU-only bound is 210 cycles per live record, so A2 still loses about 60-100
cycles per live record to TRISC1 issue. Most of that loss is outside the body:
- staging: 31 pushes + 77 RISC cycles
- record scan: about 27 cycles
- walk: about 13 cycles per call

One cycle per live record = 0.0158 ms/view.

**Step 2: candidates on top of A2.** Saving in ms/view, as the range across the
four fits:

| candidate | saving ms/view | gate (0.3) |
|---|---|---|
| M1: fill the 3 spare middle nops | +0.13 to +0.14 | no |
| M2: MOP holds 7 middle instructions (1 push) | +0.11 to +0.27 | no |
| M3: shared-load pair, pushed (F pair 109) | -0.13 to -0.41 | loss |
| S1: RAW_STAGE=1 (-17 RISC instructions per record) | +0.27 | no (zero-build probe) |
| **S2a: TRISC0 decode-ahead writes the 14 pre-encoded SFPLOADI words per record into an L1 ring; TRISC1 only loads and pushes** | **+0.82 to +0.90** | **yes** |
| S2a': same, pessimistic L1 load cost (35 RISC) | +0.66 | yes |
| **S2: S2a + live-only record list from TRISC0 (scan 27 → 8)** | **+0.91 to +1.20** | **yes** |
| S3: pipeline staging behind the front replay | +0.00 to +0.43 | fit-dependent |
| S4: S2 + S3 | +0.92 to +1.36 | yes |
| S5: bound, all scan and staging RISC work gone | +0.92 to +1.45 | bound |

## 5. Recommendation

- **Do not build** more body replay: per-pair mid sections, MOP middle (M2), or
  filling the nops (M1). Each is ≤0.27 ms/view and the replay buffer is full.
- **Build S2a, then S2 (live-record staging on an idle TRISC).**
  - Modeled +0.66 to +0.90 ms/view for S2a and +0.91 to +1.20 for S2.
  - md5 should not change by construction. TRISC1 issues the same instruction words
    in the same order; only the RISC that computes the SFPLOADI immediates changes.

**Build sketch:**
- TRISC0 runs the same record walk TRISC1 does today:
  - It decodes each live record's coefficients into the 14 SFPLOADI words.
  - It writes them, with the record's microblock mask, to an L1 ring of N slots
    (about 64 B per record).
  - It bumps a volatile produced counter.
- TRISC1 waits on the counter. For each record it does 14 L1 loads, then
  `TT_SFPLOADI` with the loaded immediate (or `INSTRN` with a pre-built word), the
  constant ops, and the A2 bodies. Then it bumps a consumed counter.
- S2 adds: TRISC0 writes only live records, so TRISC1 skips the scan.

**Risks to check in the build:**
1. The T readback every 512 records needs all threads: UNPACK waits on CB_T_RB,
   PACK runs pack_tile, and there is a mailbox sync. TRISC0 must interleave decode
   with its readback duty without deadlock. The ring must not let TRISC0 run past a
   readback point it still has to serve.
2. Blackhole L1 coherence: use `invalidate_l1_cache()` and volatile reads on the
   counters (the kernel already uses this pattern near line 86).
3. L1 headroom for the ring, given kcfg +24 KB (Tracy needs +32).
4. With a live-only list, the `g_seen % 512` readback trigger must stay at the same
   point in the record order. This is md5-safe if the readback happens before the
   next live record, because dead records do not change T.

**Gate:**
- At least 0.3 ms/view paired untraced on yyzo-bh-07 p100a, with md5 46a725ab.
- Kill early if S2a alone measures under 0.15 ms/view.
- Run RAW_STAGE=1 as a zero-build A/B in the same session to calibrate the model.
  The model predicts +0.27. If RAW_STAGE gives about 0, the RISC-cost fits (H1/H4)
  are wrong and S2a's expected saving drops toward the H3 numbers.
