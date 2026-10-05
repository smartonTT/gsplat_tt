# t245: unpacker-staged blend coefficients (model, no device)

Question: can TRISC0 or the unpacker put each live record's coefficients in DEST,
so TRISC1 stops staging them (14 ring loads + 14 SFPLOADI pushes + 17 fixed ops
per live record at iter 198)? Is it worth >= 0.3 ms/view, and can md5 46a725ab stay?

Short answer: yes, as one specific variant (U2 below). The re-fitted model gives
+0.39 to +0.69 ms/view (+0.66 to +0.94 with a MOP). It is md5-identical by
construction. Build it in stages with the kill gates below.

Files: `model.py` (imports the t230 model, which imports t205), `out/model.txt`.

## 1. Re-fit to t231

t230 predicted too much for the decode-ahead steps because it assumed decode-ahead
left 20 RISC cycles of staging per live record. Fitting the three t231
measurements (ms/view saved vs the A2 tip) at the t230 fits that survive gives:

| fit (t230) | A2 cyc/rec | RISC staging after RAW | after decode-ahead | scan left at knob 2 | S2 cyc/rec |
|---|---|---|---|---|---|
| H3 push cost, FIFO 16 | 303 | 64 (t230: 60) | **54** (t230: 20) | 5.0 (t230: 8) | 259 |
| H1 RISC stalls, FIFO 32 | 295 | 64 | **54** | 3.4 | 250 |
| H3 push cost, FIFO 8 | 312 | 64 | **54** | 5.0 | 267 |
| H4 MAD latency, FIFO 32 | 268 | 64 | 54 | no fit (S2 unreachable) | - |

The fitted S2 point (250-267 cycles) matches t231's ~255 measured estimate.
So TRISC1 still spends about 54 RISC cycles per live record on staging. That
covers the 14 L1 ring loads, the ring poll and the counter updates.
`da_stage` already loads two words ahead, so the time is L1 load latency, not
missing software pipelining. Pushing the 31 staging ops alone, with the ring
loads kept (row X), gives only +0.32 to +0.50. The real cost is the RISC side.

## 2. Ways to get the coefficients into DEST

The blend reads coefficients as lane-broadcast vectors in DEST slot 6
(`DR_S`, `S_MX..S_INV`, `alpha_blend_compute_mb.cpp`). A vector covers 4 DEST
rows x 8 columns of one parity, so each value has to sit in 32 DEST positions.

| path | verdict |
|---|---|
| SrcA/SrcB or FPU (MOVB2D, eltwise) | No. 19-bit (TF32) registers break md5, as seen when xform moved to the FPU. |
| Unpacker ROW/SCALAR broadcast | No. On BH it unpacks one row (`llk_unpack_A.h`, x_end=1). The broadcast itself is MOVB2D/MOV_8_ROW_BRCST on MATH (`llk_math_eltwise_unary_datacopy.h`), which puts TRISC1 pushes and cfg toggles back. |
| RISC-to-DEST window 0xFFBD8000 (`ckernel_dest.h`) | No. It needs 32 stores per value, about 288 per record on TRISC0, which is above the TRISC1 pace. It is only used by debug code (`ckernel_debug.h`). |
| U1/U3: unpack straight into slot 6 | No. The front replay (slots 0-12) loads MX/MY/A/B/C at fixed slot-6 addresses, and the 32-slot replay buffer is full. Double-buffering would need a second front recording or a DEST base flip, and the flip also moves the R/G/B/T accumulators. With one buffer, every record waits for the unpack round trip. The slot is busy until the last microblock call of the record (2.8 calls per live record on average). Modeled bound: +1.03 to +1.14 at 10 exposed cycles, +0.60 to +0.67 at 40. |
| **U2: unpack into a separate DEST staging area, TRISC1 copies with fixed ops** | **Feasible.** The copy decouples the unpacker from slot 6, so no double buffer is needed: the unpacker fills record N+1 while body N runs. |

U2 in detail:

- **TRISC0 L1 writes.** TRISC0 writes the 9 values to a small L1 buffer as
  16-word rows. Each row holds two values interleaved, even and odd columns,
  which is 8 stores per value.
- **UNORM values.** A UNORM q (OP, CR, CG, CB) is sent as `0x4B000000 | q`,
  which is the fp32 value 2^23 + q. That costs one OR on TRISC0.
- **Unpack into DEST.** TRISC0 issues the fp32 unpack-to-DEST into a free DEST
  row block. Each L1 row is repeated into 4 DEST rows, using a zero source
  y-stride or 4 UNPACRs on the same address. Then it posts a semaphore.
- **TRISC1 copy.** TRISC1 pushes SEMWAIT and then 27 fixed SFPU ops:
  - 5 raw values: SFPLOAD + SFPSTORE each.
  - 4 UNORM values: SFPLOAD, SFPADDI -2^23, SFPMUL fl(1/65535), SFPSTORE each.
  - 1 INV load.
  It then pushes SEMPOST. TRISC1 keeps only the mask/marker read from the ring.
- **MOP option.** The 27 ops are identical for every record, so one MOP push can
  replace them (U2m). The blend kernel does not use the MOP today.

## 3. Model on top of S2 (cycles per live record, ms/view saved)

| design | saved, all fits |
|---|---|
| U2, 27 ops pushed, 12 RISC cycles left | +0.60 .. +0.69 |
| U2, 20 RISC cycles left | +0.52 .. +0.58 |
| U2, 30 RISC cycles left (ring poll as slow as today) | +0.39 .. +0.42 |
| U2m, 27 ops in one MOP push, RISC 12 | +0.66 .. +0.94 (reaches the SFPU bound, 208) |
| U3 slot 6 direct, exposed latency 10 / 40 (not buildable) | +1.03..+1.14 / +0.60..+0.67 |
| U1 ideal, all staging gone (bound) | +1.09 .. +1.35 |
| X: keep ring loads, push the 31 ops in one go | +0.32 .. +0.50 |

TRISC0 load check, in instructions per live record:
- **Today:** about 55, plus a dead-record walk of about 27.
- **U2:** about 129 + 27 = 156. That is under the TRISC1 pace after U2
  (208-232), with roughly 25% margin.
- **If the unpacker cannot repeat a row:** TRISC0 would have to store 4 times
  as much, about 345 + 27. That is above the TRISC1 pace, so U2 would lose.
  This is the first kill gate.

## 4. md5

- **Raw fp32 (MX, MY, A, B, C, INV).** BH unpack-to-DEST at 32-bit input copies
  bits exactly (`llk_unpack_A.h`). The copy is SFPLOAD/SFPSTORE in fp32. Same
  bits.
- **UNORM values.** (2^23 + q) - 2^23 = q is exact for q < 2^16, so SFPADDI gives
  exactly what SFPCAST gives today. The SFPMUL by 0x37800080 is unchanged. Same
  bits.
- **Not used:** the SrcA/SrcB paths, which lose precision, and any software
  decode of the UNORM multiply on TRISC0, which would need an exhaustive check
  of all 65536 values.
- **Risks the md5 check will catch:**
  - Unpacker flushing of denormal or negative-zero inputs.
  - The BH zero-flag bug (budabackend#2730): an unpack-to-DEST write in the same
    cycle as a packer ZEROACC loses the flag clear. TRISC0 must not stage across
    the tile-end `tile_regs_release`. The mid-tile T readback already releases
    without ZEROACC (`non_zeroing_pack_release`).

## 5. Verdict and build spec

**Build U2.** It clears the 0.3 gate in every surviving fit, even with 30 RISC
cycles left. Behind a knob `GSPLAT_TT_BLEND_UNPACK_STAGE` (default 0; knob 1 = U2,
knob 2 = U2 + MOP). It needs `BLEND_DECODE_AHEAD=2` and BLEND_SCHED=2.

**Stage 0: micro-test**, device, about 1 hour, no performance numbers. In a
blend-only test kernel:
- TRISC0 writes known L1 rows and issues an fp32 unpack-to-DEST into the
  staging rows, with row repeat (zero y-stride, or 4 UNPACRs on the same
  address).
- TRISC1 does SEMWAIT, then SFPLOADs every lane and dumps it.
- Checks:
  - All 32 lanes of each of the 9 vectors hold the exact bits.
  - Values include 2^23+q for q in {0, 1, 65535}, negative zero, and a
    denormal.
  - The DEST section matches the math thread's current dest base after a
    `tile_regs_acquire` flip.
- **Kill** if a row cannot be repeated without 4x TRISC0 stores, or if any bit
  differs in a way the copy cannot fix.

**Stage 1: knob 1.**
- TRISC0 `da_put_rec` writes the value rows and issues the unpack plus a
  semaphore post.
- The ring keeps only the mask and markers.
- TRISC1 replaces `da_stage` with SEMWAIT + 27 fixed ops + SEMPOST.
- Use a 2-entry L1 row buffer, so TRISC0 can write N+2 while N+1 is in flight.
- The DEST staging rows go outside the R/G/B/T tiles and slot 6.

**Stage 2: knob 2.** Program the MATH MOP once at blend init, with the 27 ops
as the inner loop. Each record then pushes one MOP. Re-program after any LLK
call that changes the MOP.

**Gates per stage**, on the bicycle 30-view, under `ttp lock p100`:
- md5 46a725ab.
- A hero screenshot plus a visual tile check.
- ms/view against knob 0, in the same session.
- Tracy check:
  - tile_blend_sfpu makespan (3.760 today).
  - TRISC0 busy per tile must stay below TRISC1's.

**Kill gates:**
- Stage 1 saves less than 0.15 ms/view. That is below half the low end of the
  model and would mean the RISC cost sits in the ring poll, not the loads.
- Or TRISC0 becomes the bound: Tracy shows TRISC1 waiting at SEMWAIT on more
  than 20% of records.
- Ship at 0.30 or more. Between 0.15 and 0.30, keep the knob, try stage 2,
  then decide.
