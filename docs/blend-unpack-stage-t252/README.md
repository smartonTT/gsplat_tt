# t252: U2 unpacker-staged blend coefficients (GSPLAT_TT_BLEND_UNPACK_STAGE)

Builds the t245 U2 design (docs/blend-unpack-stage-t245/README.md section 5) on top of iter 198
(DECODE_AHEAD=2, BLEND_SCHED=2). The knob is forced to 0 unless DECODE_AHEAD == 2.

## Design as built (stage 1, knob 1)
- TRISC0 decodes each live record into 5 rows x 16 fp32 words in one of 8 L1 buffers
  (ring CB 10 grows by 40 pages): even/odd columns = MX/MY, A/B, C/OP, CR/CG, CB/-.
  OP and the colour channels are packed as 2^23 + q (0x4B000000 | q) so the unpacker copies exact bits.
- It waits for semaphore UNPACK_TO_DEST at max, issues 20 UNPACRs (unpack-to-dest, fp32, ch1 z stride
  64 B = one DEST row, ch0 z step 16 datums) so each L1 row lands 4x in DEST rows 448+4i..+3 (slot 7),
  then posts the semaphore. The DA ring now carries only the mask/marker word.
- TRISC1 waits on the semaphore, does 13 SFPLOAD/SFPSTORE pairs into slot 6 (OP/CR/CG/CB subtract
  2^23 and scale by INV with SFPADDI+SFPMUL), then SEMGET.
- Unpacker config (base, if_sel, dest addr, z stride, Tile_x_dim) is set at walk start and restored
  in-stream after the END marker.

## Stage 0 probe (GSPLAT_TT_U2_PROBE=1)
Runs both paths: the old da_stage fills slot 6, the U2 path fills rows 480+, and TRISC1 compares the
first 2 records per walk word by word, plus checks that all 4 repeated DEST rows match. The first record
of every walk gets injected edge values (-0, a denormal, 0x7FFFFF, 1.0, -pi, q=0/1/65535 for 2^23+q).
DPRINT TR1 at kernel end: `U2P n= bad= raw_bad=`.

## Results
Stage 0 probe (2234282, yyzo-bh-07, 30 views): 3409 checked records, bad=0 (U2 staged values bit-equal to the
da_stage path, edge values included); raw_bad=1 word on 2 of 3409 checks (the check reads the staging rows after
SEMGET, so TRISC0 may already be overwriting them: probe race, not a staging error). U1 md5 matches 46a725ab.
Stage 1 (knob 1), 30-view bicycle, yyzo-bh-07 p100a, commit 822292d, 3 rotated rounds, same session:

| round | def ms/view | U1 ms/view | def blend | U1 blend |
|---|---|---|---|---|
| r1 | 11.972 | 15.636 | 7.480 | 11.082 |
| r2 | 11.953 | 15.590 | 7.456 | 11.094 |
| r3 | 12.089 | 15.664 | 7.464 | 11.058 |
| mean | 12.005 | 15.630 | 7.467 | 11.078 |

U1 is +3.63 ms/view slower (blend stage +3.61). md5 46a725ab for both arms in all rounds.
Tracy def/U1 did not run: the profiler build fails at program.cpp:2282 `state.offset <= max_size` (kernel
binary too big with profiling on), so there is no TRISC0/TRISC1 split or SEMWAIT share.

Decision: killed (gate: stage 1 must save >= 0.15 ms/view; it loses 3.63). Stage 2 (MOP) not built: it can
only shorten the 27 TRISC1 ops, which cannot recover a 3.6 ms loss. Knob stays default 0.

Likely cause (not measured): there is one DEST staging slot (slot 7), so the UNPACK_TO_DEST semaphore makes
TRISC0 and TRISC1 run in lockstep per record. TRISC0 cannot unpack record n+1 until TRISC1 has copied record n
out, which removes the decode-ahead overlap that iter 198 gained (-0.69), and each record now also pays 20
UNPACRs plus the semaphore round trip on the critical path. Two DEST staging slots would be needed for overlap,
and the 13 SFPLOAD/SFPSTORE copies into slot 6 remain on TRISC1 either way.
