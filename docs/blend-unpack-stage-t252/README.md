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
Pending: device chain `drive.sh` (probe smoke, 3 rotated def/U1 rounds, Tracy def/U1).
